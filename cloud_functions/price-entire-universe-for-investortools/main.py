'''
Description: This cloud function performs large batch pricing (>50k CUSIPs) and compiles a CSV with the results.
'''
import functions_framework

import os
import time
import itertools
from datetime import datetime
from functools import wraps
import logging as python_logging    # to not confuse with google.cloud.logging
import requests
import asyncio

import multiprocess as mp
from tqdm import tqdm

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday

from google.cloud import logging

from auxiliary_variables import TESTING, USERNAME, PASSWORD, EASTERN, QUANTITY, QUANTITY_LOWER_BOUND, QUANTITY_UPPER_BOUND, UNIQUE_QUANTITIES_FOR_INVESTOR_TOOLS, UNIQUE_TRADE_TYPES_FOR_INVESTOR_TOOLS, COLUMNS_TO_KEEP, MAX_NUMBER_OF_CUSIPS_PER_BATCH, MULTIPROCESSING, INPUT_CSV_FILENAME, GOOGLE_CLOUD_BUCKET, STORAGE_CLIENT, REFERENCE_DATA_REDIS_CLIENT
from auxiliary_functions import function_timer, get_api_call
from filtering import remove_not_outstanding_cusips
from asynchronous_api_calls import price_batches as price_batches_async
from upload_csv_to_sftp_server import upload_file_to_sftp


if TESTING:
    python_logging.info = print
    python_logging.warning = print
else:
    # set up logging client; https://cloud.google.com/logging/docs/setup/python
    logging_client = logging.Client()
    logging_client.setup_logging()


MAX_RETRIES_FOR_RUNTIME_ERROR = 3    # NOTE: setting this to 1 is the same functionality as not having this decorator
MAX_RETRIES_FOR_HTTP_ERROR = 100    # NOTE: setting this to 1 is the same functionality as not having this decorator


def run_multiple_times_before_failing(function):
    '''This function is to be used as a decorator. It will run `function` over and over again until it does not 
    raise an Exception for a maximum of `max_runs` (specified below) times. It solves the following problem: when 
    the server is overloaded, certain requests fail. It is very similar to the decorator by the same name in 
    `app_engine/demo/server/modules/ficc/utils/auxiliary_functions.py`.'''
    @wraps(function)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
    def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
        runs_so_far = 0
        while runs_so_far < max(MAX_RETRIES_FOR_RUNTIME_ERROR, MAX_RETRIES_FOR_HTTP_ERROR):
            sleep_time = min(2 ** runs_so_far, 10)    # use exponential backoff to give a longer delay if there have been many failures
            runs_so_far += 1
            exception = None    # used to access the `e` variable in the outer scope which is defined in the `except` scope
            try:
                return function(*args, **kwargs)
            except requests.exceptions.HTTPError as e:    # catches `requests.exceptions.HTTPError` which is the error raised in `call_batch_pricing(...)`
                exception = e
                max_runs = MAX_RETRIES_FOR_HTTP_ERROR
            except RuntimeError as e:    # catches `RuntimeError` which is the error raised in `call_batch_pricing(...)`
                exception = e
                max_runs = MAX_RETRIES_FOR_RUNTIME_ERROR
            if runs_so_far >= max_runs:
                python_logging.warning(f'Already caught {type(exception)}: {exception}, {max_runs} times in {function.__name__}, so will now raise the error')
                raise exception
            python_logging.warning(f'Caught {type(exception)}: {exception}, and will retry {function.__name__} {max_runs - runs_so_far} more times (next run will be {sleep_time} seconds later)')
            time.sleep(sleep_time)    # have a delay to prevent overloading the server
    return wrapper


def remove_directories_and_extension(filename: str) -> str:
    '''Remove the file extension and also remove everything before and including the rightmost slash 
    from the filename.'''
    rightmost_period = filename.rfind('.')
    filename = filename[:rightmost_period]
    rightmost_slash_pos = filename.rfind('/')
    if rightmost_slash_pos == -1: return filename    # no slash was found
    return filename[rightmost_slash_pos + 1:]


def process_quantity(user_quantity, default_quantity, is_large_batch=False):
    '''Taken directly from `app_engine/demo/server/modules/batch_pricing.py::process_quantity(...)`'''
    if user_quantity == None:    # no quantity was provided, so return `default_quantity` which has been modified if it is a large batch or not
        return default_quantity
    elif is_large_batch:    # quantity was provided by the user, but no need to modify it here, can do it in the API call
        return user_quantity
    try:    # quantity provided, and needs to be handled since there is no API call that is going to be made to handle it
        quantity = int(user_quantity) * 1000
        return max(min(quantity, QUANTITY_UPPER_BOUND * 1000), QUANTITY_LOWER_BOUND * 1000)    # if quantity is outside of the range [QUANTITY_LOWER_BOUND * 1000, QUANTITY_UPPER_BOUND * 1000], then put it back into the range
    except ValueError:    # catches the case where quantity value is not a valid integer
        return default_quantity    # `quantity` is initialized to `default_quantity` which has been modified if it is a large batch or not


def get_cusip_list_and_quantity_list_from_csv(csv_filename) -> list:
    '''Extract the CUSIPs and optionally the quantities from a CSV in `csv_filename`. `csv_filename` will 
    be the CSV itself if this cloud function is called via API.'''
    if type(csv_filename) == str:
        df = pd.read_csv(csv_filename, header=None)    # assume that there is no header row in the CSV
        cusip_list = df.iloc[:, 0].tolist()
        if len(df.columns) == 1:
            quantity_list = [QUANTITY for _ in range(len(cusip_list))]
        else:
            quantity_list = df.iloc[:, 1].tolist()
    else:
        # logic in this `else` clause is taken directly from `app_engine/demo/server/modules/batch_pricing.py::get_predictions_from_batch_pricing(...)`
        import codecs
        import csv
        reader = csv.reader(codecs.iterdecode(csv_filename, 'utf-8'))
        cusip_list, quantity_list, trade_type_list = [], [], []
        for row in reader:
            if len(row) > 0:    # first check if the row is empty before processing
                cusip_list.append(row[0].upper())    # uppercase each cusip
                quantity = row[1] if len(row) > 1 else None    # condition is True if the user has inputted a quantity
                quantity_list.append(quantity)
                input_trade_type = row[2] if len(row) > 2 else None    # condition is True if the user has inputted a trade type
                trade_type_list.append(input_trade_type)
        if len(trade_type_list) != 0: python_logging.warning('Currently do not support inputting trade_type for each line item')    # FIXME: implement this functionality
    quantity_list = [process_quantity(quantity, QUANTITY) for quantity in quantity_list]    # override `quantity_list`
    return cusip_list, quantity_list


def exponential_backoff(url, data, max_retries: int = 5, backoff_factor: int = 2):
    '''Perform exponential backoff to retry the function `max_retries` number of times if the API call hangs.'''
    retries = 0
    timeout = 300    # in seconds
    response = None    # initialization to avoid `UnbondLocalError` in case all retries fail
    while retries < max_retries:
        try:
            response = requests.post(url, data=data, timeout=timeout)    # `timeout` argument is in seconds; should take less than 5 minutes to price a single batch
            response.raise_for_status()    # raise error for bad status codes
            break
        except (requests.exceptions.Timeout, requests.RequestException) as e:    # retrying in the event of `requests.RequestException` to handle GCP transient issues
            retries += 1
            wait_time = min(backoff_factor ** retries, 10)    # do not wait longer than 10 seconds
            
            error_message_prefix = ''
            if isinstance(e, requests.exceptions.Timeout): error_message_prefix = f'Request timed out (did not complete within {timeout} seconds)'
            if isinstance(e, requests.RequestException): error_message_prefix = 'Request failed'
            print(f'{error_message_prefix}  inside main.py::exponential_backoff(...). {type(e)}: {e}. Retrying after {wait_time} seconds. Request called with data:\n{data}')
            
            time.sleep(wait_time)
        except Exception as e:
            print(f'Request failed. {type(e)}: {e}. Request called with data:\n{data}')
            break
    if response is None: print(f'Max retries of {max_retries} exceeded inside main.py::exponential_backoff(...).')
    return response


@run_multiple_times_before_failing
def call_batch_pricing(cusip_list: list, quantity_list: list, trade_type_list: list, username: str, password: str) -> pd.DataFrame:
    url, data = get_api_call(cusip_list, quantity_list, trade_type_list, username, password)
    if TESTING:
        print(f'Sending the following form to the POST request at: {url}')    # using `tqdm.write(...)` was supposed to maintain the progress bar even with print statements but is instead causing errors in `functions-framework`
        print(data)    # using `tqdm.write(...)` was supposed to maintain the progress bar even with print statements but is instead causing errors in `functions-framework`
    response = exponential_backoff(url, data)

    if not response.ok: raise RuntimeError('`response.ok` was `False`')    # raise error instead of printing the message to trigger the retry from the decorator: `run_multiple_times_before_failing`
    try:
        response_json = response.json()
    except Exception:
        raise RuntimeError(f'unable to call `response.json()` though `response.ok` is `True`')    # raise error instead of printing the message to trigger the retry from the decorator: `run_multiple_times_before_failing`
    
    try:
        priced_df = pd.read_json(response_json)[COLUMNS_TO_KEEP]
    except Exception:
        if 'error' in response_json and response_json['error'] == 'You have been logged out due to a period of inactivity. Refresh the page!':
            raise requests.exceptions.HTTPError('JSON contained the following error message: You have been logged out due to a period of inactivity. Refresh the page!')
        else:
            raise RuntimeError(f'unable to call `pd.read_json(...)` on the response even though `response.ok` is `True`. Running `response.json()` provides:\n{response_json}')    # raise error instead of printing the message to trigger the retry from the decorator: `run_multiple_times_before_failing`
    priced_df['quantity'] = priced_df['quantity'] // 1000    # requested by InvestorTools
    return priced_df


def price_batches(cusip_list_batches, quantity_list_batches, trade_type_list_batches, username: str, password: str) -> pd.DataFrame:
    call_batch_pricing_func = lambda cusip_list, quantity_list, trade_type_list: call_batch_pricing(cusip_list, quantity_list, trade_type_list, username, password)
    cusip_quantity_trade_type_batches = zip(cusip_list_batches, quantity_list_batches, trade_type_list_batches)
    num_batches = len(cusip_list_batches)
    if MULTIPROCESSING and num_batches > os.cpu_count():
        print(f'Using multiprocessing with {os.cpu_count()} cores inside `price_batches(...)`')
        with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
            priced_batches = pool_object.starmap(call_batch_pricing_func, cusip_quantity_trade_type_batches)    # need to use starmap since `upload_trade_history_to_trade_history_redis` has multiple arguments: https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
    else:
        print(f'Not using multiprocessing inside `price_batches(...)`')
        priced_batches = [call_batch_pricing_func(cusip_list_batch, quantity_list_batch, trade_type_list_batch) for cusip_list_batch, quantity_list_batch, trade_type_list_batch in tqdm(cusip_quantity_trade_type_batches, disable=num_batches == 1)]
    return priced_batches


def split_list_into_batches(lst: list, maximum_size_of_batch: int) -> list:
    return np.array_split(lst, np.ceil(len(lst) / maximum_size_of_batch))


def write_lists_to_error_file(cusip_list: list, quantity_list: list, filename: str = 'error.txt') -> str:
    '''Save `cusip_list` and `quantity_list` to a file with filename `filename`. Return `filename`.'''
    with open(filename, 'w') as file:
        file.write('cusip_list')
        file.write(str(cusip_list))
        file.write('quantity_list')
        file.write(str(quantity_list))
    return filename


def upload_to_storage(file_name: str, file_text, bucket_name: str = GOOGLE_CLOUD_BUCKET):
    bucket = STORAGE_CLIENT.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(file_text)
    print(f'File {file_name} uploaded to in {bucket_name}')


def upload_dataframe_as_csv_to_storage(df: pd.DataFrame, csv_filename: str, bucket_name: str = GOOGLE_CLOUD_BUCKET, blob_filename: str = None):
    '''Upload dataframe `df` to Google Cloud bucket `bucket_name` with filename `filename`.'''
    df.to_csv(csv_filename, index=False)
    if not TESTING:
        bucket = STORAGE_CLIENT.get_bucket(bucket_name)
        if blob_filename is None: blob_filename = csv_filename    # to change the filename in the bucket, change `blob_filename` here; this line removes confusion on what `filename` refers to
        blob = bucket.blob(blob_filename)
        blob.upload_from_filename(csv_filename)
        print(f'File {csv_filename} uploaded to {blob_filename} in {bucket_name}')
    return csv_filename


def get_username_and_password_from_request(request):
    # username = request.form.get('username') if request is not None and 'username' in request else USERNAME
    # password = request.form.get('password') if request is not None and 'password' in request else PASSWORD
    return USERNAME, PASSWORD


def get_cusip_list_and_quantity_list_from_file_in_request(request):
    '''This function is currently not used but may be used in the future if we allow API calls to 
    run this cloud function.'''
    csv_file = None
    if 'file' in request.files:    # file was passed in
        csv_file = request.files['file']

    input_csv = csv_file if csv_file is not None else INPUT_CSV_FILENAME
    return get_cusip_list_and_quantity_list_from_csv(input_csv)


def get_cusip_list_and_quantity_list_and_trade_type_list_from_redis(unique_quantities: list, unique_trade_types: list) -> list:
    '''Get all the CUSIPs from the reference data redis, and populate the entire list by matching each 
    CUSIP with a quantity from `unique_quantities` and a trade type from `unique_trade_types`.'''
    all_cusips = REFERENCE_DATA_REDIS_CLIENT.keys('*')
    all_cusips = [cusip.decode('utf-8') for cusip in all_cusips]
    all_cusips = sorted(all_cusips)
    if TESTING:
        all_cusips = all_cusips[:10000000]    # if the slice number is greater than 1.6M, then we are including all of the CUSIPs
        print(f'Pricing the following {len(all_cusips)} CUSIPs:\n{all_cusips}')
    all_cusips = remove_not_outstanding_cusips(all_cusips)
    cusips_quantites_trade_types = list(itertools.product(all_cusips, unique_quantities, unique_trade_types))    # performs cross product of each of the lists
    return list(zip(*cusips_quantites_trade_types))    # converts this to a cusips list, a quantities list, and a trade types list; [(1, 'a'), (2, 'b'), (3, 'c')] -> [(1, 2, 3), ('a', 'b', 'c')]


def bid_and_offer_on_same_line(df: pd.DataFrame) -> pd.DataFrame:
    '''Put bid and offer side pricing on the same line. Assumes that each CUSIP priced has a bid and an offer side price and ytw.'''
    bid_side = df[df['trade_type'] == 'Bid Side'].drop(columns=['trade_type'])
    offered_side = df[df['trade_type'] == 'Offered Side'].drop(columns=['trade_type'])
    assert len(bid_side) + len(offered_side) == len(df), f'During filtering of bid side and offered side, {len(df) - len(bid_side) + len(offered_side)} line items went missing'
    assert sorted(bid_side['cusip'].values) == sorted(offered_side['cusip'].values), f'CUSIPs are not the same between the bid side and the offered side'
    return pd.merge(bid_side, offered_side, on='cusip', suffixes=('_bid_side', '_offered_side'))


@function_timer
def call_price_batches(username: str, password: str, cusip_list_batches: list, quantity_list_batches: list, trade_type_list_batches: list) -> list:
    num_batches = len(cusip_list_batches)
    print(f'Total number of batches: {num_batches}')
    if num_batches > os.cpu_count():
        max_async_calls_at_once = 200
        print(f'Making asynchronous API calls (after making one non-asynchronous call) with {max_async_calls_at_once} maximum asynchronous calls at once')
        last_cusip_list, last_quantity_list, last_trade_type_list = cusip_list_batches.pop(), quantity_list_batches.pop(), trade_type_list_batches.pop()
        last_batch_priced = price_batches([last_cusip_list], [last_quantity_list], [last_trade_type_list], username, password)    # price the last batch as a trial to make sure that credentials are set before making the asynchronous calls
        
        priced_batches = []
        for batch_group_start_idx in range(0, len(cusip_list_batches), max_async_calls_at_once): 
            priced_batches.extend(asyncio.run(price_batches_async(cusip_list_batches[batch_group_start_idx : batch_group_start_idx + max_async_calls_at_once], 
                                                                  quantity_list_batches[batch_group_start_idx : batch_group_start_idx + max_async_calls_at_once], 
                                                                  trade_type_list_batches[batch_group_start_idx : batch_group_start_idx + max_async_calls_at_once], 
                                                                  username, 
                                                                  password)))
        priced_batches = priced_batches + last_batch_priced
    else:
        print('Not making asynchronous calls')
        priced_batches = price_batches(cusip_list_batches, quantity_list_batches, trade_type_list_batches, username, password)
    return priced_batches


class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]


def today_is_a_holiday() -> bool:
    '''Determine whether today is a US national holiday.'''
    now = datetime.now(EASTERN)
    today = pd.Timestamp(now).tz_localize(None).normalize()    # `.tz_localize(None)` is to remove the time zone; `.normalize()` is used to remove the time component from the timestamp
    current_year = now.year
    holidays_in_last_year_and_next_year = set(USHolidayCalendarWithGoodFriday().holidays(start=f'{current_year - 1}-01-01',end=f'{current_year + 1}-12-31'))
    if today in holidays_in_last_year_and_next_year:
        python_logging.info(f'Today, {today}, is a national holiday, and so we will not perform large batch pricing, and so there will not be any files in the SFTP')
        return True
    return False


def successfully_priced(priced_batch) -> bool:
    '''Determines whether `priced_batch` is a successfully priced batch. Most reliable way 
    to check if `priced_batch` is a successfully priced batch is to check whether it is a 
    `pd.DataFrame` instance. If `priced_batch` is unsuccessfully priced, it may be an error 
    instance or a `None` value.'''
    return isinstance(priced_batch, pd.DataFrame)


def main(request=None, username: str = None, password: str = None) -> str:
    if today_is_a_holiday(): return 'SUCCESS'    # do not perform large batch pricing on a national holiday; this saves money since we do not need to use Google Cloud Run instances
    used_csv_file = False
    if username is None and password is None: username, password = get_username_and_password_from_request(request)    # used to make the API call to batch pricing
    
    # cusip_list, quantity_list, trade_type_list = get_cusip_list_and_quantity_list_from_file(request)
    # used_csv_file = True

    cusip_list, quantity_list, trade_type_list = get_cusip_list_and_quantity_list_and_trade_type_list_from_redis(UNIQUE_QUANTITIES_FOR_INVESTOR_TOOLS, UNIQUE_TRADE_TYPES_FOR_INVESTOR_TOOLS)
    cusip_list, quantity_list, trade_type_list = list(cusip_list), list(quantity_list), list(trade_type_list)    # need to convert each of these to a list so that we can mutate them using `.pop(...)`

    last_cusip, last_quantity, last_trade_type = cusip_list.pop(), quantity_list.pop(), trade_type_list.pop()    # isolate only the last line item to be the one that is priced individually before making the asynchronous calls to create a speedup
    cusip_list_batches = split_list_into_batches(cusip_list, MAX_NUMBER_OF_CUSIPS_PER_BATCH) + [[last_cusip]]
    quantity_list_batches = split_list_into_batches(quantity_list, MAX_NUMBER_OF_CUSIPS_PER_BATCH) + [[last_quantity]]
    trade_type_list_batches = split_list_into_batches(trade_type_list, MAX_NUMBER_OF_CUSIPS_PER_BATCH) + [[last_trade_type]]
    priced_batches = call_price_batches(username, password, cusip_list_batches, quantity_list_batches, trade_type_list_batches)

    # iterate through each of the priced batches and output CUSIPs and other information on a batch that was not successfully priced for easier debugging
    NUM_CUSIPS_PER_PRINT_STATEMENT = 500    # avoids Python truncation of very long lists in print output
    for priced_batch, cusip_list_batch, quantity_list_batch, trade_type_list_batch in zip(priced_batches, cusip_list_batches, quantity_list_batches, trade_type_list_batches):
        if not successfully_priced(priced_batch):
            # write_lists_to_error_file(cusip_list_batch, quantity_list_batch)
            python_logging.warning(f'A batch was unable to be priced. The CUSIPs in the priced batch (printing in chunks of {NUM_CUSIPS_PER_PRINT_STATEMENT} to avoid truncation in print output):')
            for idx in range(0, len(cusip_list_batch), NUM_CUSIPS_PER_PRINT_STATEMENT):
                python_logging.warning(cusip_list_batch[idx : idx + NUM_CUSIPS_PER_PRINT_STATEMENT])
    
    successful_batches, failed_batches = [], []
    for priced_batch in priced_batches:
        if successfully_priced(priced_batch):
            successful_batches.append(priced_batch)
        else:
            failed_batches.append(priced_batch)
    priced_batches = pd.concat(successful_batches, ignore_index=True)
    num_batches_failed = len(failed_batches)
    # priced_batches = bid_and_offer_on_same_line(priced_batches)

    print('First 10 items priced')
    print(priced_batches.head(10).to_markdown())
    if len(priced_batches) > 10:
        print('Last 10 items priced')
        print(priced_batches.tail(10).to_markdown())
    
    input_csv_filename = remove_directories_and_extension(INPUT_CSV_FILENAME) if used_csv_file else str(len(priced_batches))
    current_datetime_string = datetime.now(EASTERN).strftime('%Y-%m-%d--%H-%M-%S')
    current_date_string = current_datetime_string[:10]    # the first 10 characters of `current_datetime_string` corresponds to the date
    output_csv_filename = f'priced_{current_datetime_string}_{input_csv_filename}.csv'
    upload_dataframe_as_csv_to_storage(priced_batches, output_csv_filename, GOOGLE_CLOUD_BUCKET, f'{current_date_string}/{output_csv_filename}')    # store the priced file in a folder where the name of the folder is the date
    if not TESTING: upload_file_to_sftp(output_csv_filename)
    if num_batches_failed > 0: raise RuntimeError(f'{num_batches_failed} batches failed out of {len(successful_batches) + num_batches_failed} total batches')
    return 'SUCCESS'


@functions_framework.http
def hello_http(request) -> str:
    '''HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'name' in request_json:
        name = request_json['name']
    elif request_args and 'name' in request_args:
        name = request_args['name']
    else:
        name = 'World'
    return 'Hello {}!'.format(name)
    
    Price a list of CUSIPs with corresponding quantities for trade type `TRADE_TYPE` and 
    store the result in a CSV with name `output_csv_filename` in Google Cloud bucket with  
    name `GOOGLE_CLOUD_BUCKET`. Return the location of the output file.'''
    return main(request)


if __name__ == '__main__':    # runner for testing convenience
    main(username=USERNAME, password=PASSWORD)
