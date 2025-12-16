'''
Description: This script performs large batch pricing and compiles a CSV with the results.
'''
import csv
import time
import itertools
from datetime import datetime
from functools import wraps
import requests
import asyncio

import numpy as np
import pandas as pd

from auxiliary_variables import USERNAME, PASSWORD, CSV_FILEPATH, EASTERN, UNIQUE_QUANTITIES, UNIQUE_TRADE_TYPES, COLUMNS_TO_KEEP, MAX_NUMBER_OF_CUSIPS_PER_BATCH, MAX_ASYNC_CALLS_PER_SERVER, NUM_SERVERS, PRINT_RETRY_MESSAGES
from auxiliary_functions import function_timer, get_api_call
from asynchronous_api_calls import price_batches as price_batches_async


def run_multiple_times_before_failing(function):
    '''This function is to be used as a decorator. It will run `function` over and over again until it does not 
    raise an Exception for a maximum of `max_runs` (specified below) times. It solves the following problem: when 
    the server is overloaded, certain requests fail. It is very similar to the decorator by the same name in 
    `app_engine/demo/server/modules/ficc/utils/auxiliary_functions.py`.'''
    @wraps(function)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
    def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
        max_runs_for_runtime_error = 5    # NOTE: setting this to 1 is the same functionality as not having this decorator
        max_runs_for_http_error = 100    # NOTE: setting this to 1 is the same functionality as not having this decorator
        runs_so_far = 0
        while runs_so_far < max(max_runs_for_runtime_error, max_runs_for_http_error):
            sleep_time = min(2 ** runs_so_far, 10)    # use exponential backoff to give a longer delay if there have been many failures
            runs_so_far += 1
            exception = None    # used to access the `e` variable in the outer scope which is defined in the `except` scope
            try:
                return function(*args, **kwargs)
            except requests.exceptions.HTTPError as e:    # catches `requests.exceptions.HTTPError` which is the error raised in `call_batch_pricing(...)`
                exception = e
                max_runs = max_runs_for_http_error
            except RuntimeError as e:    # catches `RuntimeError` which is the error raised in `call_batch_pricing(...)`
                exception = e
                max_runs = max_runs_for_runtime_error
            if runs_so_far >= max_runs:
                print(f'WARNING: Already caught {type(exception)}: {exception}, {max_runs} times in {function.__name__}, so will now raise the error')
                raise exception
            if PRINT_RETRY_MESSAGES: print(f'WARNING: Caught {type(exception)}: {exception}, and will retry {function.__name__} {max_runs - runs_so_far} more times (next run will be {sleep_time} seconds later)')
            time.sleep(sleep_time)    # have a delay to prevent overloading the server
    return wrapper


@function_timer
def get_cusips_quantities_tradetypes_from_csv(csv_filename):
    '''Get the CUSIPs from a CSV. Assumes that each line has a single CUSIP and each CUSIP is on its own line.'''
    cusips, quantities, trade_types = [], [], []
    with open(csv_filename, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            cusips.append(row[0])
            if len(row) > 1:
                quantities.append(row[1])
            if len(row) > 2:
                trade_types.append(row[2])
    assert (len(cusips) == len(quantities) == len(trade_types)) or (len(quantities) == len(trade_types) == 0), 'Currently support a CSV with the following format: (1) every row is a unique CUSIP, or (2) every row has a CUSIP, quantity, and trade type'
    print(f'Extracted {len(cusips)} CUSIPs from {csv_filename}')
    return cusips, quantities, trade_types


@function_timer
def get_cusip_list_and_quantity_list_and_trade_type_list(unique_cusips: list, unique_quantities: list, unique_trade_types: list) -> list:
    '''Using the list of CUSIPs in `unique_cusips`, populate the entire list of line items by matching each 
    CUSIP with a quantity from `unique_quantities` and a trade type from `unique_trade_types`.'''
    # all_cusips = sorted(all_cusips)
    cusips_quantites_trade_types = list(itertools.product(unique_cusips, unique_quantities, unique_trade_types))    # performs cross product of each of the lists
    return list(zip(*cusips_quantites_trade_types))    # converts this to a cusips list, a quantities list, and a trade types list; [(1, 'a'), (2, 'b'), (3, 'c')] -> [(1, 2, 3), ('a', 'b', 'c')]


def exponential_backoff(url, data, max_retries: int = 5, backoff_factor: int = 2):
    '''Perform exponential backoff to retry the function `max_retries` number of times if the API call hangs.'''
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, data=data, timeout=300)    # `timeout` argument is in seconds; should take less than 5 minutes to price a single batch
            response.raise_for_status()    # raise error for bad status codes
            break
        except requests.exceptions.Timeout as e:
            retries += 1
            wait_time = min(backoff_factor ** retries, 10)    # do not wait longer than 10 seconds
            if PRINT_RETRY_MESSAGES: print(f'WARNING: Request at {url} timed out with {type(e)}: {e}. Retrying after {wait_time} seconds. Request called with data:\n{data}')
            time.sleep(wait_time)
        except requests.RequestException as e:
            print(f'WARNING: Request failed. {type(e)}: {e}. Request called with data:\n{data}')
            break
    return response


def split_list_into_batches(lst: list, maximum_size_of_batch: int) -> list:
    return np.array_split(lst, np.ceil(len(lst) / maximum_size_of_batch))


@run_multiple_times_before_failing
def call_batch_pricing(cusip_list: list, quantity_list: list, trade_type_list: list, username: str, password: str, time: str = None) -> pd.DataFrame:
    url, data = get_api_call(cusip_list, quantity_list, trade_type_list, username, password, time=time)
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
    return priced_df


def price_batches(cusip_list_batches, quantity_list_batches, trade_type_list_batches, username: str, password: str, time: str = None) -> pd.DataFrame:
    call_batch_pricing_func = lambda cusip_list, quantity_list, trade_type_list: call_batch_pricing(cusip_list, quantity_list, trade_type_list, username, password, time)
    cusip_quantity_trade_type_batches = zip(cusip_list_batches, quantity_list_batches, trade_type_list_batches)
    priced_batches = [call_batch_pricing_func(cusip_list_batch, quantity_list_batch, trade_type_list_batch) for cusip_list_batch, quantity_list_batch, trade_type_list_batch in cusip_quantity_trade_type_batches]
    return priced_batches


@function_timer
def call_price_batches(username: str, password: str, cusip_list_batches: list, quantity_list_batches: list, trade_type_list_batches: list, time: str = None) -> list:
    num_batches = len(cusip_list_batches)
    print(f'Total number of batches: {num_batches}')
    print(f'Making asynchronous API calls (after making one non-asynchronous call) with {MAX_ASYNC_CALLS_PER_SERVER} maximum asynchronous calls for each of the {NUM_SERVERS} servers')
    last_cusip_list, last_quantity_list, last_trade_type_list = cusip_list_batches.pop(), quantity_list_batches.pop(), trade_type_list_batches.pop()
    last_batch_priced = price_batches([last_cusip_list], [last_quantity_list], [last_trade_type_list], username, password, time)    # price the last batch as a trial to make sure that credentials are set before making the asynchronous calls
    
    priced_batches = []
    num_batches_per_call = MAX_ASYNC_CALLS_PER_SERVER * NUM_SERVERS
    for batch_group_start_idx in range(0, len(cusip_list_batches), num_batches_per_call):
        priced_batches.extend(asyncio.run(price_batches_async(cusip_list_batches[batch_group_start_idx : batch_group_start_idx + num_batches_per_call], 
                                                              quantity_list_batches[batch_group_start_idx : batch_group_start_idx + num_batches_per_call], 
                                                              trade_type_list_batches[batch_group_start_idx : batch_group_start_idx + num_batches_per_call], 
                                                              username, 
                                                              password, 
                                                              time)))
    priced_batches = priced_batches + last_batch_priced
    return priced_batches


def main():
    current_datetime = datetime.now(EASTERN)
    priced_file_name_suffix = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
    current_time_string = current_datetime.strftime('%H:%M:%S')
    cusip_list, quantity_list, trade_type_list = get_cusips_quantities_tradetypes_from_csv(CSV_FILEPATH)
    if len(quantity_list) == 0: cusip_list, quantity_list, trade_type_list = get_cusip_list_and_quantity_list_and_trade_type_list(cusip_list, UNIQUE_QUANTITIES, UNIQUE_TRADE_TYPES)
    cusip_list, quantity_list, trade_type_list = list(cusip_list), list(quantity_list), list(trade_type_list)    # need to convert each of these to a list so that we can mutate them using `.pop(...)`

    last_cusip, last_quantity, last_trade_type = cusip_list.pop(), quantity_list.pop(), trade_type_list.pop()    # isolate only the last line item to be the one that is priced individually before making the asynchronous calls to create a speedup
    cusip_list_batches = split_list_into_batches(cusip_list, MAX_NUMBER_OF_CUSIPS_PER_BATCH) + [[last_cusip]]
    quantity_list_batches = split_list_into_batches(quantity_list, MAX_NUMBER_OF_CUSIPS_PER_BATCH) + [[last_quantity]]
    trade_type_list_batches = split_list_into_batches(trade_type_list, MAX_NUMBER_OF_CUSIPS_PER_BATCH) + [[last_trade_type]]

    priced_batches = call_price_batches(USERNAME, PASSWORD, cusip_list_batches, quantity_list_batches, trade_type_list_batches)
    priced_batches = pd.concat(priced_batches, ignore_index=True)
    
    print('First 10 items priced')
    print(priced_batches.head(10).to_markdown())
    print('Last 10 items priced')
    print(priced_batches.tail(10).to_markdown())

    priced_csv_filename = f'priced_{priced_file_name_suffix}.csv'
    priced_batches.to_csv(priced_csv_filename, index=False)


if __name__ == '__main__':
    main()
