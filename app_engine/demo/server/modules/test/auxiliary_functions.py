'''
'''
import time
import os
import csv
from functools import wraps
import requests
import pandas as pd

from google.cloud import bigquery

from modules.get_creds import get_creds

from modules.test.auxiliary_variables import USERNAME, PASSWORD, QUANTITY, TRADE_TYPE, DIRECTORY, LOGGED_OUT_MESSAGE, LoggedOutError

from modules.auxiliary_variables import FEATURES_FOR_OUTPUT_CSV, NUMERICAL_ERROR, DOLLAR_PRICE_MODEL_DISPLAY_TEXT, ACCEPTABLE_DISPLAY_TYPES_FOR_YIELD_CURVE
from modules.exclusions import CUSIP_ERROR_MESSAGE


REQUEST_URL = 'http://localhost:5000'    # for testing locally, use: 'http://localhost:5000' and for testing the server as is deployed, use 'https://api.ficc.ai' or 'https://server-3ukzrmokpq-uc.a.run.app'


def run_multiple_times_before_failing(test_function):
    '''This function is to be used as a decorator. It will run `test_function` over and over again until it hits 
    a success for a maximum of `max_runs` (specified below) times. It solves the following problem: sometimes 
    the test fails randomly, but when the exact code is run again, it succeeds. In this case, we want to make 
    sure that we do not raise a false alarm that the product is broken, when it was a random test failure.'''
    @wraps(test_function)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
    def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
        max_runs = 3    # NOTE: max_runs = 1 is the same functionality as not having this decorator
        while max_runs > 0:
            try:
                test_function(*args, **kwargs)
                return None
            except (AssertionError, requests.exceptions.ConnectionError) as e:
                max_runs -= 1
                if max_runs == 0: raise e
                time.sleep(1)    # have a one second delay to prevent overloading the server
    return wrapper


def run_multiple_times_if_logged_out(function):
    '''This function is to be used as a decorator. It will run `function` over and over again until it 
    hits a success for a maximum of `max_runs` (specified below) times. It solves the following problem:  
    sometimes the API call fails randomly, but when the exact call is made again, it succeeds. In this 
    case, we want to make sure that we do not raise a false alarm that the product is broken, when it was 
    an API call that just happened to fail randomly.
    TODO: figure out why the API calls are not stable.'''
    @wraps(function)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
    def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
        max_runs = 100    # NOTE: max_runs = 1 is the same functionality as not having this decorator
        while max_runs > 0:
            try:
                return function(*args, **kwargs)
            except LoggedOutError as e:
                max_runs -= 1
                if max_runs == 0: raise e
                time.sleep(1)    # have a one second delay to prevent overloading the server
    return wrapper


def get_bq_client():
    '''Initialize the credentials and the bigquery client.'''
    get_creds()
    return bigquery.Client()


@run_multiple_times_if_logged_out
def response_from_individual_pricing(cusip, trade_type, quantity, return_execution_time=False):
    '''Individually prices `cusip` at `trade_type` and `quantity`.'''
    start_time = time.time()
    request_obj = requests.get(f'{REQUEST_URL}/api/price?cusip={cusip}&tradeType={trade_type}&amount={quantity}&username={USERNAME}&password={PASSWORD}')
    time_elapsed = round(time.time() - start_time, 3)    # round the value in order to get a readable error statement
    response_dict = request_obj.json()
    if 'error' in response_dict and response_dict['error'] == LOGGED_OUT_MESSAGE: raise LoggedOutError
    return (response_dict, time_elapsed) if return_execution_time else response_dict


def get_filename_from_cusip_list(cusip_list):
    '''Create a filename based on `cusip_list`. Only use the first few CUSIPs in case the 
    list is long in order to not get a `File name too long` error.'''
    cusip_list_as_string = '_'.join(cusip_list[:3]) if type(cusip_list) == list else cusip_list
    return f'{DIRECTORY}/{cusip_list_as_string}.csv'


def get_cusip_list_quantity_list_tradetype_list(filename):
    '''Get the list of CUSIPs and list of quantities and trade types from `filename` where we 
    assume that the first column is the list of CUSIPs, the second column is the list of quantities, 
    and the third column is the list of trade types.'''
    cusip_list, quantity_list, trade_type_list = [], [], []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:    # `for` loop is very similar to logic in `get_predictions_from_batch_pricing(...)`
            if len(row) > 0:    # only process the row if it is not empty
                cusip_list.append(row[0].upper())    # uppercase each cusip

                quantity = QUANTITY
                if len(row) > 1:    # user has inputted a quantity
                    user_quantity = row[1]
                    try:
                        quantity = int(user_quantity)    # multiplying by 1000 happens in `process_quantity` on the server
                    except ValueError:    # catches the case where a quantity value is not a valid integer
                        pass    # `quantity` is initialized to `default_quantity`
                quantity_list.append(quantity)

                trade_type = TRADE_TYPE
                if len(row) > 2:    # user has inputted a quantity
                    user_trade_type = row[2]
                    if user_trade_type in ('S', 'P', 'D'): trade_type = user_trade_type
                trade_type_list.append(trade_type)

    assert len(cusip_list) == len(quantity_list) == len(trade_type_list), '`cusip_list`, `quantity_list`, and `trade_type_list` do not have the same number of items'
    return cusip_list, quantity_list, trade_type_list


def check_if_string_value_can_be_represented_as_a_number(string, name=None):
    '''Checks whether `string` which is a string type, throws an error when attempting to represent 
    it as a float.'''
    if name is None: name = 'input'
    try:    # check if `string` is a number
        string = float(string)
    except ValueError as e:    # `ValueError: could not convert string to float` arises when attempting to convert a string to a float that cannot be represented as a number
        print(e)
        assert False, f'{name} should be a numerical value, but was instead {string}'


def check_if_string_value_cannot_be_represented_as_a_number(string, name=None):
    '''Checks whether `string` which is a string type, does not throw an error when attempting to 
    represent it as a float.'''
    if name is None: name = 'input'
    try:    # check if `string` is a number
        string = float(string)
    except ValueError:    # `ValueErrorcould not convert string to float` arises when attempting to convert a string to a float that cannot be represented as a number
        pass
    else:
        assert False, f'{name} should not have been able to be converted to a number, but was and had the value {string}'


@run_multiple_times_if_logged_out
def response_from_batch_pricing(filename, cusip_list=None, quantity_list=None, trade_type_list=None, trade_type=TRADE_TYPE, return_cusip_list_and_quantity_list=False, return_execution_time=False, create_file: bool = True):
    '''Batch prices `cusip_list` by putting it into the csv `filename` where the corresponding 
    quantities are in `quantity_list` and trade types are in `trade_type_list`. If no `trade_type_list` 
    is provided, then default to `trade_type`. If `create_file` is `True`, then the test creates a CSV 
    and uses the CSV to make the post request, otherwise the call is made with additional arguments in  
    the form when making the post request.'''
    if cusip_list is None:    # if `cusip_list` is `None`, then assume that the file at `filename` already has all of the CUSIP data
        cusip_list, quantity_list, trade_type_list = get_cusip_list_quantity_list_tradetype_list(filename)
        filename_extension_pos = filename.find('.')
        assert filename_extension_pos != -1, 'Could not find "." in the filename which is necessary for us to modify the filename since we need to separate the filename from the extension'
        filename_wo_extension, extension = filename[:filename_extension_pos], filename[filename_extension_pos:]
        filename = filename_wo_extension + '_pricing' + extension    # modify the filename to be used for pricing
    if type(cusip_list) != list:    # `cusip_list` is a single CUSIP
        cusip_list = [cusip_list]
        if quantity_list is not None:    # `quantity_list` should be a single quantity
            quantity_list = [quantity_list]
        if trade_type_list is not None:    # `trade_type_list` should be a single quantity
            trade_type_list = [trade_type_list]
    
    if quantity_list is None and trade_type_list is not None: raise NotImplementedError('No support for batch pricing with trade types without quantities')    # TODO: implement this feature
    if trade_type_list is None: trade_type_list = [trade_type] * len(cusip_list)
    
    if create_file:
        array_for_csv = cusip_list if quantity_list is None else list(zip(cusip_list, quantity_list, trade_type_list))    # group each cusips with its corresponding quantity
        pd.DataFrame(array_for_csv).to_csv(filename, header=None, index=None)
        data = {'username': USERNAME, 'password': PASSWORD, 'amount': QUANTITY, 'tradeType': trade_type}    # note that `amount` only fills in values that are empty
        file = {'file': open(filename, 'rb')}
    else:
        data =  {'username': USERNAME, 'password': PASSWORD, 'amount': QUANTITY, 'tradeType': trade_type, 'cusipList': cusip_list, 'quantityList': quantity_list, 'tradetypeList': trade_type_list}
        file = dict()
    
    request_ref = f'{REQUEST_URL}/api/batchpricing'
    try:    # perform a try...except so that the file can be removed even if an error is thrown
        start_time = time.time()
        request_obj = requests.post(request_ref, data=data, files=file)
        time_elapsed = round(time.time() - start_time, 3)    # round the value in order to get a readable error statement
    except Exception as e:
        os.remove(filename)
        raise e
    
    return_values = request_obj if return_cusip_list_and_quantity_list is False else (request_obj, cusip_list, quantity_list)
    return_values = (return_values, time_elapsed) if return_execution_time else return_values
    try:
        response_dict = request_obj.json()
    except Exception:    # unable to convert `request_obj` to a response dict
        if create_file: os.remove(filename)
        return return_values
    if 'error' in response_dict and response_dict['error'] == LOGGED_OUT_MESSAGE:
        if create_file: os.remove(filename)
        raise LoggedOutError
    if create_file: os.remove(filename)
    return return_values


def _get_cusip_to_count_dict(cusip_list):
    '''Returns a dictionary mapping each CUSIP in `cusip_list` to the number of occurences in `cusip_list`.'''
    if type(cusip_list) != list:    # `cusip_list` is a single CUSIP
        cusip_list = [cusip_list]
    cusip_count = dict()    # store the count of each CUSIP in `cusip_list`
    for cusip in cusip_list:
        cusip_count[cusip] = cusip_count.get(cusip, 0) + 1    # second argument of .get(...) provides a default value if the key is not found in the map
    return cusip_count


def get_spreadsheet_as_list(request_obj, header_columns: list = FEATURES_FOR_OUTPUT_CSV, spreadsheet_returned_as_json_string: bool = False):
    '''Takes the request object, `request_obj` and returns the content of the spreadsheet as a list of lists 
    where each sublist contains the values of one row of the spreadsheet. Assumes that `request_obj` is a 
    successful response from batch pricing.'''
    assert request_obj.ok, f'Unsuccessful response with status code: {request_obj.status_code} and message: {request_obj.text}'    # successful response; checks whether the status_code is less than 400

    if spreadsheet_returned_as_json_string:
        df = pd.read_json(request_obj.json())
        header = df.columns.tolist()
        content = df.values.tolist()
    else:
        content = request_obj.content.decode('utf-8')
        content = content.split('\n')
        header = content[0]
        header = header.split(',')
        content = content[1:-1]    # first row is the column names, last row is an empty line
        content = [row.split(',') for row in content]

    assert header_columns == header, f'The header of the spreadsheet should be {header_columns} but is instead {header}'    # column names in predictions csv are correct
    return content


def check_that_batch_pricing_gives_output_for_all_cusips(request_obj, cusip_list, check_quantity_equals_default=True, check='output', reason_for_dollar_price_model=None, error_code=None, post_request_made_with_file: bool = True):
    '''Check that each CUSIP in `cusip_list` has some output in the request_obj. `check` can be one of five values: 
    (1) 'output', (2) 'success', (3) 'failure', (4), 'price_only', (5) 'dollar_price_model_used'. If `check` is 'output', 
    then no additional checks are made with regards to the the price and ytw predictions. If `check` is 'success', then 
    check that the price and ytw predictions do not resemble errors, i.e., the predicted price and predicted ytw are valid. 
    If `check` is 'failure', then check that the price and ytw predictions resemble errors. If `check` is 'price_only', 
    then check that the price prediction does not resemble an error. If `check` is 'dollar_price_model_used', then check 
    that the price is a numerical value and the ytw predictions are string messages, since we currently do not predict a 
    yield when using the dollar price model. If `error_code` is not `None` and `check` is 'failure', then check that the 
    error message corresponds to that of `error_code`. If `call_made_with_api` is `True`, then the output of the request 
    is a JSON string, and so when calling `get_spreadsheet_as_list`, the optional argument of `spreadsheet_returned_as_json_string` 
    must be set to `True`.'''
    content = get_spreadsheet_as_list(request_obj, spreadsheet_returned_as_json_string=not post_request_made_with_file)

    cusip_count = _get_cusip_to_count_dict(cusip_list)
    cusip_idx = FEATURES_FOR_OUTPUT_CSV.index('cusip')
    quantity_idx = FEATURES_FOR_OUTPUT_CSV.index('quantity')

    if check in ('success', 'failure', 'price_only', 'dollar_price_model_used'):
        price_idx = FEATURES_FOR_OUTPUT_CSV.index('price')
        ytw_idx = FEATURES_FOR_OUTPUT_CSV.index('ytw')
    if check == 'failure': ytw_date_idx = FEATURES_FOR_OUTPUT_CSV.index('yield_to_worst_date')

    if check == 'dollar_price_model_used' and reason_for_dollar_price_model is not None:
        reason_for_dollar_price_model = DOLLAR_PRICE_MODEL_DISPLAY_TEXT[reason_for_dollar_price_model]
    
    for idx, row in enumerate(content):
        cusip = row[cusip_idx]
        if check in ('success', 'failure', 'price_only', 'dollar_price_model_used'):
            price = row[price_idx]
            ytw = row[ytw_idx]

        assert cusip in cusip_count, f'CUSIP {cusip} is missing (or not with the correct frequency) from the returned spreadsheet'    # check that the CUSIP is one of those in the original cusip_list passed into batch pricing
        cusip_count[cusip] -= 1
        if cusip_count[cusip] == 0: cusip_count.pop(cusip)    # remove CUSIP from map if count equals 0
        if check_quantity_equals_default: assert str(row[quantity_idx]) == str(QUANTITY * 1000), f'The quantity in the spreadsheet {row[quantity_idx]} should equal the quantity passed in {QUANTITY * 1000}'    # check that the corresponding quantity for each CUSIP is 500000

        if check in ('success', 'price_only'): assert float(price) != NUMERICAL_ERROR, f'For CUSIP {cusip_list[idx]}, the predicted price {price} should not be equal to {NUMERICAL_ERROR}'    # check that the price is not the value used for numerical error (indicating that an actual price was provided)
        if check == 'success': assert float(ytw) != NUMERICAL_ERROR, f'For CUSIP {cusip_list[idx]}, the predicted ytw {ytw} should not be equal to {NUMERICAL_ERROR}'    # check that the ytw is not the value used for numerical error (indicating that an actual price was provided)

        if check == 'failure':
            assert float(price) == NUMERICAL_ERROR, f'For CUSIP {cusip_list[idx]}, the predicted price {price} should be equal to {NUMERICAL_ERROR}'    # check that the price is not the value used for numerical error (indicating that an actual price was provided)
            assert float(ytw) == NUMERICAL_ERROR, f'For CUSIP {cusip_list[idx]}, the predicted ytw {ytw} should be equal to {NUMERICAL_ERROR}'    # check that the ytw is not the value used for numerical error (indicating that an actual price was provided)
            if error_code is not None:
                assert error_code in CUSIP_ERROR_MESSAGE, f'Error code of {error_code} is invalid (not in `CUSIP_ERROR_MESSAGE` dictionary)'
                assert row[ytw_date_idx] == CUSIP_ERROR_MESSAGE[error_code], f'Row should have had an error of {CUSIP_ERROR_MESSAGE[error_code]}, but instead had an error of {row[ytw_date_idx]}'

        if check == 'dollar_price_model_used':
            check_if_string_value_can_be_represented_as_a_number(price, 'price')
            assert float(price) != NUMERICAL_ERROR, f'For CUSIP {cusip_list[idx]}, the predicted price {price} should not be equal to {NUMERICAL_ERROR}'    # check that the price is not the value used for numerical error (indicating that an actual price was provided)
            if reason_for_dollar_price_model is not None:
                assert ytw == reason_for_dollar_price_model, f'YTW value should be {reason_for_dollar_price_model}, but was instead {ytw}'
            else:
                check_if_string_value_cannot_be_represented_as_a_number(ytw, 'YTW')
    
    assert len(cusip_count) == 0, f'Spreadsheet has extra CUSIPs'    # checks that all CUSIPs were found in the predictions csv


def check_that_batch_pricing_gives_price_for_all_cusips(request_obj, cusip_list, check_quantity_equals_default=True, post_request_made_with_file: bool = True):
    '''Check that each CUSIP in `cusip_list` has a valid price and ytw in the request_obj.'''
    check_that_batch_pricing_gives_output_for_all_cusips(request_obj, cusip_list, check_quantity_equals_default, 'success', post_request_made_with_file=post_request_made_with_file)


def check_that_batch_pricing_gives_price_but_maybe_not_yield_for_all_cusips(request_obj, cusip_list, check_quantity_equals_default=True, post_request_made_with_file: bool = True):
    '''Check that each CUSIP in `cusip_list` has a valid price in the request_obj. No check is made on the ytw.'''
    check_that_batch_pricing_gives_output_for_all_cusips(request_obj, cusip_list, check_quantity_equals_default, 'price_only', post_request_made_with_file=post_request_made_with_file)


def check_that_batch_pricing_gives_price_and_dollar_price_used_for_all_cusips(request_obj, cusip_list, check_quantity_equals_default=True, reason_for_dollar_price_model=None, post_request_made_with_file: bool = True):
    '''Check that each CUSIP in `cusip_list` has a valid price and ytw in the request_obj.'''
    check_that_batch_pricing_gives_output_for_all_cusips(request_obj, cusip_list, check_quantity_equals_default, 'dollar_price_model_used', reason_for_dollar_price_model, post_request_made_with_file=post_request_made_with_file)


def check_that_batch_pricing_gives_error_for_all_cusips(request_obj, cusip_list, check_quantity_equals_default=True, error_code=None, post_request_made_with_file: bool = True):
    '''Check that each CUSIP in `cusip_list` has some error outputs in the request_obj.'''
    check_that_batch_pricing_gives_output_for_all_cusips(request_obj, cusip_list, check_quantity_equals_default, 'failure', error_code=error_code, post_request_made_with_file=post_request_made_with_file)


@run_multiple_times_if_logged_out
def response_from_similar_bonds(**kwargs):
    '''Get similar bonds for `cusip` for particular `kwargs`.'''
    if 'rating' in kwargs and '+' in kwargs['rating']:
        kwargs['rating'] = kwargs['rating'][:-1] + '%2B'    # replace the '+' sign at the end of the string with %2B since '+' is a reserved character for URL's: https://stackoverflow.com/questions/5450190/how-to-encode-the-plus-symbol-in-a-url; look in `getSimilarBonds(...)` in `src/services/priceService.js` for same correction done when calling the URL from the front end
    default_values = {'minCoupon': 0, 
                      'maxCoupon': 1000,    # default value needs to be in 1/100th's 
                      'desc': '', 
                      'yield': '', 
                      'price': '', 
                      'realtime': 'previous_day', 
                      'issuerChoice': 'any_issuer', 
                      'minMaturityDate': '2025-01-01', 
                      'maxMaturityDate': '2125-12-31', 
                      'amount': 'undefined', 
                      'userTriggered': True}
    for key in default_values:
        if key not in kwargs:
            kwargs[key] = default_values[key]
    args_string = '&'.join([f'{arg_name}={arg_value}' for arg_name, arg_value in kwargs.items()])
    request_obj = requests.get(f'{REQUEST_URL}/api/getsimilarbonds?username={USERNAME}&password={PASSWORD}&' + args_string)
    response_dict = request_obj.json()
    if 'error' in response_dict and response_dict['error'] == LOGGED_OUT_MESSAGE: raise LoggedOutError
    return response_dict


def check_if_response_from_similar_bonds_when_individually_priced_is_successful(cusip):
    '''Tests that getting similar bonds after individual pricing `cusip` is successful.'''
    response_dict = response_from_individual_pricing(cusip, TRADE_TYPE, QUANTITY)
    assert 'error' not in response_dict, f'For CUSIP {cusip}, quantity (in thousands) {QUANTITY}, and trade type {TRADE_TYPE}, pricing should not have an error, but has an error: {response_dict["error"]}'
    response_dict = response_dict[0]
    # mapping between the arg name in the response from individual pricing and the one needed for the similar bonds http request
    response_arg_to_url_arg = {'incorporated_state_code': 'state', 
                               'rating': 'rating', 
                               'purpose_class': 'purposeClass', 
                               'ficc_ytw': 'yield', 
                               'price': 'price'}
    response_dict = response_from_similar_bonds(cusip=cusip, **{url_arg: response_dict[response_arg] for response_arg, url_arg in response_arg_to_url_arg.items()})
    assert 'error' not in response_dict, f'For CUSIP {cusip}, getting similar bonds should not have an error, but has an error: {response_dict["error"]}'
    return response_dict


def _get_current_date_and_time_if_none(date, time):
    '''If `date` is `None`, then set `date` to be the current date. If `time` is `None`, then set `time` to be the current time.'''
    if date is None or time is None:
        current_datetime = pd.Timestamp.now('US/Eastern')
        current_date, current_time = current_datetime.date(), current_datetime.time()
        if date is None: date = str(current_date)
        if time is None: time = current_time.strftime('%H:%M')
    return date, time


@run_multiple_times_if_logged_out
def _response_from_yield_curve(display_type, input_date, input_time, return_execution_time):
    '''Get yield curve for `input_date` and `input_time`. If `display_type` is 'plot', then make API call corresponding to 
    the yield curve plot, and if `display_type` is 'table', then make API call corresponding to the yield curve table.
    NOTE: cannot have the argument for time be `time` since the `time` library is used for timing the function.'''
    assert display_type in ACCEPTABLE_DISPLAY_TYPES_FOR_YIELD_CURVE, f'`display_type` is {display_type}, but must be in {ACCEPTABLE_DISPLAY_TYPES_FOR_YIELD_CURVE}'
    api_url = 'realtimeyieldcurve' if display_type == 'plot' else 'yield'
    start_time = time.time()
    request_obj = requests.get(f'{REQUEST_URL}/api/{api_url}?date={input_date}&time={input_time}&username={USERNAME}&password={PASSWORD}')
    time_elapsed = round(time.time() - start_time, 3)    # round the value in order to get a readable error statement
    response_dict = request_obj.json()
    if 'error' in response_dict and response_dict['error'] == LOGGED_OUT_MESSAGE: raise LoggedOutError
    return (response_dict, time_elapsed) if return_execution_time else response_dict


@run_multiple_times_if_logged_out
def response_from_yield_curve_plot(date=None, time=None, return_execution_time=False):
    '''Get yield curve plot for `date` and `time`. If `date` is `None` default to current date. If `time` is `None`, default to current time.'''
    date, time = _get_current_date_and_time_if_none(date, time)
    return _response_from_yield_curve('plot', date, time, return_execution_time)


@run_multiple_times_if_logged_out
def response_from_yield_curve_table(date=None, time=None, return_execution_time=False):
    '''Get yield curve table for `date` and `time`. If `date` is `None` default to current date. If `time` is `None`, default to current time.'''
    date, time = _get_current_date_and_time_if_none(date, time)
    return _response_from_yield_curve('table', date, time, return_execution_time)
