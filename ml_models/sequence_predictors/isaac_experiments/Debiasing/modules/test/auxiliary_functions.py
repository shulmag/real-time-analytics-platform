'''
'''
import os
from functools import wraps
import requests
import pandas as pd

from google.cloud import bigquery

from modules.finance import FEATURES_FOR_OUTPUT_CSV, NUMERICAL_ERROR
from modules.get_creds import get_creds

from modules.test.auxiliary_variables import USERNAME, PASSWORD, QUANTITY, TRADE_TYPE, DIRECTORY


REQUEST_URL = 'http://localhost:5000'    # for testing the server as is deployed, change this to 'https://api.ficc.ai' or 'https://server-3ukzrmokpq-uc.a.run.app'


def run_multiple_times_before_failing(test_function):
    '''This function is to be used as a decorator. It will run `test_function` over and over again until it 
    hits a success for a maximum of `max_runs` (specified below) times. It solves the following problem: sometimes 
    the API call fails randomly, but when the exact code is run again, it succeeds. In this instance, we want 
    to make sure that we do not raise a false alarm that the product is broken, when it was just an API call 
    that just happened to fail randomly.
    TODO: figure out why the API calls are not stable.'''
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
    return wrapper


def get_bq_client():
    '''Initialize the credentials and the bigquery client.'''
    get_creds()
    return bigquery.Client()


def response_from_individual_pricing(cusip, trade_type, quantity):
    '''Individually prices `cusip` at `trade_type` and `quantity`.'''
    request_obj = requests.get(f'{REQUEST_URL}/api/price?cusip={cusip}&tradeType={trade_type}&amount={quantity}&username={USERNAME}&password={PASSWORD}')
    return request_obj.json()


def get_filename_from_cusip_list(cusip_list):
    cusip_list_as_string = '_'.join(cusip_list) if type(cusip_list) == list else cusip_list
    return f'{DIRECTORY}/{cusip_list_as_string}.csv'


def response_from_batch_pricing(filename, cusip_list=None, quantity_list=None, trade_type=TRADE_TYPE):
    '''Batch prices `cusip_list` by putting it into the csv `filename` where the corresponding 
    quantities are in `quantity_list` and for trade type of `trade_type`.'''
    if cusip_list is not None:    # if `cusip_list` is `None`, then assume that the file at `filename` already has all of the CUSIP data
        if type(cusip_list) != list:    # `cusip_list` is a single CUSIP
            cusip_list = [cusip_list]
            if quantity_list is not None:    # `quantity_list` should be a single quantity
                quantity_list = [quantity_list]
        
        array_for_csv = cusip_list if quantity_list is None else list(zip(cusip_list, quantity_list))    # group each cusips with its corresponding quantity
        pd.DataFrame(array_for_csv).to_csv(filename, header=None, index=None)
    data = {'username': USERNAME, 'password': PASSWORD, 'amount': QUANTITY, 'tradeType': trade_type}    # note that `amount` only fills in values that are empty
    file = {'file': open(filename, 'rb')}
    
    request_ref = f'{REQUEST_URL}/api/batchpricing'
    try:    # perform a try...except so that the file can be removed even if an error is thrown
        request_obj = requests.post(request_ref, data=data, files=file)
        os.remove(filename)
        return request_obj
    except Exception as e:
        os.remove(filename)
        raise e


def _get_cusip_to_count_dict(cusip_list):
    '''Returns a dictionary mapping each CUSIP in `cusip_list` to the number of occurences in `cusip_list`.'''
    if type(cusip_list) != list:    # `cusip_list` is a single CUSIP
        cusip_list = [cusip_list]
    cusip_count = dict()    # store the count of each CUSIP in `cusip_list`
    for cusip in cusip_list:
        cusip_count[cusip] = cusip_count.get(cusip, 0) + 1    # second argument of .get(...) provides a default value if the key is not found in the map
    return cusip_count


def get_spreadsheet_as_list(request_obj):
    '''Takes the request object, `request_obj` and returns the content of the spreadsheet as a list of lists 
    where each sublist contains the values of one row of the spreadsheet. Assumes that `request_obj` is a 
    successful response from batch pricing.'''
    assert request_obj.ok, 'The response from batch pricing was not successful'    # successful response; checks whether the status_code is less than 400
    content = request_obj.content.decode('utf-8')
    content = content.split('\n')

    header = content[0]
    header = header.split(',')
    assert FEATURES_FOR_OUTPUT_CSV == header, f'The header of the spreadsheet should be {FEATURES_FOR_OUTPUT_CSV} but is instead {header}'    # column names in predictions csv are correct

    content = content[1:-1]    # first row is the column names, last row is an empty line
    return [row.split(',') for row in content]


def check_that_batch_pricing_gives_output_for_all_cusips(request_obj, cusip_list, check_quantity_equals_default=True):
    '''Check that each CUSIP in `cusip_list` has some output in the request_obj.'''
    content = get_spreadsheet_as_list(request_obj)

    cusip_count = _get_cusip_to_count_dict(cusip_list)
    cusip_idx = FEATURES_FOR_OUTPUT_CSV.index('cusip')
    quantity_idx = FEATURES_FOR_OUTPUT_CSV.index('quantity')
    for row in content:
        cusip = row[cusip_idx]
        assert cusip in cusip_count, f'CUSIP {cusip} is missing (or not with the correct frequency) from the returned spreadsheet'    # check that the CUSIP is one of those in the original cusip_list passed into batch pricing
        cusip_count[cusip] -= 1
        if cusip_count[cusip] == 0: cusip_count.pop(cusip)    # remove CUSIP from map if count equals 0
        if check_quantity_equals_default: assert row[quantity_idx] == str(QUANTITY * 1000), f'The quantity in the spreadsheet {row[quantity_idx]} should equal the quantity passed in {QUANTITY * 1000}'    # check that the corresponding quantity for each CUSIP is 500000
    assert len(cusip_count) == 0, f'Spreadsheet has extra CUSIPs'    # checks that all CUSIPs were found in the predictions csv


def check_that_batch_pricing_gives_price_for_all_cusips(request_obj, cusip_list, check_quantity_equals_default=True):
    '''Check that each CUSIP in `cusip_list` has a valid price and ytw in the request_obj.'''
    content = get_spreadsheet_as_list(request_obj)

    cusip_count = _get_cusip_to_count_dict(cusip_list)
    cusip_idx = FEATURES_FOR_OUTPUT_CSV.index('cusip')
    quantity_idx = FEATURES_FOR_OUTPUT_CSV.index('quantity')
    price_idx = FEATURES_FOR_OUTPUT_CSV.index('price')
    ytw_idx = FEATURES_FOR_OUTPUT_CSV.index('ytw')
    quantity_idx = FEATURES_FOR_OUTPUT_CSV.index('quantity')
    for idx, row in enumerate(content):
        cusip = row[cusip_idx]
        assert cusip in cusip_count, f'CUSIP {cusip} is missing (or not with the correct frequency) from the returned spreadsheet'    # check that the CUSIP is one of those in the original cusip_list passed into batch pricing
        cusip_count[cusip] -= 1
        if cusip_count[cusip] == 0: cusip_count.pop(cusip)    # remove CUSIP from map if count equals 0
        assert float(row[price_idx]) != NUMERICAL_ERROR, f'For CUSIP {cusip_list[idx]}, the predicted price {row[price_idx]} should not be equal to {NUMERICAL_ERROR}'    # check that the price is not the value used for numerical error (indicating that an actual price was provided)
        assert float(row[ytw_idx]) != NUMERICAL_ERROR, f'For CUSIP {cusip_list[idx]}, the predicted ytw {row[ytw_idx]} should not be equal to {NUMERICAL_ERROR}'    # check that the ytw is not the value used for numerical error (indicating that an actual price was provided)
        if check_quantity_equals_default: assert row[quantity_idx] == str(QUANTITY * 1000)    # check that the corresponding quantity for each CUSIP is 500000
    assert len(cusip_count) == 0, f'Spreadsheet has extra CUSIPs'    # checks that all CUSIPs were found in the predictions csv


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
                      'moodysRating': 'undefined', 
                      'minMaturityYear': 2023, 
                      'maxMaturityYear': 2123, 
                      'amount': 'undefined', 
                      'userTriggered': True}
    for key in default_values:
        if key not in kwargs:
            kwargs[key] = default_values[key]
    args_string = '&'.join([f'{arg_name}={arg_value}' for arg_name, arg_value in kwargs.items()])
    request_obj = requests.get(f'{REQUEST_URL}/api/getsimilarbonds?username={USERNAME}&password={PASSWORD}&' + args_string)
    return request_obj.json()


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
                               'price': 'price', 
                               'moodys_long': 'moodysRating'}
    response_dict = response_from_similar_bonds(cusip=cusip, **{url_arg: response_dict[response_arg] for response_arg, url_arg in response_arg_to_url_arg.items()})
    assert 'error' not in response_dict, f'For CUSIP {cusip}, getting similar bonds should not have an error, but has an error: {response_dict["error"]}'
    return response_dict
