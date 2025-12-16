'''
'''
import time

from auxiliary_variables import DIRECTORY, FEATURES_FOR_OUTPUT_CSV, QUANTITY, TRADE_TYPE, USERNAME, LOGGING_PRECISION, TOLERANCE
from auxiliary_functions import run_multiple_times_before_failing, response_from_individual_pricing, response_from_batch_pricing, sqltodf, get_bq_client, load_cusips_from_gcs


def check_logs_for_additional_info_for_error_message(cusip: str, 
                                                     quantity, 
                                                     trade_type, 
                                                     price_from_batch_pricing: float, 
                                                     price_from_individual_pricing: float, 
                                                     ytw_from_batch_pricing: float, 
                                                     ytw_from_individual_pricing: float, 
                                                     prices_are_equal: bool, 
                                                     ytw_are_equal: bool, 
                                                     check_price_only: bool):
    '''This function is no longer used because the logging mechanism has changed to be done in batches, 
    so it may take a while (> 15 mins) before the logs are populated with the desired entry.'''
    time.sleep(2)    # have a 2 second delay since the logging is done asynchronously; the time of 2 seconds is chosen arbitrarily as a length that I feel long enough for the logging to be completed
    query = f'''SELECT time, cusip, ficc_price, yield_spread, ficc_ycl, batch
                FROM eng-reactor-287421.api_calls_tracker.usage_data
                WHERE user="{USERNAME}"
                ORDER BY time desc
                LIMIT 2'''
    df_logging = sqltodf(query, get_bq_client())
    batch, individual = df_logging[df_logging['batch'] == True].iloc[0], df_logging[df_logging['batch'] == False].iloc[0]
    batch_cusip, batch_ycl, batch_ys, batch_timestamp = batch['cusip'], batch['ficc_ycl'], batch['yield_spread'], batch['time']
    if batch_ycl is not None: batch_ycl = round(batch_ycl, LOGGING_PRECISION)
    if batch_ys is not None: batch_ys = round(batch_ys, LOGGING_PRECISION)
    individual_cusip, individual_ycl, individual_ys, individual_timestamp = individual['cusip'], round(individual['ficc_ycl'], LOGGING_PRECISION), round(individual['yield_spread'], LOGGING_PRECISION), individual['time']
    if individual_ycl is not None: individual_ycl = round(individual_ycl, LOGGING_PRECISION)
    if individual_ys is not None: individual_ys = round(individual_ys, LOGGING_PRECISION)
    assert batch_cusip == individual_cusip == cusip, f'The batch priced CUSIP from logging {batch_cusip} must equal the individually priced CUSIP from logging {individual_cusip} must equal the passed in CUSIP {cusip}'

    error_message = lambda price_or_ytw, batch_price_or_ytw, individual_price_or_ytw: f'For CUSIP {cusip}, quantity (in thousands) {quantity}, and trade type {trade_type}, the {price_or_ytw} from individual pricing {individual_price_or_ytw} (ys: {individual_ys}, ycl: {individual_ycl}), priced at {individual_timestamp.replace(tzinfo=None)} ET is not within a tolerance of {TOLERANCE} from the {price_or_ytw} from batch pricing {batch_price_or_ytw} (ys: {batch_ys}, ycl: {batch_ycl}), priced at {batch_timestamp.replace(tzinfo=None)} ET'
    assert prices_are_equal, error_message('price', price_from_batch_pricing, price_from_individual_pricing)
    if check_price_only is False: assert ytw_are_equal, error_message('ytw', ytw_from_batch_pricing, ytw_from_individual_pricing)


def check_individual_pricing_equals_batch_pricing(cusip, trade_type=TRADE_TYPE, quantity=QUANTITY, check_price_only=False):
    '''Tests that individual pricing and batch pricing return the same price for `cusip` for 
    `trade_type` and `quantity`. If `check_price_only` is `True`, then do not check the ytw 
    estimate, which is necessary if the CUSIP is priced using the dollar price model because 
    we currently do not provide a ytw estimate for the dollar price model.'''
    filename = f'{DIRECTORY}/{cusip}.csv'

    # individual pricing
    response_dict = response_from_individual_pricing(cusip, trade_type, quantity)
    time.sleep(1)    # 1 second delay in order to not bombard the server with API calls which reduces chance of getting this error: requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
    assert 'error' not in response_dict, f'The response should not have an error, but has an error: {response_dict["error"]}'
    response_dict = response_dict[0]    # retrieve index 0 since the dictionary is in a one item list
    price_from_individual_pricing = float(response_dict['price'])    # convert to float to be able to perform math operations
    if check_price_only is False: ytw_from_individual_pricing = float(response_dict['ficc_ytw'])    # convert to float to be able to perform math operations

    # batch pricing
    price_idx = FEATURES_FOR_OUTPUT_CSV.index('price')
    ytw_idx = FEATURES_FOR_OUTPUT_CSV.index('ytw')
    request_obj = response_from_batch_pricing(filename, cusip, quantity, trade_type)
    assert request_obj.ok, f'For CUSIP {cusip}, quantity (in thousands) {quantity}, and trade type {trade_type}, the response from batch pricing was not successful'    # successful response; checks whether the status_code is less than 400
    content = request_obj.content.decode('utf-8')
    content = content.split('\n')[1]    # first row is the column names, last row is an empty line, so we only want the middle (1-indexed) row
    content = content.split(',')    # separate the columns in the predictions csv
    price_from_batch_pricing = float(content[price_idx])    # convert to float to be able to perform math operations
    if check_price_only is False: ytw_from_batch_pricing = float(content[ytw_idx])    # convert to float to be able to perform math operations

    prices_are_equal = abs(price_from_individual_pricing - price_from_batch_pricing) <= TOLERANCE
    ytw_are_equal = True if check_price_only else abs(ytw_from_individual_pricing - ytw_from_batch_pricing) <= TOLERANCE
    if not prices_are_equal or not ytw_are_equal:    # throw assertion error with yield spread and yield curve level which is present in the logging
        # check_logs_for_additional_info_for_error_message(cusip, quantity, trade_type, price_from_batch_pricing, price_from_individual_pricing, ytw_from_batch_pricing, ytw_from_individual_pricing, prices_are_equal, ytw_are_equal, check_price_only)
        error_message = lambda price_or_ytw, batch_price_or_ytw, individual_price_or_ytw: f'For CUSIP {cusip}, quantity (in thousands) {quantity}, and trade type {trade_type}, the {price_or_ytw} from individual pricing {individual_price_or_ytw} is not within a tolerance of {TOLERANCE} from the {price_or_ytw} from batch pricing {batch_price_or_ytw}'
        assert prices_are_equal, error_message('price', price_from_batch_pricing, price_from_individual_pricing)
        if check_price_only is False: assert ytw_are_equal, error_message('ytw', ytw_from_batch_pricing, ytw_from_individual_pricing)


def check_individual_pricing_equals_batch_pricing_different_trade_types_and_quantities(cusip):
    '''Tests that individual pricing and batch pricing return the same price for `cusip` for 
    different values of `trade_type` and `quantity`.'''
    trade_types = ['S', 'P', 'D']
    quantities = [100, 500, 1000]    # arbitrarily chosen
    for trade_type in trade_types:
        for quantity in quantities:
            check_individual_pricing_equals_batch_pricing(cusip, trade_type, quantity)


@run_multiple_times_before_failing
def test_64971XQM3_different_trade_types_and_quantities():
    '''Tests that individual pricing and batch pricing return the same price for 64971XQM3 for 
    different values of `trade_type` and `quantity`.'''
    check_individual_pricing_equals_batch_pricing_different_trade_types_and_quantities('64971XQM3')


@run_multiple_times_before_failing
def test_yield_spread_model_cusips():
    '''Tests that individual pricing and batch pricing return the same price for CUSIPs that 
    use the yield spread model.'''
    cusip_list = ['13063D7Q5']
    for cusip in cusip_list:
        check_individual_pricing_equals_batch_pricing(cusip)


@run_multiple_times_before_failing
def test_dollar_price_model_cusips():
    '''Tests that individual pricing and batch pricing return the same price for CUSIPs that 
    use the dollar price model.'''
    mature_cusips = load_cusips_from_gcs("short_maturity")
    cusip_list = ['71910EAM1',    # negative yields in the history
                  '052398GZ1',    # defaulted
                   mature_cusips[0]]    # maturing soon
    for cusip in cusip_list:
        check_individual_pricing_equals_batch_pricing(cusip, check_price_only=True)
