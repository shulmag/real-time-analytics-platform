'''
'''
import time

from auxiliary_variables import DIRECTORY, USERNAME, TRADE_TYPE, QUANTITY, NUMERICAL_ERROR
from auxiliary_functions import run_multiple_times_before_failing, response_from_individual_pricing, response_from_batch_pricing, get_bq_client, sqltodf


INDIVIDUAL_PRICING_ERROR_PRICE = None    # value used in log when a CUSIP cannot be individually priced
BATCH_PRICING_ERROR_PRICE = NUMERICAL_ERROR    # value used in log when a CUSIP cannot be batch priced

NUM_SECONDS_SLEEP = 10    # number of seconds to sleep before querying the logs to ensure that the logs have been updated since the logging is done asynchronously; the number of seconds is chosen arbitrarily as a length that I feel long enough for the logging to be completed


@run_multiple_times_before_failing
def test_logging_individual():
    '''Tests that individual pricing is successfully logged. The CUSIPs should be different from 
    the CUSIPs in `test_logging_batch_success_error` so to not get logging results confused.'''
    random_cusip = '64971XQM3'    # send dummy requests with `random_cusip` so that the most recent CUSIPs are not from a previous run of the test
    response_from_individual_pricing(random_cusip, TRADE_TYPE, QUANTITY)
    response_from_individual_pricing(random_cusip, TRADE_TYPE, QUANTITY)

    successful_cusip = '646136TN1'
    response_from_individual_pricing(successful_cusip, TRADE_TYPE, QUANTITY)
    failed_cusip = '40064UAW2'    # CUSIP is not outstanding
    response_from_individual_pricing(failed_cusip, TRADE_TYPE, QUANTITY)
    time.sleep(NUM_SECONDS_SLEEP)
    
    query = f'''SELECT time, cusip, direction, quantity, ficc_price, error
                FROM eng-reactor-287421.api_calls_tracker.usage_data
                WHERE user="{USERNAME}"
                ORDER BY time desc
                LIMIT 2'''
    df = sqltodf(query, get_bq_client())
    cusip_list = df['cusip'].tolist()
    assert set(cusip_list) == {successful_cusip, failed_cusip}, f'The most recently priced CUSIPs {set(cusip_list)} are not the ones that were priced in this test {set([successful_cusip, failed_cusip])}'
    assert df['direction'].tolist() == [TRADE_TYPE] * 2, f'The trade types of the most recently priced CUSIPs {df["direction"].tolist()} do not match the ones in this test {[TRADE_TYPE] * 2}'
    assert df['quantity'].tolist() == [QUANTITY] * 2, f'The trade types of the most recently priced CUSIPs {df["quantity"].tolist()} do not match the ones in this test {[QUANTITY] * 2}'
    
    successful_cusip_df = df[df['cusip'] == successful_cusip]
    failed_cusip_df = df[df['cusip'] == failed_cusip]
    price_for_successful_cusip = successful_cusip_df['ficc_price'].values[0]
    price_for_failed_cusip = failed_cusip_df['ficc_price'].values[0]
    assert not (price_for_successful_cusip == INDIVIDUAL_PRICING_ERROR_PRICE), f'When pricing CUSIP {successful_cusip}, the price should not be {INDIVIDUAL_PRICING_ERROR_PRICE}, but is {price_for_successful_cusip}.'
    assert price_for_failed_cusip == INDIVIDUAL_PRICING_ERROR_PRICE, f'When pricing CUSIP {failed_cusip}, the price should be {INDIVIDUAL_PRICING_ERROR_PRICE}, but is {price_for_failed_cusip}.'
    error_for_successful_cusip = successful_cusip_df['error'].values[0]
    error_for_failed_cusip = failed_cusip_df['error'].values[0]
    assert error_for_successful_cusip == False, f'When pricing CUSIP {successful_cusip}, the error should be False, but is {error_for_successful_cusip}'
    assert error_for_failed_cusip == True, f'When pricing CUSIP {failed_cusip}, the error should be True, but is {error_for_failed_cusip}'


@run_multiple_times_before_failing
def test_logging_batch_success_error():
    '''Tests that batch pricing is successfully logged. The CUSIPs in `random_cusip_list` should 
    be different from the CUSIPs in `orig_cusip_list` so to not get logging results confused.'''
    random_cusip_list = ['64971XQM3'] * 2    # send dummy requests with these CUSIPs so that the most recent CUSIPs are not from a previous run of the test
    response_from_batch_pricing(f'{DIRECTORY}/{"_".join(random_cusip_list)}.csv', random_cusip_list)

    successful_cusip = '950885SN4'
    failed_cusip = '452226Y31'    # CUSIP does not have sufficient data
    orig_cusip_list = [successful_cusip, failed_cusip]
    response_from_batch_pricing(f'{DIRECTORY}/{"_".join(orig_cusip_list)}.csv', orig_cusip_list)
    time.sleep(NUM_SECONDS_SLEEP)
    
    query = f'''SELECT time, cusip, direction, quantity, ficc_price, error
                FROM eng-reactor-287421.api_calls_tracker.usage_data
                WHERE user="{USERNAME}"
                ORDER BY time desc
                LIMIT 2'''
    df = sqltodf(query, get_bq_client())
    cusip_list = df['cusip'].tolist()
    assert set(cusip_list) == set(orig_cusip_list), f'The most recently priced CUSIPs {set(cusip_list)} are not the ones that were priced in this test {set(orig_cusip_list)}'
    assert df['direction'].tolist() == [TRADE_TYPE] * 2, f'The trade types of the most recently priced CUSIPs {df["direction"].tolist()} do not match the ones in this test {[TRADE_TYPE] * 2}'
    assert df['quantity'].tolist() == [QUANTITY] * 2, f'The trade types of the most recently priced CUSIPs {df["quantity"].tolist()} do not match the ones in this test {[QUANTITY] * 2}'

    successful_cusip_df = df[df['cusip'] == successful_cusip]
    failed_cusip_df = df[df['cusip'] == failed_cusip]
    price_for_successful_cusip = successful_cusip_df['ficc_price'].values[0]
    price_for_failed_cusip = failed_cusip_df['ficc_price'].values[0]
    assert not (price_for_successful_cusip == BATCH_PRICING_ERROR_PRICE), f'When pricing CUSIP {successful_cusip}, the price should not be {BATCH_PRICING_ERROR_PRICE}, but is {price_for_successful_cusip}.'
    assert price_for_failed_cusip == BATCH_PRICING_ERROR_PRICE, f'When pricing CUSIP {failed_cusip}, the price should be {BATCH_PRICING_ERROR_PRICE}, but is {price_for_failed_cusip}.'    
    error_for_successful_cusip = successful_cusip_df['error'].values[0]
    error_for_failed_cusip = failed_cusip_df['error'].values[0]
    assert error_for_successful_cusip == False, f'When pricing CUSIP {successful_cusip}, the error should be False, but is {error_for_successful_cusip}'
    assert error_for_failed_cusip == True, f'When pricing CUSIP {failed_cusip}, the error should be True, but is {error_for_failed_cusip}'


@run_multiple_times_before_failing
def test_logging_batch_yield_spread_dollar_price():
    '''Tests that batch pricing is successfully logged. One of the CUSIPs should use 
    the dollar price model, and the other CUSIP should use the yield spread model.'''
    random_cusip_list = ['64971XQM3'] * 2    # send dummy requests with these CUSIPs so that the most recent CUSIPs are not from a previous run of the test
    response_from_batch_pricing(f'{DIRECTORY}/{"_".join(random_cusip_list)}.csv', random_cusip_list)

    yield_spread_cusip = '000416Y92'
    dollar_price_cusip = '527839DQ4'    # CUSIP has negative yields in history
    orig_cusip_list = [yield_spread_cusip, dollar_price_cusip]
    response_from_batch_pricing(f'{DIRECTORY}/{"_".join(orig_cusip_list)}.csv', orig_cusip_list)
    time.sleep(NUM_SECONDS_SLEEP)
    
    query = f'''SELECT time, cusip, direction, quantity, model_used
                FROM eng-reactor-287421.api_calls_tracker.usage_data
                WHERE user="{USERNAME}"
                ORDER BY time desc
                LIMIT 2'''
    df = sqltodf(query, get_bq_client())
    cusip_list = df['cusip'].tolist()
    assert set(cusip_list) == set(orig_cusip_list), f'The most recently priced CUSIPs {set(cusip_list)} are not the ones that were priced in this test {set(orig_cusip_list)}'
    assert df['direction'].tolist() == [TRADE_TYPE] * 2, f'The trade types of the most recently priced CUSIPs {df["direction"].tolist()} do not match the ones in this test {[TRADE_TYPE] * 2}'
    assert df['quantity'].tolist() == [QUANTITY] * 2, f'The trade types of the most recently priced CUSIPs {df["quantity"].tolist()} do not match the ones in this test {[QUANTITY] * 2}'
    assert df['model_used'].tolist() == ['yield_spread', 'dollar_price'], f'The models used of the most recently priced CUSIPs {df["model_used"].tolist()} do not match the ones in this test {["yield_spread", "dollar_price"]}'
