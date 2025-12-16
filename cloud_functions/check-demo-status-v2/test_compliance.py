'''
'''
from functools import partial

import math
import time
import os
import csv
import requests
import json
import pandas as pd

from auxiliary_variables import REQUEST_URL, USERNAME, PASSWORD, DIRECTORY, QUANTITY, TRADE_TYPE, LOGGED_OUT_MESSAGE, LoggedOutError, FEATURES_FOR_OUTPUT_CSV, ADDITIONAL_FEATURES_FOR_COMPLIANCE_CSV, TOLERANCE, NUMERICAL_ERROR
from auxiliary_functions import run_multiple_times_before_failing, run_multiple_times_if_logged_out, get_filename_from_cusip_list, get_spreadsheet_as_list, run_tests_with_and_without_file, handle_nan

DEFINE_TESTS_THAT_USE_BIGQUERY_TO_GET_DATA = False    # boolean flag used in local testing to decide whether we will define the tests that use BigQuery to create the data; usually we will not run these tests because (a) they are very expensive and slow due to the BigQuery calls, and (2) no one is using the compliance module for dates this far in the past; NOTE: these tests no longer use BigQuery to create data since the server converts any input in the future or way in the past to price at realtime

price_idx = FEATURES_FOR_OUTPUT_CSV.index('price')
bid_ask_price_delta_idx = ADDITIONAL_FEATURES_FOR_COMPLIANCE_CSV.index('bid_ask_price_delta')
run_tests_with_and_without_file_compliance_with_realtime = partial(run_tests_with_and_without_file, check_if_content_is_equal=True, 
                                                                                                    tolerance=TOLERANCE * 5,    # increase tolerance since compliance takes a while to run
                                                                                                    columns_to_ignore=set([price_idx, len(FEATURES_FOR_OUTPUT_CSV) + bid_ask_price_delta_idx]))    # ignore price column because this may fluctuate a bit due to minor fluctuations in the yield
run_tests_with_and_without_file_compliance_without_realtime = partial(run_tests_with_and_without_file, check_if_content_is_equal=True, 
                                                                                                       tolerance=TOLERANCE,    # since there is no realtime pricing, all columns should be identical (i.e., delay between calls does not affect the output) and so use default tolerance in case of odd rounding
                                                                                                       columns_to_ignore=set())    # since there is no realtime pricing, all columns should be identical (i.e., delay between calls does not affect the output)


def get_cusip_list_quantity_list_tradetype_list_userprice_list_tradedatetime_list(filename: str):
    '''Based heavily off of `ficc/app_engine/demo/server/modules/test/auxiliary_functions.py::response_from_batch_pricing(...)`.'''
    cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list = [], [], [], [], []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:    # `for` loop is very similar to logic in `get_predictions_from_batch_pricing(...)`
            if len(row) > 0:    # only process the row if it is not empty
                cusip_list.append(row[0].upper())    # uppercase each cusip

                quantity = QUANTITY
                if len(row) > 1:    # user has inputted a quantity
                    user_quantity = row[1]
                    try:
                        quantity = int(user_quantity) * 1000
                    except ValueError:    # catches the case where a quantity value is not a valid integer
                        pass    # `quantity` is initialized to `default_quantity`
                quantity_list.append(quantity)

                trade_type = TRADE_TYPE
                if len(row) > 2:    # user has inputted a quantity
                    user_trade_type = row[2]
                    if user_trade_type in ('S', 'P', 'D'): trade_type = user_trade_type
                trade_type_list.append(trade_type)

                user_price = None
                if len(row) > 3:
                    user_price = row[3]
                    try:
                        user_price = float(user_price)
                    except ValueError:    # catches the case where a quantity value is not a valid float
                        pass    # `user_price` is initialized to `None`
                user_price_list.append(user_price)

                trade_datetime = None
                if len(row) > 4:    # user has inputted a quantity
                    trade_datetime = row[4]
                trade_datetime_list.append(trade_datetime)

    assert len(cusip_list) == len(quantity_list) == len(trade_type_list) == len(user_price_list) == len(trade_datetime_list), '`cusip_list`, `quantity_list`, `trade_type_list`, `user_price_list`, and `trade_datetime_list` do not have the same number of items'
    return cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list


@run_multiple_times_if_logged_out
def response_from_compliance(filename, cusip_list=None, quantity_list=None, trade_type_list=None, user_price_list=None, trade_datetime_list=None, return_execution_time=False, create_file: bool = True):
    '''Based heavily off of `cloud_functions/check-demo-status-v2/auxiliary_functions.py::response_from_batch_pricing(...)`.'''
    if cusip_list is None:    # if `cusip_list` is `None`, then assume that the file at `filename` already has all of the CUSIP data
        cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list = get_cusip_list_quantity_list_tradetype_list_userprice_list_tradedatetime_list(filename)
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
        if user_price_list is not None:    # `user_price_list` should be a single quantity
            user_price_list = [user_price_list]
        if trade_datetime_list is not None:    # `trade_datetime_list` should be a single quantity
            trade_datetime_list = [trade_datetime_list]
    
    if quantity_list is None or trade_type_list is None or user_price_list is None: raise NotImplementedError('No support for compliance without quantities or without trade types or without user prices')    # TODO: implement this feature
    data = {'username': USERNAME, 'password': PASSWORD, 'download': True}    # setting `download` to `True` so that the data is returned as a CSV file which is the assumed download format for the automated test
    if create_file:
        array_for_csv = cusip_list if quantity_list is None else list(zip(cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list if trade_datetime_list is not None else [None] * len(cusip_list)))    # group each cusips with its corresponding quantity, trade_type, user_price, and trade_datetime
        pd.DataFrame(array_for_csv).to_csv(filename, header=None, index=None)
        file = {'file': open(filename, 'rb')}
    else:
        data = data | {'cusipList': cusip_list, 'quantityList': quantity_list, 'tradeTypeList': trade_type_list, 'userPriceList': user_price_list, 'tradeDatetimeList': trade_datetime_list}
        file = dict()
    
    request_ref = f'{REQUEST_URL}/api/compliance'
    try:    # perform a try...except so that the file can be removed even if an error is thrown
        start_time = time.time()
        request_obj = requests.post(request_ref, data=data, files=file)
        time_elapsed = round(time.time() - start_time, 3)    # round the value in order to get a readable error statement
    except Exception as e:
        if create_file: os.remove(filename)
        raise e
    
    return_values = (request_obj, time_elapsed) if return_execution_time else request_obj
    try:
        response_dict = request_obj.json()
        response_dict = json.loads(response_dict)
    except Exception:    # unable to convert `request_obj` to a response dict
        if create_file: os.remove(filename)
        return return_values
    if 'error' in response_dict and response_dict['error'] == LOGGED_OUT_MESSAGE:
        if create_file: os.remove(filename)
        raise LoggedOutError
    if create_file: os.remove(filename)
    return return_values


def _test_cusip_order_preserved(cusip_list: list, quantity_list: list, trade_type_list: list, user_price_list: list, trade_datetime_list: list = None, with_file: bool = False, features_to_check_if_identical: list = []):
    request_obj = response_from_compliance(get_filename_from_cusip_list(cusip_list), cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list, create_file=with_file)
    features_for_output_csv = FEATURES_FOR_OUTPUT_CSV + ADDITIONAL_FEATURES_FOR_COMPLIANCE_CSV
    content = get_spreadsheet_as_list(request_obj, features_for_output_csv, not with_file)
    
    cusip_idx = features_for_output_csv.index('cusip')
    if len(features_to_check_if_identical) > 0:
        indices_to_check = [features_for_output_csv.index(feature) for feature in features_to_check_if_identical]
        values = {feature_index: None for feature_index in indices_to_check}
    for idx, (row, cusip) in enumerate(zip(content, cusip_list)):    # makes sure that each price is not the value used for numerical error and that it is different than the previous price
        row_cusip = row[cusip_idx]
        assert cusip == row_cusip, f'For row {idx}, CUSIP {row_cusip} does not equal the CUSIP that was entered: {cusip}'
        if len(features_to_check_if_identical) > 0:
            for feature_index in indices_to_check:
                current_value = values[feature_index]
                if current_value is not None:
                    assert current_value == row[feature_index]
                else:
                    values[feature_index] = row[feature_index]
    return content    # used for downstream comparison to other `content`s


def _test_cusip_order_preserved_and_all_refused_to_price(cusip_list: list, quantity_list: list, trade_type_list: list, user_price_list: list, trade_datetime_list: list = None, with_file: bool = True):
    '''Tests that the data passed in causes the compliance module to refuse to price the CUSIPs. One example of this is that 
    the user prices entered are all `None`.'''
    request_obj = response_from_compliance(get_filename_from_cusip_list(cusip_list), cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list, create_file=with_file)
    features_for_output_csv = FEATURES_FOR_OUTPUT_CSV + ADDITIONAL_FEATURES_FOR_COMPLIANCE_CSV
    content = get_spreadsheet_as_list(request_obj, features_for_output_csv, not with_file)
    
    cusip_idx = features_for_output_csv.index('cusip')
    price_idx = features_for_output_csv.index('price')
    ytw_idx = features_for_output_csv.index('ytw')
    for idx, (row, cusip) in enumerate(zip(content, cusip_list)):    # makes sure that each price is not the value used for numerical error and that it is different than the previous price
        row_cusip = row[cusip_idx]
        assert cusip == row_cusip, f'For row {idx}, CUSIP {row_cusip} does not equal the CUSIP that was entered: {cusip}'
        row_price = handle_nan(row[price_idx])
        row_ytw = handle_nan(row[ytw_idx])
        assert row_price == row_ytw == NUMERICAL_ERROR, f'For row {idx}, the price and ytw should each be {NUMERICAL_ERROR} but are {row_price} and {row_ytw} respectively'
    return content    # used for downstream comparison to other `content`s


if DEFINE_TESTS_THAT_USE_BIGQUERY_TO_GET_DATA:
    @run_multiple_times_before_failing
    def test_duplicate_cusips_different_trade_datetimes_realtime_some_bigquery_some_redis():
        '''Tests that compliance returns a valid response when some of the CUSIPs have a trade datetime 
        inputted for point in time pricing, while others do not have a trade datetime inputted and so will 
        do real-time pricing. Additionally, some of the CUSIPs will be duplicate and some will have different 
        trade datetimes. Some will also be CUSIPs that we refuse to price. Some will use the dollar price model. 
        Since some of the trade datetimes are before 2024-06-27, which is defined in 
        `point_in_time_pricing.py::DATE_FROM_WHICH_ALL_PAST_TRADES_ARE_STORED_IN_REDIS`, BigQuery will be used 
        for those trades to create the trade history and similar trade history. Since all the dates are before 
        `point_in_time_pricing.py::DATE_FROM_WHICH_PAST_REFERENCE_DATA_IS_STORED_IN_REDIS`, BigQuery will be used 
        to create the reference data for all trades.
        NOTE: test is not being run because it has high BigQuery costs; see Jira task: https://ficcai.atlassian.net/browse/FA-2359'''
        cusip_list = ['64971XQM3', '6461367J4', '64971XQM3', '950885SN4', '431669AR1', '647201CH3']    # 431669AR1 has defaulted, 647201CH3 is a PAC bond so will use dollar price model
        quantity_list = [100, 250, 1000, 250, 750, 2000]
        trade_type_list = ['P', 'P', 'P', 'S', 'S', 'P']
        user_price_list = [99.5, 99.5, 105.234, 101.250, 95.432, 98.765]
        trade_datetime_list = ['2024-05-01 12:00:00', '2024-05-01 12:00:00', '', '2024-07-02 12:00:00', '', '']    # dates before 2024-06-27 will use BigQuery to create trade history and similar trade history
        run_tests_with_and_without_file_compliance_with_realtime(lambda with_file: _test_cusip_order_preserved(cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list, with_file=with_file))


@run_multiple_times_before_failing
def test_duplicate_cusips_different_trade_datetimes_realtime_all_redis():
    '''Tests that compliance returns a valid response when some of the CUSIPs have a trade datetime 
    inputted for point in time pricing, while others do not have a trade datetime inputted and so will 
    do real-time pricing. Additionally, some of the CUSIPs will be duplicate and some will have different 
    trade datetimes. Some will also be CUSIPs that we refuse to price. Some will use the dollar price model. 
    Also, an iteration of this test will be run with trade datetimes after `point_in_time_pricing.py::DATE_FROM_WHICH_MODELS_TRAINED_WITH_NEW_REFERENCE_DATA_IS_IN_PRODUCTION` 
    to ensure that compliance is supported with new reference data.'''
    cusip_list = ['64971XQM3', '6461367J4', '64971XQM3', '950885SN4', '431669AR1', '647201CH3', '64971XQM3']    # 431669AR1 has defaulted, 647201CH3 is a PAC bond so will use dollar price model; last entry is the same as the first entry to test if cache is used (must be checked manually)
    quantity_list = [100, 250, 1000, 250, 750, 2000, 500]
    trade_type_list = ['P', 'P', 'P', 'S', 'S', 'P', 'S']
    user_price_list = [99.5, 99.5, 105.234, 101.250, 95.432, 98.765, 100]
    trade_datetime_list = ['2024-10-03 10:00:00', '2024-10-03 10:00:00', '', '2024-10-03 12:00:00', '', '', '2024-10-03 10:00:00']
    run_tests_with_and_without_file_compliance_with_realtime(lambda with_file: _test_cusip_order_preserved(cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list, with_file=with_file))
    trade_datetime_list = ['2025-01-03 10:00:00', '2025-01-03 10:00:00', '', '2025-01-02 12:00:00', '', '', '2025-01-02 10:00:00']    # datetimes after `point_in_time_pricing.py::DATE_FROM_WHICH_MODELS_TRAINED_WITH_NEW_REFERENCE_DATA_IS_IN_PRODUCTION`
    run_tests_with_and_without_file_compliance_with_realtime(lambda with_file: _test_cusip_order_preserved(cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list, with_file=with_file))


if DEFINE_TESTS_THAT_USE_BIGQUERY_TO_GET_DATA:
    @run_multiple_times_before_failing
    def test_same_trade_datetime_bigquery():
        '''Tests that compliance returns a valid response when all CUSIPs have the same trade datetime. Since the 
        trade datetime is before 2024-06-27, which is defined in `point_in_time_pricing.py::DATE_FROM_WHICH_ALL_PAST_TRADES_ARE_STORED_IN_REDIS`, 
        BigQuery will be used to create the trade history and similar trade history and the reference data.
        NOTE: test is not being run because it has high BigQuery costs; see Jira task: https://ficcai.atlassian.net/browse/FA-2359'''
        cusip_list = ['64971XQM3', '950885SN4']
        quantity_list = [100, 250]
        trade_type_list = ['P', 'S']
        user_price_list = [99.5, 101.250]
        trade_datetime_list = ['2024-05-01 12:00:00', '2024-05-01 12:00:00']
        run_tests_with_and_without_file_compliance_without_realtime(lambda with_file: _test_cusip_order_preserved(cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list, with_file=with_file))


if DEFINE_TESTS_THAT_USE_BIGQUERY_TO_GET_DATA:
    @run_multiple_times_before_failing
    def test_same_trade_datetime_redis_for_trade_history_bigquery_for_reference_data():
        '''Tests that compliance returns a valid response when all CUSIPs have the same trade datetime. Since the 
        trade datetime is after 2024-06-27, which is defined in `point_in_time_pricing.py::DATE_FROM_WHICH_ALL_PAST_TRADES_ARE_STORED_IN_REDIS`, 
        redis will be used to create the trade history and similar trade history, but since the date is before 
        2024-09-24, which is defined in `point_in_time_pricing.py::DATE_FROM_WHICH_PAST_REFERENCE_DATA_IS_STORED_IN_REDIS`, 
        BigQuery will be used to create the reference data.
        NOTE: test is not being run because it has high BigQuery costs; see Jira task: https://ficcai.atlassian.net/browse/FA-2359'''
        cusip_list = ['64971XQM3', '950885SN4']
        quantity_list = [100, 250]
        trade_type_list = ['P', 'S']
        user_price_list = [99.5, 101.250]
        trade_datetime_list = ['2024-08-01 12:00:00', '2024-08-01 12:00:00']
        run_tests_with_and_without_file_compliance_without_realtime(lambda with_file: _test_cusip_order_preserved(cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list, with_file=with_file))


@run_multiple_times_before_failing
def test_same_trade_datetime_reference_data_redis():
    '''Tests that compliance returns a valid response when all CUSIPs have the same trade datetime. Since the 
    trade datetime is after 2024-09-24, which is defined in `point_in_time_pricing.py::DATE_FROM_WHICH_PAST_REFERENCE_DATA_IS_STORED_IN_REDIS`, 
    redis will be used to create the trade history and similar trade history and the reference data.'''
    cusip_list = ['64971XQM3', '950885SN4']
    quantity_list = [100, 250]
    trade_type_list = ['P', 'S']
    user_price_list = [99.5, 101.250]
    trade_datetime_list = ['2024-10-03 12:00:00', '2024-10-03 12:00:00']
    run_tests_with_and_without_file_compliance_without_realtime(lambda with_file: _test_cusip_order_preserved(cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list, with_file=with_file))


@run_multiple_times_before_failing
def test_reference_data_redis_error_cusip():
    '''Tests that compliance returns a valid response when all CUSIPs have the same trade datetime. Since the 
    trade datetime is after 2024-09-24, which is defined in `point_in_time_pricing.py::DATE_FROM_WHICH_PAST_REFERENCE_DATA_IS_STORED_IN_REDIS`, 
    redis will be used to create the trade history and similar trade history and the reference data.
    NOTE: this failed before when introducing support for inter-dealer'''
    cusip_list = ['75724PAL5']    # error: "The quantity attempting to be priced is larger than the amount outstanding of ..."
    quantity_list = [1000]
    trade_type_list = ['P']
    user_price_list = [101.675]
    trade_datetime_list = ['2025-01-05 12:00:00']
    run_tests_with_and_without_file_compliance_without_realtime(lambda with_file: _test_cusip_order_preserved(cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list, with_file=with_file))
    run_tests_with_and_without_file_compliance_without_realtime(lambda with_file: _test_cusip_order_preserved(cusip_list, quantity_list, trade_type_list, user_price_list, with_file=with_file))    # realtime
    cusip_list = ['75724PAL5', '75724PAR2']
    quantity_list = [1000, 35]
    run_tests_with_and_without_file_compliance_without_realtime(lambda with_file: _test_cusip_order_preserved(cusip_list, quantity_list, trade_type_list * 2, user_price_list * 2, trade_datetime_list * 2, with_file=with_file))
    run_tests_with_and_without_file_compliance_without_realtime(lambda with_file: _test_cusip_order_preserved(cusip_list, quantity_list, trade_type_list * 2, user_price_list * 2, with_file=with_file))    # realtime


@run_multiple_times_before_failing
def test_realtime():
    '''Tests that compliance returns a valid response when all CUSIPs are priced in realtime (no trade datetime provided).'''
    cusip_list = ['64971XQM3', '950885SN4']
    quantity_list = [100, 250]
    trade_type_list = ['P', 'S']
    user_price_list = [99.5, 101.250]
    run_tests_with_and_without_file_compliance_with_realtime(lambda with_file: _test_cusip_order_preserved(cusip_list, quantity_list, trade_type_list, user_price_list, with_file=with_file))


@run_multiple_times_before_failing
def test_past_and_future_trade_datetimes():
    '''Tests that compliance returns a valid response when the CUSIPs are attempted to be priced way in the past or the future. 
    In this case, the CUSIPs should be priced in realtime.'''
    cusip_list = ['64971XQM3', '64971XQM3', '64971XQM3']
    quantity_list = [100, 100, 100]
    trade_type_list = ['S', 'S', 'S']
    user_price_list = [99.5, 99.5, 99.5]
    trade_datetime_list = ['2024-02-01 12:00:00', '2027-02-01 12:00:00', '']    # all of these should price to realtime and so should be identical in the final output
    run_tests_with_and_without_file_compliance_with_realtime(lambda with_file: _test_cusip_order_preserved(cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list, with_file=with_file, features_to_check_if_identical=['trade_datetime']))

@run_multiple_times_before_failing
def test_additional_trade_datetimes():
    '''To ensure that point-in-time pricing is always supported, there should be additional compliance tests that test a variety of dates,
    especially during the window of 2025-05-19 to 2025-07-01, during which the yield spread model training was put on hold due to a Jira Task.'''
    cusip_list = ['646039YM3', '646039YM3', '13063D7Q5','13063D7Q5']
    quantity_list = [100, 100, 100, 100]
    trade_type_list = ['S', 'S', 'S', 'S']
    user_price_list = [99.5, 99.5, 99.5, 99.5]
    trade_datetime_list = ['2025-05-21', '2025-06-02', '2025-06-11', '2025-06-30'] 
    run_tests_with_and_without_file_compliance_without_realtime(lambda with_file: _test_cusip_order_preserved(cusip_list, quantity_list, trade_type_list, user_price_list, trade_datetime_list, with_file=with_file))


@run_multiple_times_before_failing
def test_missing_user_trade_price():
    '''Tests that compliance returns a valid response when some of the CUSIPs are missing a user trade price.'''
    cusip_list = ['64971XQM3', '950885SN4']
    quantity_list = [100, 250]
    trade_type_list = ['P', 'S']
    user_price_list = ['', '']
    run_tests_with_and_without_file_compliance_with_realtime(lambda with_file: _test_cusip_order_preserved_and_all_refused_to_price(cusip_list, quantity_list, trade_type_list, user_price_list, with_file=with_file))


@run_multiple_times_before_failing
def test_empty_csv():
    '''Tests that an empty csv does not cause a compliance error. Heavily based off 
    ficc/app_engine/demo/server/tests/unit/test_batch_pricing_succeeds.py::test_empty_csv.'''
    filename = f'{DIRECTORY}/empty_compliance.csv'
    with open(filename, 'w') as _: pass    # creating an empty CSV file: https://www.tutorialspoint.com/How-to-create-an-empty-file-using-Python
    request_obj = response_from_compliance(filename)
    assert request_obj.ok, 'The response from compliance for an empty CSV was not successful'    # successful response; checks whether the status_code is less than 400
