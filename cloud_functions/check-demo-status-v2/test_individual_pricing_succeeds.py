'''
'''

from auxiliary_variables import QUANTITY, TRADE_TYPE, CUSIP_ERROR_MESSAGE, DOLLAR_PRICE_MODEL_DISPLAY_TEXT
from auxiliary_functions import run_multiple_times_before_failing, response_from_individual_pricing, check_if_string_value_can_be_represented_as_a_number, check_if_string_value_cannot_be_represented_as_a_number, load_cusips_from_gcs


def check_if_successful_price_received(cusip, trade_type=TRADE_TYPE, quantity=QUANTITY):
    '''Tests that individual pricing successfully returns a price for `cusip` for 
    `trade_type` and `quantity`.'''
    response_dict = response_from_individual_pricing(cusip, trade_type, quantity)
    assert 'error' not in response_dict, f'For CUSIP {cusip}, quantity (in thousands) {quantity}, and trade type {trade_type}, pricing should not have an error, but has an error: {response_dict["error"]}'


def check_if_successful_price_received_and_dollar_price_model_used(cusip, trade_type=TRADE_TYPE, quantity=QUANTITY, reason_for_dollar_price_model=None):
    '''Tests that individual pricing successfully returns a price for `cusip` for 
    `trade_type` and `quantity`, and also that there is a predicted price, but not a 
    predicted yield, which indicates that the dollar price model was used, since we 
    currently do not predict a yield when using the dollar price model.'''
    response_dict = response_from_individual_pricing(cusip, trade_type, quantity)
    assert 'error' not in response_dict, f'For CUSIP {cusip}, quantity (in thousands) {quantity}, and trade type {trade_type}, pricing should not have an error, but has an error: {response_dict["error"]}'
    response_dict = response_dict[0]    # extracting the dict since there is only a single item since this function is only used for individual pricing
    price, ytw = response_dict['price'], response_dict['ficc_ytw']    # TODO: make sure these are the correct fields in `response_dict`
    check_if_string_value_can_be_represented_as_a_number(price, 'price')
    
    if reason_for_dollar_price_model is not None:
        reason_for_dollar_price_model = DOLLAR_PRICE_MODEL_DISPLAY_TEXT[reason_for_dollar_price_model]
        assert ytw == reason_for_dollar_price_model, f'YTW value should be {reason_for_dollar_price_model}, but was instead {ytw}'
    else:
        check_if_string_value_cannot_be_represented_as_a_number(ytw, 'YTW')


def check_if_error_received(cusip, error_key, trade_type=TRADE_TYPE, quantity=QUANTITY):
    '''Tests that individual pricing successfully returns an error for `cusip` for 
    `trade_type` and `quantity`, and that the error message corresponding to `error_key` 
    is the correct one.'''
    def error_message_matches(response_error, error_key):
        expected_error = CUSIP_ERROR_MESSAGE[error_key]
        if callable(expected_error):    # if `expected_error` is a function, then call it with some dummy input to get a string error message and compare everything before and after the dummy input when checking that the error message is correct
            dummy_input_number = 1
            expected_error = expected_error(dummy_input_number)
            num_chars_before_input = expected_error.index(str(dummy_input_number))
            num_chars_after_input = len(expected_error) - num_chars_before_input - 1
            before_input_matches = expected_error[:num_chars_before_input] == response_error[:num_chars_before_input]
            after_input_matches = expected_error[-num_chars_after_input:] == response_error[-num_chars_after_input:]
            return before_input_matches and after_input_matches
        else:
            return response_error == expected_error

    response_dict = response_from_individual_pricing(cusip, trade_type, quantity)
    assert 'error' in response_dict, f'The response should have an error, since CUSIP {cusip} has issue corresponding to the following error_key: {error_key}'
    assert error_message_matches(response_dict['error'], error_key), f'CUSIP {cusip} should have had an error of {CUSIP_ERROR_MESSAGE[error_key]}, but instead had an error of {response_dict["error"]}'


@run_multiple_times_before_failing
def test_64971XQM3_all_trade_types():
    '''Tests that individual pricing successfully returns a price for 64971XQM3 for all trade types.'''
    trade_types = ['D', 'S', 'P']
    for trade_type in trade_types:
        check_if_successful_price_received('64971XQM3', trade_type)


@run_multiple_times_before_failing
def test_assorted_8_digit():
    '''Tests that individual pricing is successful for 8 digit CUSIPs.'''
    cusip_list = ['64971XQM',
                  '6461367J',
                  '13063DU8',
                  '160429B8',
                  '13063DLJ',
                  '54466HJM',
                  '650036CJ']
    for cusip in cusip_list:
        check_if_successful_price_received(cusip)

    cusip_list_with_error_key = [('40064UAW', 'not_outstanding'), ('431669AR', 'insufficient_data')]
    for cusip, error_key in cusip_list_with_error_key:
        check_if_error_received(cusip, error_key)


@run_multiple_times_before_failing
def test_successful_pricing():
    '''Tests that individual pricing successfully returns a price.'''
    cusip_list = ['13063D7Q5', 
                #   '95648XAN5',    # has a data issue where the bond is called but the `refund_price` is missing
                #   '00037CUQ1',    # CUSIP is now no longer outstanding; previously, this CUSIP was causing errors since some of the trades in the history had a calc date that was the same day as the trade date, which resulted in a `RuntimeWarning: invalid value encountered in scalar divide` inside `modules/ficc/utils/nelson_siegel_model.py:26` and `modules/ficc/utils/nelson_siegel_model.py:28`
                  '646136TN1',    # has a data issue where the outstanding amount is 0, but we should still be able to price it and not raise the error that the amount priced is greater than the outstanding amount
                  '047851DC1']    # `next_coupon_payment_date` is missing because it is a new issue and so the `first_coupon_date` is in the future and should be the `next_coupon_payment_date` which the server now accounts for in `exclusions.py::missing_important_dates_or_dates_are_out_of_bounds(...)`
    for cusip in cusip_list:
        check_if_successful_price_received(cusip)


@run_multiple_times_before_failing
def test_not_outstanding():
    '''Tests that individual pricing successfully returns an error message indicating that the CUSIP is outstanding.'''
    cusip_list = ['40064UAW2', '73358WK60']
    for cusip in cusip_list:
        check_if_error_received(cusip, 'not_outstanding')


@run_multiple_times_before_failing
def test_bonds_with_maturity_date_within_60_days():
    '''Tests that bonds with a maturity date within 60 days return a dollar price but no yield.'''
    cusip_list = load_cusips_from_gcs("short_maturity")
    for cusip in cusip_list:
        check_if_successful_price_received_and_dollar_price_model_used(cusip, reason_for_dollar_price_model='maturing_soon')


@run_multiple_times_before_failing
def test_defaulted():
    '''Tests that individual pricing successfully returns a dollar price and a message indicating that the CUSIP has defaulted. To find more CUSIPs like this, use the following query:
    SELECT DISTINCT cusip, trade_date FROM `auxiliary_views_v2.trade_history_same_issue_5_yr_mat_bucket_1_materialized` WHERE default_exists is True ORDER BY trade_date DESC LIMIT 10.'''
    cusip_list = ['052398GZ1', 
                  '697528AV9', 
                  '71885DDC3', 
                  '130493CK3']
    for cusip in cusip_list:
        check_if_successful_price_received_and_dollar_price_model_used(cusip, reason_for_dollar_price_model='defaulted')


@run_multiple_times_before_failing
def test_private_placement_or_bank_loan():
    '''Tests that individual pricing successfully returns an error message indicating that the bond is a private 
    placement bond or a bank loan. To find more CUSIPs like this, use the following query (the `maturity_date` is 
    used to make sure the CUSIP is not close to maturing and can be used for a while in this automated test without 
    needing replacement):
    `SELECT cusip FROM `reference_data_v2.reference_data_flat` where sale_type = 4 ORDER BY maturity_date DESC LIMIT 10`.'''
    cusip_list = ['684906HS2', 
                  '51855TBD6', 
                  '792892JX0', 
                  '89438UBQ0', 
                  '89438UBN7']
    for cusip in cusip_list:
        check_if_error_received(cusip, 'bank_loan')


@run_multiple_times_before_failing
def test_high_yield_in_history():
    '''Tests that individual pricing successfully returns a dollar price and also a message indicating that the recent yields are 
    abnormally high. To find more CUSIPs like this, use the following query:
    `SELECT yield, cusip, maturity_date, next_call_date FROM `auxiliary_views_v2.trade_history_same_issue_5_yr_mat_bucket_1_materialized` WHERE trade_date > "2024-06-01" and yield > 10 ORDER BY next_call_date, maturity_date DESC LIMIT 10`.'''
    cusip_list = ['594751AM1']
    for cusip in cusip_list:
        check_if_successful_price_received_and_dollar_price_model_used(cusip, reason_for_dollar_price_model='high_yield_in_history')


@run_multiple_times_before_failing
def test_irregular_coupon_rate():
    '''Tests that a CUSIP with coupon_type 10 successfully returns an error message indicating that the bond has 
    an irregular coupon rate.'''
    cusip_list = ['106134AP4', '83412PEH1']
    for cusip in cusip_list:
        check_if_error_received(cusip, 'irregular_coupon_rate')


@run_multiple_times_before_failing
def test_missing_or_negative_yields():
    '''Tests that individual pricing successfully uses a dollar price model for a bond with negative yields in 
    the history.'''
    cusip_list = ['71910EAM1']
    for cusip in cusip_list:
        check_if_successful_price_received_and_dollar_price_model_used(cusip, reason_for_dollar_price_model='missing_or_negative_yields')


@run_multiple_times_before_failing
def test_pac_bond():
    '''Tests that individual pricing successfully uses the yield spread model for a PAC bond. 
    Previously, these were priced with the dollar price model. In the current reference data, 
    there is no indicator to select the PAC bonds.'''
    cusip_list = ['93978TN88']
    for cusip in cusip_list:
        check_if_successful_price_received(cusip)


@run_multiple_times_before_failing
def test_quantity_greater_than_outstanding_amount():
    ''''Tests that individual pricing successfully returns an error message indicating that the quantity of 
    the hypothetical trade is larger than the outstanding amount.'''
    cusip_and_quantity_list = [('014464XR7', 5000)]
    for cusip, quantity in cusip_and_quantity_list:
        check_if_error_received(cusip, 'quantity_greater_than_outstanding_amount', quantity=quantity)


@run_multiple_times_before_failing
def test_under_review():
    '''Tests that individual pricing successfully returns an error message indicating that the bond is under review.'''
    cusip_list = ['798712BK0']
    for cusip in cusip_list:
        check_if_error_received(cusip, 'under_review')


@run_multiple_times_before_failing
def test_invalid_check_digit():
    '''Tests that individual pricing successfully returns an error message indicating that the bond has an invalid check digit.'''
    cusip_list = ['5311127AC']
    for cusip in cusip_list:
        check_if_error_received(cusip, 'invalid_check_digit')


@run_multiple_times_before_failing
def test_extra_spaces():
    '''Tests that individual pricing successfully returns a price for a CUSIP with a leading space or a trailing space.'''
    cusip_list = [' 64971XQM',    # leading space
                  '64971XQM ']    # trailing space
    for cusip in cusip_list:
        check_if_successful_price_received(cusip)
    check_if_successful_price_received(' 64971XQM')
