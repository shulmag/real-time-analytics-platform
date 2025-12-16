'''
'''
import functions_framework

import time
from datetime import timedelta
import smtplib
from email.mime.text import MIMEText

from auxiliary_functions import access_secret_version

FAILURE_EMAIL_RECIPIENTS = ['ficc-eng@ficc.ai', 'eng@ficc.ai']
SUCCESS_EMAIL_RECIPIENTS = FAILURE_EMAIL_RECIPIENTS

TIME_THRESHOLD_TO_COMPLETE_ALL_TESTS = 60 * 30    # in seconds


def run_tests(tests_to_run, print_time_per_test=False):
    tests_run = []
    if print_time_per_test: time_per_test = []    # stores the time taken to run each test
    try:
        start_time = time.time()
        for test in tests_to_run:
            tests_run.append(test.__name__)
            if print_time_per_test: test_start_time = time.time()
            test()
            if print_time_per_test: time_per_test.append(round(time.time() - test_start_time, 2))
        return (timedelta(seconds=time.time() - start_time), tests_run, time_per_test) if print_time_per_test else (timedelta(seconds=time.time() - start_time), tests_run)
    except AssertionError as e:
        return timedelta(seconds=time.time() - start_time), f'{tests_run[-1]} failed: {e}'
    except Exception as e:
        return timedelta(seconds=time.time() - start_time), f'Error in implementation of {tests_run[-1]}: {e}'


def test_individual_pricing_succeeds():
    from test_individual_pricing_succeeds import test_64971XQM3_all_trade_types, test_assorted_8_digit, test_successful_pricing, test_not_outstanding, test_bonds_with_maturity_date_within_60_days, test_defaulted, test_private_placement_or_bank_loan, test_high_yield_in_history, test_irregular_coupon_rate, test_missing_or_negative_yields, test_pac_bond, test_quantity_greater_than_outstanding_amount, test_under_review, test_invalid_check_digit, test_extra_spaces
    return run_tests([test_64971XQM3_all_trade_types, test_assorted_8_digit, test_successful_pricing, test_not_outstanding, test_bonds_with_maturity_date_within_60_days, test_defaulted, test_private_placement_or_bank_loan, test_high_yield_in_history, test_irregular_coupon_rate, test_missing_or_negative_yields, test_pac_bond, test_quantity_greater_than_outstanding_amount, test_under_review, test_invalid_check_digit, test_extra_spaces])


def test_batch_pricing_succeeds():
    from test_batch_pricing_succeeds import test_64971XQM3, test_not_outstanding, test_lowercase_tradetype_inputs, test_decimal_quantity_inputs, test_quantity_tradetype_inputs, test_quantity_inputs_order_preserved_random, test_quantity_inputs_order_preserved_645002XL5_13068LGH2_797272QY0, test_default_quantity_and_trade_type_are_maintained, test_assorted, test_assorted_8_digit, test_excel_scientific_notation, test_bonds_with_missing_or_negative_yields_in_history, test_bonds_where_quantity_is_greater_than_outstanding_amount, test_bonds_with_maturity_date_within_60_days, test_pac_bonds, test_different_kinds_of_bonds_in_one_batch, test_yield_spread_model_and_dollar_price_model_together, test_caching, test_empty_csv, test_empty_line_in_csv, test_missing_cusips_in_csv, test_only_missing_cusips_in_csv, test_large_batch
    return run_tests([test_64971XQM3, test_not_outstanding, test_lowercase_tradetype_inputs, test_decimal_quantity_inputs, test_quantity_tradetype_inputs, test_quantity_inputs_order_preserved_random, test_quantity_inputs_order_preserved_645002XL5_13068LGH2_797272QY0, test_default_quantity_and_trade_type_are_maintained, test_assorted, test_assorted_8_digit, test_excel_scientific_notation, test_bonds_with_missing_or_negative_yields_in_history, test_bonds_where_quantity_is_greater_than_outstanding_amount, test_bonds_with_maturity_date_within_60_days, test_pac_bonds, test_different_kinds_of_bonds_in_one_batch, test_yield_spread_model_and_dollar_price_model_together, test_caching, test_empty_csv, test_empty_line_in_csv, test_missing_cusips_in_csv, test_only_missing_cusips_in_csv, test_large_batch])


def test_individual_pricing_equals_batch_pricing():
    from test_individual_pricing_equals_batch_pricing import test_64971XQM3_different_trade_types_and_quantities, test_yield_spread_model_cusips, test_dollar_price_model_cusips
    return run_tests([test_64971XQM3_different_trade_types_and_quantities, test_yield_spread_model_cusips, test_dollar_price_model_cusips])


def test_bid_ask_spreads():
    from test_bid_ask_spreads import test_cusips_from_jim_perrello, test_64972GWZ3
    return run_tests([test_cusips_from_jim_perrello, test_64972GWZ3])
    

def test_similar_bonds_succeeds():
    from test_similar_bonds_succeeds import test_similar_bonds_64971XQM3, test_no_similar_bonds
    return run_tests([test_similar_bonds_64971XQM3, test_no_similar_bonds])


def test_yield_curve():
    from test_yield_curve import test_yield_curve_plot_at_current_datetime, test_yield_curve_table_at_current_datetime
    return run_tests([test_yield_curve_plot_at_current_datetime, test_yield_curve_table_at_current_datetime])
    

def test_logging():
    from test_logging import test_logging_individual, test_logging_batch_success_error, test_logging_batch_yield_spread_dollar_price
    return run_tests([test_logging_individual, test_logging_batch_success_error, test_logging_batch_yield_spread_dollar_price])


def test_speed():
    from test_speed import test_64971XQM3, test_98322QPL5, test_batch_100
    return run_tests([test_64971XQM3, test_98322QPL5, test_batch_100], print_time_per_test=True)    # outputs the amount of time each test takes


def test_compliance():
    from test_compliance import test_same_trade_datetime_reference_data_redis, test_reference_data_redis_error_cusip, test_realtime, test_duplicate_cusips_different_trade_datetimes_realtime_all_redis, test_past_and_future_trade_datetimes, test_additional_trade_datetimes, test_missing_user_trade_price, test_empty_csv
    return run_tests([test_same_trade_datetime_reference_data_redis, test_reference_data_redis_error_cusip, test_realtime, test_duplicate_cusips_different_trade_datetimes_realtime_all_redis, test_past_and_future_trade_datetimes, test_additional_trade_datetimes, test_missing_user_trade_price, test_empty_csv])

def test_model_correctness():
    from test_model_correctness import compare_avg_spread
    return run_tests([compare_avg_spread])

def send_email(subject, message, recipients=FAILURE_EMAIL_RECIPIENTS):
    sender_email = access_secret_version('notifications_username')
    password = access_secret_version('notifications_password')
    smtp_server = 'smtp.gmail.com'
    port = 587
    
    server = smtplib.SMTP(smtp_server, port)
    server.starttls()
    server.login(sender_email, password)
    
    message = MIMEText(message)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = ', '.join(recipients)
    
    try:
        server.sendmail(sender_email, recipients, message.as_string())
    except Exception as e:
        print(e)
    server.quit()


def remove_hours_and_fractional_seconds(time):
    time = str(time).split('.')[0]    # remove the fractional seconds
    return time[time.find(':') + 1:]    # remove the hours


@functions_framework.http
def hello_http(request, send_success_email_but_only_when_slow: bool = True):    # first argument needs to be `request` for the cloud function to work properly on Google cloud
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
    '''
    all_error_messages = ''
    all_success_messages = ''
    num_groups_of_tests_failed = 0    # counts the number of groups of tests that have failed; if a large number of groups failed, then the product is down and/or there is a critical error that needs to be addressed

    def create_print_output(test_objective, time_and_message):
        output_time_per_test = len(time_and_message) == 3
        if output_time_per_test:
            time, message, time_per_test = time_and_message
        else:
            time, message = time_and_message
        
        time = remove_hours_and_fractional_seconds(time)
        nonlocal all_error_messages, all_success_messages, num_groups_of_tests_failed
        if type(message) == list:
            if output_time_per_test: message = [test_message + f' took {seconds} seconds' for test_message, seconds in zip(message, time_per_test)]
            added_message = f'Succeeded to {test_objective} in {time}. The following {len(message)} tests passed:\n' + '\n'.join(message) + '\n\n'
            all_success_messages += added_message
        else:
            added_message = f'Failed to {test_objective} in {time}.\n' + message + '\n\n'
            all_error_messages += added_message
            num_groups_of_tests_failed += 1
        print(added_message)
        return len(message)    # meaningful only when all the tests have passed since `message` is a list in this case only

    print('BEGIN automated testing')
    num_tests_total = 0
    start_time = time.time()
    # num_tests_total += create_print_output('log activity', test_logging())    # removing logging tests because the BigQuery calls are expensive (~$1.50 per test ==> ~$24 per day); run logging tests first so to avoid quota errors which arise from repeatedly attempting to write to the BigQuery table: `<class 'google.api_core.exceptions.Forbidden'>:403 Quota exceeded: Your table exceeded quota for imports or query appends per table. For more information, see https://cloud.google.com/bigquery/docs/troubleshoot-quotas; reason: quotaExceeded, location: load_job_per_table.long, message: Quota exceeded: Your table exceeded quota for imports or query appends per table. For more information, see https://cloud.google.com/bigquery/docs/troubleshoot-quotas`
    num_tests_total += create_print_output('individually price CUSIPs', test_individual_pricing_succeeds())
    num_tests_total += create_print_output('batch price CUSIPs', test_batch_pricing_succeeds())
    num_tests_total += create_print_output('match individual prices and batch prices', test_individual_pricing_equals_batch_pricing())
    num_tests_total += create_print_output('make sure that bid ask spreads had the correct sign', test_bid_ask_spreads())
    num_tests_total += create_print_output('find similar bonds', test_similar_bonds_succeeds())
    num_tests_total += create_print_output('see yield curve', test_yield_curve())
    num_tests_total += create_print_output('price quickly', test_speed())
    num_tests_total += create_print_output('run the compliance module', test_compliance())
    num_tests_total += create_print_output('model correctness (avg spread)', test_model_correctness())
    time_elapsed = time.time() - start_time
    too_slow = True if time_elapsed > TIME_THRESHOLD_TO_COMPLETE_ALL_TESTS else False
    time_elapsed = remove_hours_and_fractional_seconds(timedelta(seconds=time.time() - start_time))
    print(f'END automated testing in {time_elapsed}')

    email_subject = '[SLOW] ' if too_slow else ''
    email_body_slow_addendum = '' if not too_slow else 'This took longer than expected to complete. Check the speed tests. Check if this occurred in the previous run.\n\n\n'
    if num_groups_of_tests_failed > 0:    # there is an automated test failure
        if num_groups_of_tests_failed >= 4: email_subject = '[CRITICAL] Many automated tests failed; product is down!! ' + email_subject
        email_subject += f'Product Failed in {time_elapsed}'
        send_email(email_subject, email_body_slow_addendum + all_error_messages)
        status = f'Failed in {time_elapsed}\n\n\n{all_error_messages}'
    else:
        email_subject += f'Product Succeeded in {time_elapsed}'
        if send_success_email_but_only_when_slow and too_slow: send_email(email_subject, email_body_slow_addendum + f'All {num_tests_total} tests passed\n\n\n' + all_success_messages, SUCCESS_EMAIL_RECIPIENTS)
        status = f'Succeeded in {time_elapsed}\n\n\n{all_success_messages}'
    
    print(status)
    return status