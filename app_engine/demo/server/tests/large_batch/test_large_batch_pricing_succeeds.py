'''
'''
from modules.ficc.utils.auxiliary_functions import sqltodf

from modules.test.auxiliary_variables import DIRECTORY
from modules.test.auxiliary_functions import response_from_batch_pricing, get_bq_client, check_that_batch_pricing_gives_output_for_all_cusips, check_that_batch_pricing_gives_price_for_all_cusips


LIMIT = 1000


def _check_that_batch_pricing_gives_output_for_all_cusips_for_file(filename_wo_directory_or_extension):
    filename = f'{DIRECTORY}/{filename_wo_directory_or_extension}.csv'
    request_obj, cusip_list, _ = response_from_batch_pricing(filename, return_cusip_list_and_quantity_list=True)
    check_that_batch_pricing_gives_output_for_all_cusips(request_obj, cusip_list)


def test_random_1000_to_price():
    '''Tests 1000 CUSIPs that we should be able to price.'''
    bq_client = get_bq_client()

    query = f'''
        SELECT DISTINCT cusip
        FROM `reference_data_v2.reference_data_flat`
        WHERE maturity_description_code = 2
          AND maturity_date > current_date
          AND next_call_date > current_date
          AND refund_date > current_date
          AND (interest_payment_frequency = 1 OR interest_payment_frequency = 16)
          AND ref_valid_to_date > current_timestamp
          AND (coupon_type = 8 OR coupon_type = 17 OR coupon_type = 4 OR coupon_type = 10)
          AND outstanding_indicator is true
        LIMIT {LIMIT}'''
    df = sqltodf(query, bq_client)
    cusip_list = df['cusip'].tolist()
    filename = f'{DIRECTORY}/{LIMIT}cusips_to_price.csv'
    request_obj = response_from_batch_pricing(filename, cusip_list)
    check_that_batch_pricing_gives_price_for_all_cusips(request_obj, cusip_list)


def test_random_1000_may_not_support():
    '''Tests 1000 CUSIPs of which there may be some that we do not support. The goal here is to 
    make sure that batch pricing CUSIPs that we do not support will not break the product.'''
    bq_client = get_bq_client()

    query = f'''
        SELECT DISTINCT cusip
        FROM `reference_data_v2.reference_data_flat`
        WHERE maturity_description_code = 2
          AND maturity_date > current_date
          AND next_call_date > current_date
          AND refund_date > current_date
          AND (interest_payment_frequency = 1 OR interest_payment_frequency = 16)
          AND ref_valid_to_date > current_timestamp
          AND (coupon_type = 8 OR coupon_type = 17 OR coupon_type = 4 OR coupon_type = 10)
          AND outstanding_indicator is true
        LIMIT {LIMIT}'''
    df = sqltodf(query, bq_client)
    cusip_list = df['cusip'].tolist()
    filename = f'{DIRECTORY}/{LIMIT}cusips_may_not_support.csv'
    request_obj = response_from_batch_pricing(filename, cusip_list)
    check_that_batch_pricing_gives_output_for_all_cusips(request_obj, cusip_list)


def test_6_12_50k():
    '''Tests that batch pricing all 50000 CUSIPs in 6-12_50k.csv provides an output (no batch pricing error).'''
    _check_that_batch_pricing_gives_output_for_all_cusips_for_file('6-12_50k')


def test_100k():
    '''Tests that batch pricing all 100000 CUSIPs in 100k.csv provides an output (no batch pricing error).'''
    _check_that_batch_pricing_gives_output_for_all_cusips_for_file('100k')
