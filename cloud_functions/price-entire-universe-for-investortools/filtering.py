'''
Description: Functions that allow filtering of the entire list of CUSIPs.
'''
import os
import pickle
import multiprocess as mp
from tqdm import tqdm

import pandas as pd

from google.cloud import storage

from auxiliary_variables import MULTIPROCESSING, REFERENCE_DATA_REDIS_CLIENT, GOOGLE_CLOUD_BUCKET, NOT_OUTSTANDING_CUSIPS_PICKLE_FILENAME
from auxiliary_functions import function_timer, download_pickle_file


def is_outstanding(reference_data):
    '''This condition is taken directly from `ficc/app_engine/demo/server/modules/data_preparation_for_pricing.py::get_data_for_single_cusip(...).'''
    outstanding_indicator = reference_data['outstanding_indicator']
    return pd.notna(outstanding_indicator) and outstanding_indicator == True    # must use `==` instead of `is` since the data type is `np.bool_` and `is` checks if it is the exact same object as the Python boolean literal


def has_not_defaulted(reference_data):
    '''This condition is taken directly from `ficc/app_engine/demo/server/modules/data_preparation_for_pricing.py::get_data_for_single_cusip(...).'''
    default_exists, default_indicator = reference_data['default_exists'], reference_data['default_indicator']
    return pd.notna(default_exists) and default_exists == False and pd.notna(default_exists) and default_indicator == False    # must use `==` instead of `is` since the data type is `np.bool_` and `is` checks if it is the exact same object as the Python boolean literal


def has_a_regular_coupon_rate(reference_data):
    '''This condition is taken directly from `ficc/app_engine/demo/server/modules/data_preparation_for_pricing.py::get_data_for_single_cusip(...)::irregular_coupon_rate(...).'''
    interest_payment_frequency = reference_data['interest_payment_frequency']
    coupon_type = reference_data['coupon_type']
    return pd.notna(interest_payment_frequency) and interest_payment_frequency in (1, 2, 3, 5, 16) and pd.notna(coupon_type) and coupon_type in (3, 4, 8, 17, 23, 24)


def is_not_close_to_maturing(reference_data):
    '''This condition is a variation of the condition in `ficc/app_engine/demo/server/modules/data_preparation_for_pricing.py::price_cusips_list(...), that 
    checks whether the CUSIP is at least 60 days away from the calculation date. In this case, we make sure that the CUSIP is at least 60 days away from the 
    maturity date.
    NOTE: did not implement this yet since creating the `days_to_maturity_date` feature is not immediately trivial.'''
    raise NotImplementedError


@function_timer
def filter_cusips_with_errors(cusip_list):
    '''Filter each CUSIP in `cusip_list` with some of the restrictions that cause the product to 
    refuse to price the CUSIP. This will help speed up the pricing procedure further downstream. 
    The current restrictions checked here are (1) if the CUSIP is outstanding, (2) if it has not 
    defaulted, (3) has a regular coupon rate, and (4) if it is not close to maturing.'''
    def apply_all_filters(reference_data):
        '''Helper function using an and clause to enforce all of the filtering functions.'''
        return all([func(reference_data) for func in [is_outstanding, has_not_defaulted, has_a_regular_coupon_rate]])
    
    print(f'Before filtering, there are {len(cusip_list)} CUSIPs')
    if MULTIPROCESSING and len(cusip_list) > os.cpu_count():
        print(f'Using multiprocessing with {os.cpu_count()} cores inside `filter_cusips_with_errors(...)`')
        with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
            reference_data_list = tqdm(pool_object.imap(lambda cusip: pickle.loads(REFERENCE_DATA_REDIS_CLIENT.get(cusip)[0]), cusip_list), total=len(cusip_list))    # index 0 indicates the most recent snapshot of the reference data
        with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
            cusip_list = tqdm(pool_object.imap(apply_all_filters, zip(cusip_list, reference_data_list)), total=len(cusip_list))
    else:
        print(f'Not using multiprocessing inside `filter_cusips_with_errors(...)`')
        reference_data_list = [pickle.loads(REFERENCE_DATA_REDIS_CLIENT.get(cusip)[0]) for cusip in tqdm(cusip_list, total=len(cusip_list))]    # index 0 indicates the most recent snapshot of the reference data
        cusip_list = [cusip for cusip, reference_data in tqdm(zip(cusip_list, reference_data_list), total=len(cusip_list)) if apply_all_filters(reference_data)]
    print(f'After filtering, there are {len(cusip_list)} CUSIPs')
    return cusip_list


def remove_not_outstanding_cusips(cusip_list, use_pickle_file: bool = True):
    '''Use the pickle file storing all of the not outstanding CUSIPs to filter out `cusip_list`.'''
    assert use_pickle_file is True, 'Do not yet have functionality for not using the pickle file, i.e., `use_pickle_file` must be `True`'
    not_outstanding_cusips_set = download_pickle_file(storage.Client(), GOOGLE_CLOUD_BUCKET, NOT_OUTSTANDING_CUSIPS_PICKLE_FILENAME)
    num_cusips = len(cusip_list)
    cusip_list = [cusip for cusip in cusip_list if cusip not in not_outstanding_cusips_set]
    print(f'{num_cusips - len(cusip_list)} CUSIPs were removed because these CUSIPs are not outstanding')
    return cusip_list
