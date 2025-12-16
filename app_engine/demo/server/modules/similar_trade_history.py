'''
Description: Functions that support processing and using similar trade history data.
'''
import warnings
import pickle
import pandas as pd
# import traceback    # used to print error stack trace; cannot print traceback because Google Cloud logging interprets it as an actual error that has not been caught causing alerts to go off

from modules.ficc.utils.diff_in_days import diff_in_days_two_dates
from modules.ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR

from modules.auxiliary_variables import FEATURES_NECESSARY_FOR_CREATING_SIMILAR_TRADE_HISTORY_GROUPS, SIMILAR_TRADE_HISTORY_REDIS_CLIENT, REFERENCE_DATA_FEATURE_TO_INDEX
from modules.auxiliary_functions import get_feature_value
from modules.issue_key_map import ISSUE_KEY_MAP


def is_old_issue_key(issue_key: str) -> bool:
    '''Determines whether `issue_key` is from previous reference data, which encodes it as an integer, 
    i.e., checks whether `issue_key` can be converted to an integer.
    
    >>> is_old_issue_key('123')
    True
    >>> is_old_issue_key(123)
    True
    >>> is_old_issue_key(123.0)
    True
    >>> is_old_issue_key('abc')
    False
    '''
    try:
        int(issue_key)
        return True
    except ValueError:
        return False


def get_similar_trade_history_group(reference_data, current_date: pd.Timestamp) -> tuple:
    '''Returns the group identifying the similar trade history corresponding to `reference_data` on 
    `current_date`, i.e., returns a tuple with 3 times: (1) issue_key, (2) maturity bucket, (3) coupon 
    bucket. May raise an error if certain features are not found, and so the error handling should be 
    done further downstream.'''
    issue_key = get_feature_value(reference_data, 'issue_key')
    #COMMENTING OUT AS THIS IS CREATING TOO MUCH LOGGING 2025-08-28
    # if is_old_issue_key(issue_key):
    #     warnings.warn(f'Using an old issue key: {issue_key} for CUSIP: {get_feature_value(reference_data, "cusip")}', RuntimeWarning)
    #     issue_key = int(issue_key)
    #     issue_key = ISSUE_KEY_MAP.get(issue_key, issue_key)    # map `issue_key` to a new value if possible, otherwise, keep as is to retain point-in-time pricing functionality

    maturity_date = get_feature_value(reference_data, 'maturity_date')
    years_to_maturity_date_by_5 = (diff_in_days_two_dates(maturity_date, current_date, convention='exact') // NUM_OF_DAYS_IN_YEAR) // 5
    coupon = get_feature_value(reference_data, 'coupon')
    coupon_by_1 = -1 if coupon == 0 else coupon // 1
    return issue_key, int(years_to_maturity_date_by_5), int(coupon_by_1)    # this line will raise a `ValueError: cannot convert float NaN to integer` if `coupon_by_1` is `np.nan`; the previous line does not raise an error because `np.nan // 1` equals `np.nan`


def similar_group_to_similar_key(features: list):
    '''Converts the group of features (represented as a tuple) for which all similar trades are marked 
    into a primitive type (e.g. string) that can be used as a key for Redis. Redis does not allow tuples 
    to be used as keys. The current mapping concatenates the features into a single string separated by 
    underscores.
    NOTE: this function needs to be identical to `features_to_string(...)` defined in `update_similar_trade_history_redis(...)` 
    in `cloud_functions/fast_trade_history_redis_update/main.py`.'''
    return '_'.join([str(feature) for feature in features])


def get_similar_trade_history_data(reference_data, cusip: str, current_date: pd.Timestamp) -> list:
    '''Get similar trade history data from the similar trade history redis. The `cusip` argument is only used 
    for print line debugging.'''
    if reference_data is None: return []
    try:
        similar_trade_history_group = get_similar_trade_history_group(reference_data, current_date)
        similar_trade_history_group = similar_group_to_similar_key(similar_trade_history_group)
        similar_trade_history_data = pickle.loads(SIMILAR_TRADE_HISTORY_REDIS_CLIENT.get(similar_trade_history_group)) if SIMILAR_TRADE_HISTORY_REDIS_CLIENT.exists(similar_trade_history_group) else []
    except Exception as e:
        #COMMENTING OUT AS THIS IS CREATING TOO MUCH LOGGING 2025-08-28
        # if type(reference_data) == str:
        #     print(f'Unable to get reference features for CUSIP: {cusip} since reference data and / or trade history was not found in the corresponding redis. {type(e)}: {e}')
        # else:
        #     try:
        #         error_message_suffix = '\t'.join([f'{feature}: {get_feature_value(reference_data, feature)}' for feature in FEATURES_NECESSARY_FOR_CREATING_SIMILAR_TRADE_HISTORY_GROUPS])
        #     except Exception:
        #         error_message_suffix = f'At least one of {FEATURES_NECESSARY_FOR_CREATING_SIMILAR_TRADE_HISTORY_GROUPS} is missing'
        #     print(f'Unable to create a similar trade group for CUSIP: {cusip} due to {type(e)}: {e}. {error_message_suffix}')
        # print(traceback.format_exc())    # cannot print traceback because Google Cloud logging interprets it as an actual error that has not been caught causing alerts to go off
        similar_trade_history_data = []
    return similar_trade_history_data
