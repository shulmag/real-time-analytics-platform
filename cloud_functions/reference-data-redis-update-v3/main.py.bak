import functions_framework

import os
from functools import wraps
from collections import deque    # used to easily remove items from the back
import time
import logging as python_logging

import pickle
import numpy as np
import pandas as pd
import redis
from datetime import datetime, timedelta

from google.cloud import bigquery, logging


# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/user/ficc/ficc/mitas_creds.json'


TESTING = False
if TESTING:
    python_logging.info = print    # using `print` to see output in console
    python_logging.debug = print    # using `print` to see output in console
else:    # do not need logging client if testing the function locally since all of the output will be to the console
    # set up logging client; https://cloud.google.com/logging/docs/setup/python
    logging_client = logging.Client()
    logging_client.setup_logging(log_level=python_logging.DEBUG)    # need to set the `log_level` to `DEBUG` since the default `log_level` for `client.setup_logging(...)` in the `google-cloud-logging` library is INFO, meaning that only logs at the level of INFO and higher (e.g., WARNING, ERROR, and CRITICAL) will be captured and sent to Google Cloud Logging

BQ_CLIENT = bigquery.Client()

YEAR_MONTH_DAY = '%Y-%m-%d'

REFERENCE_DATA_TABLE_NAME = 'eng-reactor-287421.reference_data_v2.reference_data_flat'
REFERENCE_DATA_REDIS_CLIENT = redis.Redis(host='10.108.4.37', port=6379, db=0)

FIRST_RECORD_VALID_FROM_DATE = '2010-01-01'    # ficc convention to denote the first entry of the reference data (arbitrary date in the past)
MAX_NUM_DAYS_FOR_REFERENCE_DATA_POINT_IN_TIME = 720    # number of days for which if there are more than reference data updates for a particular CUSIP; this value should be the same as `MAX_NUM_DAYS_FOR_TRADES_IN_HISTORY` in `cloud_functions/fast_trade_history_redis_update/main.py`

# feature names must be identical to `modules/auxiliary_variables.py::REFERENCE_DATA_FEATURES`
REFERENCE_DATA_FEATURES = ['coupon',
                           'cusip',
                           'ref_valid_from_date',
                           'ref_valid_to_date',
                           'incorporated_state_code',
                           'organization_primary_name',
                           'instrument_primary_name',
                           'issue_key',
                           'issue_text',
                           'conduit_obligor_name',
                           'is_called',
                           'is_callable',
                           'is_escrowed_or_pre_refunded',
                           'first_call_date',
                           'call_date_notice',
                           'callable_at_cav',
                           'par_price',
                           'call_defeased',
                           'call_timing',
                           'call_timing_in_part',
                           'extraordinary_make_whole_call',
                           'extraordinary_redemption',
                           'make_whole_call',
                           'next_call_date',
                           'next_call_price',
                           'call_redemption_id',
                           'first_optional_redemption_code',
                           'second_optional_redemption_code',
                           'third_optional_redemption_code',
                           'first_mandatory_redemption_code',
                           'second_mandatory_redemption_code',
                           'third_mandatory_redemption_code',
                           'par_call_date',
                           'par_call_price',
                           'maximum_call_notice_period',
                           'called_redemption_type',
                           'muni_issue_type',
                           'refund_date',
                           'refund_price',
                           'redemption_cav_flag',
                           'max_notification_days',
                           'min_notification_days',
                           'next_put_date',
                           'put_end_date',
                           'put_feature_price',
                           'put_frequency',
                           'put_start_date',
                           'put_type',
                           'maturity_date',
                           'sp_long',
                           'sp_stand_alone',
                           'sp_icr_school',
                           'sp_prelim_long',
                           'sp_outlook_long',
                           'sp_watch_long',
                           'sp_Short_Rating',
                           'sp_Credit_Watch_Short_Rating',
                           'sp_Recovery_Long_Rating',
                           'moodys_long',
                           'moodys_short',
                           'moodys_Issue_Long_Rating',
                           'moodys_Issue_Short_Rating',
                           'moodys_Credit_Watch_Long_Rating',
                           'moodys_Credit_Watch_Short_Rating',
                           'moodys_Enhanced_Long_Rating',
                           'moodys_Enhanced_Short_Rating',
                           'moodys_Credit_Watch_Long_Outlook_Rating',
                           'has_sink_schedule',
                           'next_sink_date',
                           'sink_indicator',
                           'sink_amount_type_text',
                           'sink_amount_type_type',
                           'sink_frequency',
                           'sink_defeased',
                           'additional_next_sink_date',
                           'sink_amount_type',
                           'additional_sink_frequency',
                           'min_amount_outstanding',
                           'max_amount_outstanding',
                           'default_exists',
                           'has_unexpired_lines_of_credit',
                           'years_to_loc_expiration',
                           'escrow_exists',
                           'escrow_obligation_percent',
                           'escrow_obligation_agent',
                           'escrow_obligation_type',
                           'child_linkage_exists',
                           'put_exists',
                           'floating_rate_exists',
                           'bond_insurance_exists',
                           'is_general_obligation',
                           'has_zero_coupons',
                           'delivery_date',
                           'issue_price',
                           'primary_market_settlement_date',
                           'issue_date',
                           'outstanding_indicator',
                           'federal_tax_status',
                           'maturity_amount',
                           'available_denom',
                           'denom_increment_amount',
                           'min_denom_amount',
                           'accrual_date',
                           'bond_insurance',
                           'coupon_type',
                           'current_coupon_rate',
                           'daycount_basis_type',
                           'debt_type',
                           'default_indicator',
                           'first_coupon_date',
                           'interest_payment_frequency',
                           'issue_amount',
                           'last_period_accrues_from_date',
                           'next_coupon_payment_date',
                           'odd_first_coupon_date',
                           'orig_principal_amount',
                           'original_yield',
                           'outstanding_amount',
                           'previous_coupon_payment_date',
                           'sale_type',
                           'settlement_type',
                           'additional_project_txt',
                           'asset_claim_code',
                           'additional_state_code',
                           'backed_underlying_security_id',
                           'bank_qualified',
                           'capital_type',
                           'conditional_call_date',
                           'conditional_call_price',
                           'designated_termination_date',
                           'DTCC_status',
                           'first_execution_date',
                           'formal_award_date',
                           'maturity_description_code',
                           'muni_security_type',
                           'mtg_insurance',
                           'orig_cusip_status',
                           'orig_instrument_enhancement_type',
                           'other_enhancement_type',
                           'other_enhancement_company',
                           'pac_bond_indicator',
                           'project_name',
                           'purpose_class',
                           'purpose_sub_class',
                           'refunding_issue_key',
                           'refunding_dated_date',
                           'sale_date',
                           'sec_regulation',
                           'secured',
                           'series_name',
                           'sink_fund_redemption_method',
                           'state_tax_status',
                           'tax_credit_frequency',
                           'tax_credit_percent',
                           'use_of_proceeds',
                           'use_of_proceeds_supplementary',
                           # 'material_event_history',    # this feature doubles the query cost and is not used in the product
                           # 'default_event_history',    # removed by Developer 2023-05-25
                           # 'most_recent_event',
                           # 'event_exists',
                           'series_id',
                           'security_description',
                           ]


def remove_hours_and_fractional_seconds_beyond_3_digits(time):
    # time = str(time).split('.')[0]    # remove the fractional seconds
    time = str(time)[:-3]    # total of 6 digits after the decimal, so we keep everything but the last 3
    return time[time.find(':') + 1:]    # remove the hours


def function_timer(function_to_time):
    '''This function is to be used as a decorator. It will print out the execution time of `function_to_time`.'''
    @wraps(function_to_time)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
    def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
        python_logging.info(f'Begin execution of {function_to_time.__name__}')    # python_logging.info(f'Begin execution of {function_to_time.__name__}')
        start_time = time.time()
        result = function_to_time(*args, **kwargs)
        end_time = time.time()
        python_logging.info(f'Execution time of {function_to_time.__name__}: {remove_hours_and_fractional_seconds_beyond_3_digits(timedelta(seconds=end_time - start_time))}')    # python_logging.info(f'Execution time of {function_to_time.__name__}: {remove_hours_and_fractional_seconds_beyond_3_digits(timedelta(seconds=end_time - start_time))}')
        return result
    return wrapper


def get_new_reference_data_query() -> str:
    '''Return a query to select the latest reference data for a given cusip. The condition 
    `date(filefile_received_from_provider_timestamp_date) = current_date("America/New_York") AND ref_valid_to_date > current_datetime("America/New_York")` 
    will retrieve reference data from an update that occurred today (hence, `ATE(file_received_from_provider_timestamp) = current_date("America/New_York")`) 
    and data that is most current (hence, `DATETIME(ref_valid_to_date) > CURRENT_DATETIME("America/New_York")`). The `ORDER BY cusip` 
    clause allows us to read and debug more clearly since it is in a replicable order.'''
    return f'''SELECT {", ".join(REFERENCE_DATA_FEATURES)} 
               FROM {REFERENCE_DATA_TABLE_NAME}
               WHERE cusip IS NOT NULL AND DATE(file_received_from_provider_timestamp) = CURRENT_DATE("America/New_York") AND DATETIME(ref_valid_to_date) > CURRENT_DATETIME("America/New_York")
               ORDER BY cusip'''


def sqltodf(sql_query: str) -> pd.DataFrame:
    bqr = BQ_CLIENT.query(sql_query).result()
    return bqr.to_dataframe()


def get_data_from_pickle_file_if_query_matches(query: str, file_name: str) -> pd.DataFrame:
    '''Assume that `file_name` refers to a file that contains a pair: (1) query, (2) data from this query. 
    Check if `query` matches the query in the file from `file_name` in and if so, then return the data. 
    Otherwise, call `sqltodf(...)` on `query` and return the result.'''
    if TESTING and os.path.exists(file_name):
        with open(file_name, 'rb') as file:
            query_from_file, data = pickle.load(file)
        
        if query == query_from_file:
            python_logging.debug(f'query from file matches query so returing data found in {file_name}')
            return data
        else:
            python_logging.debug(f'query from file:\n{query_from_file}\ndoes not match query:\n{query},\nso calling `sqltodf` with `query`')
    else:
        python_logging.debug(f'{file_name} does not exist, so calling `sqltodf` with `query`')
    
    data = sqltodf(query)
    with open(file_name, 'wb') as file:
        pickle.dump((query, data), file)
    return data


@function_timer
def get_new_reference_data() -> pd.DataFrame:
    new_reference_data_query = get_new_reference_data_query()
    python_logging.debug(f'Query to get new reference data:\n{new_reference_data_query}')
    return get_data_from_pickle_file_if_query_matches(new_reference_data_query, 'new_reference_data.pkl') if TESTING else sqltodf(new_reference_data_query)    # stores the query and the data in a pickle file for easy retrieval during testing


def get_differences_between_snapshots(snapshot1: np.ndarray, snapshot2: np.ndarray) -> dict:
    '''Return a dictionary where the key is the index and the value is the value in `snapshot2` 
    corresponding to where `snapshot1` and `snapshot2` differ.'''
    na_mismatch_mask = pd.isna(snapshot1) != pd.isna(snapshot2)    # identify where at least one value is NaN, NaT, or None
    
    valid_mask = ~(pd.isna(snapshot1) | pd.isna(snapshot2))    # identify where both values are NOT NA and are different
    value_diff_mask = snapshot1[valid_mask] != snapshot2[valid_mask]
    
    differences_mask = np.full(snapshot1.shape, False)    # initialize with False
    differences_mask[valid_mask] = value_diff_mask    # mark value differences
    differences_mask[na_mismatch_mask] = True    # mark NA mismatches

    differences_indices = np.where(differences_mask)[0]    # get indices where values are different
    return {idx: snapshot2[idx] for idx in differences_indices}


def get_deque_to_upload_to_redis(cusip: str, reference_data: np.ndarray, valid_from_date: str, reference_data_deque: deque = None) -> deque:
    '''`reference_data_deque` is an optional argument that if passed in, allows us to forgo getting the reference data from the redis.'''
    if TESTING: python_logging.debug(f'Uploading data to redis for CUSIP: {cusip}')
    assert isinstance(valid_from_date, str), f'Expected `valid_from_date`, which has value: {valid_from_date}, to be a string, but got {type(valid_from_date).__name__}'    # previously, we were passing `valid_from_date` as `pd.Timestamp` which was not the intended type, so using this assert statement to ensure this error does not happen again
    if valid_from_date != FIRST_RECORD_VALID_FROM_DATE and (reference_data_deque is not None or REFERENCE_DATA_REDIS_CLIENT.exists(cusip)):    # since `reference_data_deque` is not `None`, we already have the reference data redis deque and do not need to check whether it exists in the redis
        if reference_data_deque is None: reference_data_deque = pickle.loads(REFERENCE_DATA_REDIS_CLIENT.get(cusip))
        
        past_date_threshold = datetime.now().date() - timedelta(days=MAX_NUM_DAYS_FOR_REFERENCE_DATA_POINT_IN_TIME)    # get the date which is `MAX_NUM_DAYS_FOR_REFERENCE_DATA_POINT_IN_TIME` in the past from the current date
        pos_ref_valid_to_date = REFERENCE_DATA_FEATURES.index('ref_valid_to_date')
        
        # iterate backwards through the deque and remove reference data snapshot if the reference data valid to date is older than `MAX_NUM_DAYS_FOR_REFERENCE_DATA_POINT_IN_TIME` in the past from the current date; iterating backwards since reference data numpy arrays are in descending order of `ref_valid_from_date`
        while len(reference_data_deque) > 0 and (reference_data_deque[-1][pos_ref_valid_to_date]).date() < past_date_threshold:    # need to use `.date()` since the `ref_valid_to_date` is a pd.Timestamp
            reference_data_deque.pop()
        
        if len(reference_data_deque) > 0:
            pos_ref_valid_from_date = REFERENCE_DATA_FEATURES.index('ref_valid_from_date')
            reference_data_ref_valid_from_date = reference_data[pos_ref_valid_from_date]
            most_recent_reference_data = reference_data_deque[0]    # reference data numpy arrays are in descending order of `ref_valid_from_date`
            if most_recent_reference_data[pos_ref_valid_from_date] != reference_data_ref_valid_from_date:    # if the `ref_valid_from_date` for most recent reference data is the same as that of the proposed update (`reference_data_deque[0]`), then these snapshots are identical and the update should not happen (i.e., this is not a "new" snapshot)
                differences_between_new_snapshot_and_current_most_recent_snapshot = get_differences_between_snapshots(reference_data, most_recent_reference_data)
                differences_between_new_snapshot_and_current_most_recent_snapshot[pos_ref_valid_to_date] = reference_data_ref_valid_from_date    # update the `ref_valid_to_date` of the most recent item in the deque when updating the deque because it will be set to a datetime far in the future and needs to be set to `ref_valid_from_date` for the new update
                reference_data_deque[0] = differences_between_new_snapshot_and_current_most_recent_snapshot
                reference_data_deque.appendleft(reference_data)    # reference data numpy arrays are in descending order of `ref_valid_from_date`
        else:
            reference_data_deque = deque([reference_data])
    else:
        if valid_from_date == FIRST_RECORD_VALID_FROM_DATE:
            python_logging.debug(f'Creating reference data for the first time for {cusip}')
        else:    # not REFERENCE_DATA_REDIS_CLIENT.exists(cusip)
            python_logging.warning(f'No previous reference data exists in the redis for {cusip} and this reference data snapshot is not the first snapshot in our internal data. Rebuild the redis for this key using `ficc/cloud_functions/reference_data_redis_update_v2/rebuild_redis.py`')
        reference_data_deque = deque([reference_data])

    if TESTING: python_logging.debug(reference_data_deque)
    return reference_data_deque


@function_timer
def update_reference_data_redis(new_reference_data: pd.DataFrame, new_entries_only: bool = False, verbose: bool = False) -> None:
    '''`new_entries_only` is a boolean flag that denotes all entries into the reference data redis to be new entries, e.g., every 
    value will be a one item deque based on the rows in `new_reference_data`.'''
    cusips = new_reference_data['cusip'].tolist()    # due to the `ORDER BY ` clause in the query, `new_reference_data` should be sorted in alphabetical order by CUSIP which is helpful for print statement debugging
    python_logging.debug(f'Updating the reference data for {len(cusips)} cusips: {cusips}')
    
    start_time = time.time()
    total_keys_transferred = 0
    BATCH_SIZE = 1000    # arbitrary selection
    df_chunks = [new_reference_data[idx : idx + BATCH_SIZE] for idx in range(0, len(new_reference_data), BATCH_SIZE)]
    if verbose: print(f'Took {timedelta(time.time() - start_time)} seconds to create `df_chunks`')
    start_time_loop = time.time()

    for chunk in df_chunks:
        start_time_chunk = time.time()
        chunk_cusips = chunk['cusip'].tolist()
        if not new_entries_only: current_reference_data_deques = REFERENCE_DATA_REDIS_CLIENT.mget(chunk_cusips)
        
        with REFERENCE_DATA_REDIS_CLIENT.pipeline() as pipe:
            for row_idx, row in chunk.iterrows():    # `row_idx` is the index value of the row, i.e., the label of `chunk`'s index; we assume the original `new_reference_data` has index values 0...n
                cusip, valid_from_date = row['cusip'], row['ref_valid_from_date'].strftime(YEAR_MONTH_DAY)    # converting `valid_from_date` to a string just for comparison to `FIRST_RECORD_VALID_FROM_DATE` (which is a string) further downstream; using a string for comparison instead of either `datetime` or `pd.Timestamp` because of known issues comparing and converting between `datetime`s and `pd.Timestamp` (e.g., timezone, etc.)
                if new_entries_only:
                    reference_data_deque = pickle.dumps(deque([row]))
                else:
                    current_reference_data_deque = current_reference_data_deques[row_idx % BATCH_SIZE]    # mod with `BATCH_SIZE` since the `row_idx` is the index value of the row so it does not "reset" to 0 but is instead the actual label of `chunk`'s index
                    current_reference_data_deque = pickle.loads(current_reference_data_deque) if current_reference_data_deque is not None else None
                    reference_data_deque = get_deque_to_upload_to_redis(cusip, row.to_numpy(), valid_from_date, current_reference_data_deque)    # use `.to_numpy()` to reduce the space consumption of `row` by 95%
                pipe.set(cusip, pickle.dumps(reference_data_deque))
            pipe.execute()

        total_keys_transferred += len(chunk)
        end_time_chunk = time.time()
        if verbose: print(f'Transferred {total_keys_transferred} so far. Most recent chunk took {timedelta(seconds=end_time_chunk - start_time_chunk)} seconds. Total time elapsed for loop: {timedelta(seconds=end_time_chunk - start_time_loop)} seconds')
    if verbose: print(f'Update complete. Execution time: {timedelta(seconds=time.time() - start_time)}')


@functions_framework.http
def main(request):    # first argument needs to be `request` for the cloud function to work properly on Google cloud
    new_reference_data = get_new_reference_data()
    update_reference_data_redis(new_reference_data)
    return 'SUCCESS'
