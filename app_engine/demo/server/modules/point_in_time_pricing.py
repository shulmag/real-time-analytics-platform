'''
Description: This file contains function to support point-in-time (historical) pricing, i.e., how our product would have done if it had been called at a specific datetime in the past.
'''
import os
from collections import deque
import multiprocess as mp
from datetime import datetime, time
from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay

from modules.ficc.utils.auxiliary_functions import function_timer, sqltodf, run_five_times_before_raising_redis_connector_error
from modules.ficc.utils.gcp_storage_functions import upload_data, download_pickle_file

from modules.auxiliary_variables import bq_client, storage_client, YEAR_MONTH_DAY, HOUR_MIN_SEC, USE_SIMILAR_TRADES_MODEL_FOR_YIELD_SPREAD_PREDICTIONS_FOR_POINT_IN_TIME_PRICING, AUXILIARY_VIEWS_DATASET, FEATURES_NECESSARY_FOR_CREATING_SIMILAR_TRADE_HISTORY_GROUPS, REFERENCE_DATA_REDIS_CLIENT, TRADE_HISTORY_REDIS_CLIENT, REFERENCE_DATA_FEATURES, REFERENCE_DATA_FEATURE_TO_INDEX, MAX_NUM_TRADES_IN_HISTORY_TO_DISPLAY_ON_UI, DATE_FROM_WHICH_ALL_PAST_TRADES_ARE_STORED_IN_REDIS, DATE_FROM_WHICH_PAST_REFERENCE_DATA_IS_STORED_IN_REDIS, DATE_FROM_WHICH_MODELS_TRAINED_WITH_NEW_REFERENCE_DATA_IS_IN_PRODUCTION
from modules.auxiliary_functions import create_df_chunks
from modules.similar_trade_history import get_similar_trade_history_data, get_similar_trade_history_group

# from tensorflow import keras    # loading the package in the specific function that needs it to reduce the start up latency if the package is not needed


MODEL_FOLDERS = ('similar_trades_model', 'yield_spread_model', 'dollar_price_model')
MODEL_BUCKET = 'gs://automated_training'

TIME_AFTER_END_OF_CURRENT_BUSINESS_DAY = time(17, 0, 0)    # 17 comes from converting 5pm to military time
MAX_NUM_WEEK_DAYS_IN_THE_PAST_TO_CHECK = 10    # denotes the maximum number of week days back that we go to search for the model before raising an error

MULTIPROCESSING = True
VERBOSE = True

MAX_NUM_OF_CUSIPS_FOR_BIGQUERY = 18000    # represents the maximum number of CUSIPs for which we can do a filtering of the query in BigQuery by CUSIP, i.e., each CUSIP is 9 characters, with additional characters for the comma and spacing between CUSIPs, so generating `IN (CUSIP_1, CUSIP_2, ..., CUSIP_1800)` must be less than 256k characters so that the entire SQL query is able to be run by BigQuery
MAX_NUMBER_OF_CUSIPS_TO_USE_GROUPS_FOR_SIMILAR_TRADE_HISTORY_QUERY_OPTIMIZATION = 1000

LIMIT = 10    # we only use at most 5 trades in the history, and so we should keep a few extra trades in case some of the first 5 trades have issues and we cannot use them (e.g., missing necessary fields); originally this value was set to 32, but reducing to 10 can save BigQuery costs


def load_model_from_date(date: str, folder: str, bucket: str):
    '''When using the `cache_output` decorator, we should not have any optional arguments as this may interfere with 
    how the cache lookup is done (optional arguments may not be put into the args set). `model_cache` is an optional 
    argument that may be a dictionary containing a key of (date, folder, bucket) triple with a corresponding value 
    of the model; this argument is used because multiprocessing does not allow the `cache_output` decorator to work. 
    As of 2024-06-07, we assume that the model name has the entire YYYY-MM-DD in the name.'''
    assert folder in MODEL_FOLDERS
    
    tag = 'v2-' if datetime.strptime(date, YEAR_MONTH_DAY) > DATE_FROM_WHICH_MODELS_TRAINED_WITH_NEW_REFERENCE_DATA_IS_IN_PRODUCTION else ''
    if folder == 'dollar_price_model':
        model_prefix = f'dollar-{tag}'
    elif USE_SIMILAR_TRADES_MODEL_FOR_YIELD_SPREAD_PREDICTIONS_FOR_POINT_IN_TIME_PRICING:
        model_prefix = f'similar-trades-{tag}'
    else:
        model_prefix = ''
    
    bucket_folder_model_path = os.path.join(os.path.join(bucket, folder), f'{model_prefix}model-{date}')    # create path of the form: <bucket>/<folder>/<model>
    base_model_path = os.path.join(bucket, f'{model_prefix}model-{date}')    # create path of the form: <bucket>/<model>
    for model_path in (bucket_folder_model_path, base_model_path):    # iterate over possible paths and try to load the model
        print(f'Attempting to load model from {model_path}')
        try:
            from tensorflow import keras    # loading the package in the specific function that needs it to reduce the start up latency if the package is not needed
            # set environment variables when using the dataset to make predictions
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'    # controls how TensorFlow schedules threads when using GPU devices: TensorFlow creates dedicated threads for each GPU device, allowing independent thread scheduling for each GPU and this can improve performance in multi-GPU setups by avoiding contention between GPUs, as each GPU has its own private threads for execution
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # suppresses TensorFlow log messages at or below a certain severity level: hides all TensorFlow log messages except for errors
            
            model = keras.models.load_model(model_path)
            print(f'Model loaded from {model_path}')
            return model
        except Exception as e:
            print(f'Model failed to load from {model_path} with exception: {e}')


def next_week_day_if_datetime_is_after_business_hours(datetime_of_interest: datetime):
    '''Go to the week day after `datetime_of_interest` if the time is after the end of the business day.'''
    if datetime_of_interest.time() > TIME_AFTER_END_OF_CURRENT_BUSINESS_DAY: datetime_of_interest = datetime_of_interest + BusinessDay(1)    # using `BusinessDay` instead of `CustomBusinessDay` with the `USFederalHolidayCalendar` since we do not want to skip holidays because the desired model may have been created on a holiday, which is fine because that model was trained with data before the holiday
    return datetime_of_interest.replace(hour=0, minute=0, second=0, microsecond=0)


@function_timer
def load_model(datetime_of_interest: datetime, 
               folder: str, 
               max_num_week_days_in_the_past_to_check: int = MAX_NUM_WEEK_DAYS_IN_THE_PAST_TO_CHECK, 
               bucket: str = MODEL_BUCKET, 
               model_cache: dict = None):
    '''This function finds the appropriate model, either in the automated_training directory, or in a special directory.'''
    datetime_of_interest = next_week_day_if_datetime_is_after_business_hours(datetime_of_interest)
    date_string = datetime_of_interest.strftime(YEAR_MONTH_DAY)

    model_cache_key = (date_string, folder, bucket)
    if model_cache is not None and model_cache_key in model_cache:
        if VERBOSE: print(f'Key: {model_cache_key} found in `model_cache`')
        return model_cache[model_cache_key]
    else:
        if model_cache is not None and VERBOSE: print(f'Key: {model_cache_key} not found in `model_cache` which has keys: {model_cache.keys()}')

    for num_business_days_in_the_past in range(max_num_week_days_in_the_past_to_check):
        model_date_string = (datetime_of_interest - BusinessDay(num_business_days_in_the_past)).strftime(YEAR_MONTH_DAY)    # using `BusinessDay` instead of `CustomBusinessDay` with the `USFederalHolidayCalendar` since we do not want to skip holidays because the desired model may have been created on a holiday, which is fine because that model was trained with data before the holiday
        model = load_model_from_date(model_date_string, folder, bucket)
        if model is not None: return model
    raise FileNotFoundError(f'No model for {folder} was found from {date_string} to {model_date_string}')


def create_cusip_in_cusip_list_clause(cusip_list: list = None, table_identifier: str = '') -> str:
    if cusip_list is None: return ''
    if table_identifier != '': table_identifier = table_identifier + '.'    # add the period so that the identifier can be concatenated directly into the reference feature in the same way as if it is an empty string
    cusip_list_as_tuple_string = str(tuple(cusip_list)) if len(cusip_list) > 1 else f'("{cusip_list[0]}")'
    return f' AND {table_identifier}cusip IN {cusip_list_as_tuple_string}'    # use `tuple(...)` to have the string representation with parentheses instead of square brackets


def create_trade_history_array_struct_in_bigquery(table_name: str, table_name_for_dates: str = None) -> str:
    '''Returns the `ARRAY_ARG(STRUCT(...))` containing the features necessary for the trade history array.'''
    if table_name_for_dates is None: table_name_for_dates = table_name
    return f'''ARRAY_AGG( STRUCT( {table_name}.msrb_valid_from_date,    -- can be removed but does not provide great speedup
                    {table_name}.msrb_valid_to_date,    -- can be removed but does not provide great speedup
                    {table_name}.rtrs_control_number,
                    {table_name}.trade_datetime,
                    {table_name}.publish_datetime,    -- can be removed but does not provide great speedup
                    {table_name}.yield,
                    {table_name}.dollar_price,
                    CASE
                        WHEN {table_name}.par_traded IS NULL AND is_trade_with_a_par_amount_over_5MM IS TRUE THEN 5000000
                        ELSE {table_name}.par_traded
                    END
                    AS par_traded,
                    trade_type,
                    is_non_transaction_based_compensation,    -- can be removed but does not provide great speedup
                    is_lop_or_takedown,    -- can be removed but does not provide great speedup
                    brokers_broker,    -- can be removed but does not provide great speedup
                    is_alternative_trading_system,    -- can be removed but does not provide great speedup
                    is_weighted_average_price,    -- can be removed but does not provide great speedup
                    CASE
                        WHEN {table_name}.settlement_date IS NULL AND {table_name}.assumed_settlement_date IS NOT NULL THEN {table_name}.assumed_settlement_date
                        ELSE {table_name}.settlement_date
                    END
                    AS settlement_date,
                    b.calc_date,
                    b.calc_date_selection AS calc_day_cat,
                    {table_name_for_dates}.maturity_date,
                    {table_name_for_dates}.next_call_date,
                    {table_name_for_dates}.par_call_date,
                    {table_name_for_dates}.refund_date,
                    {table_name}.transaction_type,    -- can be removed but does not provide great speedup
                    {table_name}.sequence_number)    -- can be removed but does not provide great speedup
                ORDER BY
                    {table_name}.trade_datetime DESC, {table_name}.publish_datetime DESC, {table_name}.sequence_number DESC
                LIMIT {LIMIT})'''


def create_trade_history_query_join_condition(reference_data_table_name: str, calculation_date_and_price_table_name: str) -> str:
    return f'''ON {reference_data_table_name}.rtrs_control_number = {calculation_date_and_price_table_name}.rtrs_control_number
                    AND {reference_data_table_name}.trade_datetime = {calculation_date_and_price_table_name}.trade_datetime
                    AND {reference_data_table_name}.publish_datetime = {calculation_date_and_price_table_name}.publish_datetime
                    -- AND {reference_data_table_name}.msrb_valid_to_date = {calculation_date_and_price_table_name}.msrb_valid_to_date'''


def create_trade_history_query_where_condition(reference_data_table_name: str, calculation_date_and_price_table_name: str, up_until_datetime) -> str:
    return f'''WHERE {reference_data_table_name}.msrb_valid_to_date > CURRENT_DATETIME("America/New_York")
              AND {calculation_date_and_price_table_name}.msrb_valid_to_date > CURRENT_DATETIME("America/New_York")
              AND {reference_data_table_name}.dollar_price IS NOT NULL
              AND ({reference_data_table_name}.par_traded IS NULL OR {reference_data_table_name}.par_traded >= 10000)
              AND ({reference_data_table_name}.transaction_type <> "C" or {reference_data_table_name}.transaction_type is null)
              AND {reference_data_table_name}.trade_datetime < "{up_until_datetime}"'''


def create_similar_trade_history_query(up_until_datetime, groups: list = []) -> str:
    '''The following view creates similar trade history for a bucket up to a point in time specified in `up_until_datetime`. The 
    value par_traded is assumed to be $5MM when the field par_traded is null and the is_trade_with_a_par_amount_over_5MM flag is true. 
    The exclusions are as follows:
    1) Trades with a par_traded under $10k, which we have found to be not useful for prediction.
    2) Trades with no dollar_price or yield.
    Note that these are only restrictions for trade data; we would still handle these CUSIPs if they are present in the reference data.
    `groups` is a list that has triples where each triple contains (1) issue key, (2) maturity bucket, and (3) coupon bucket. This is used to 
    add filtering to the SQL query by only selecting the similar trade history groups of interest.
    NOTE: `recent_similar` is created with the ORDER BY clause having both `trade_datetime` and `rtrs_control_number` in order to break ties appropriately 
    when `trade_datetime`s are equal. Without using the `rtrs_control_number` to break ties, the result of the query can change even though the 
    same query is run multiple times.'''
    group_based_where_clause = ''
    if len(groups) > 0: group_based_where_clause = ' AND (' + ' OR '.join([f'(b.issue_key = {issue_key} AND a.maturity_bucket = {maturity_bucket} AND a.coupon_bucket = {coupon_bucket})' for issue_key, maturity_bucket, coupon_bucket in groups]) + ')'
    return f'''SELECT b.issue_key, a.maturity_bucket, a.coupon_bucket, {create_trade_history_array_struct_in_bigquery("a")} AS recent_similar
                FROM `{AUXILIARY_VIEWS_DATASET}.trans_with_buckets` a LEFT JOIN (SELECT DISTINCT * FROM {AUXILIARY_VIEWS_DATASET}.calculation_date_and_price_v2) b
                {create_trade_history_query_join_condition("a", "b")}
                {create_trade_history_query_where_condition("a", "b", up_until_datetime)}{group_based_where_clause}
                GROUP BY b.issue_key, a.maturity_bucket, a.coupon_bucket'''


def create_trade_history_query(up_until_datetime, cusip_list: list = None) -> str:
    '''The following view creates trade history for a given CUSIP up to a point in time specified in `up_until_datetime`. For this to happen, 
    we need to do the following: look at all the trade messages and add a valid_to and valid_from timestamp to them in order to get the 
    most up-to-date trade for a given timestamp. This procedure is done in a [notebook](https://github.com/Ficc-ai/ficc/blob/dev/SQL_examples/Create_trade_history_with_reference_data.ipynb) 
    that creates a table called `msrb_final`. The table is `msrb_final` is always up-to-date, since it is a view that is created further 
    upstream to the `trade_history_same_issue_5_yr_mat_bucket_1_materialized` table. This view is joined to a table containing calculation dates for each trade. The 
    value par_traded is assumed to be $5MM when the field par_traded is null and the is_trade_with_a_par_amount_over_5MM flag is true. 
    The exclusions are as follows:
    1) Trades with a par_traded under $10k, which we have found to be not useful for prediction.
    2) Trades with no dollar_price or yield.
    Note that these are only restrictions for trade data; we would still handle these CUSIPs if they are present in the reference data. The optional 
    argument `cusip_list` can be a lists of CUSIPs which allow for the query to filter which CUSIPs to consider when making the query.
    NOTE: `recent` is created with the ORDER BY clause having both `trade_datetime` and `rtrs_control_number` in order to break ties appropriately 
    when `trade_datetime`s are equal. Without using the `rtrs_control_number` to break ties, the result of the query can change even though the 
    same query is run multiple times.'''
    return f'''SELECT a.cusip, {create_trade_history_array_struct_in_bigquery("a", "b")} AS recent
        FROM `{AUXILIARY_VIEWS_DATASET}.msrb_final` a LEFT JOIN (SELECT DISTINCT * from {AUXILIARY_VIEWS_DATASET}.calculation_date_and_price_v2) b
        {create_trade_history_query_join_condition("a", "b")}
        {create_trade_history_query_where_condition("a", "b", up_until_datetime)}{create_cusip_in_cusip_list_clause(cusip_list, table_identifier='a')}
        GROUP BY a.cusip'''


def get_reference_features(table_identifier: str = '') -> str:
    if table_identifier != '': table_identifier = table_identifier + '.'    # add the period so that the identifier can be concatenated directly into the reference feature in the same way as if it is an empty string
    reference_features_in_query = [f'{table_identifier}current_coupon_rate AS coupon',
                                   f'{table_identifier}issue_key as series_id',
                                   "CONCAT(IFNULL(organization_primary_name, ''), ' ', IFNULL(instrument_primary_name, ''), ' ', IFNULL(conduit_obligor_name, '')) AS security_description",
                                   f'{table_identifier}cusip',
                                   # 'ref_valid_from_date',
                                   # 'ref_valid_to_date',
                                   'incorporated_state_code',
                                   # 'organization_primary_name',
                                   # 'instrument_primary_name',
                                   'issue_key',
                                   'issue_text',
                                   # 'conduit_obligor_name',
                                   'is_called',
                                   'is_callable',
                                   # 'is_escrowed_or_pre_refunded',
                                   'first_call_date',
                                   # 'call_date_notice',
                                   'callable_at_cav',
                                   'par_price',
                                   'call_defeased',
                                   'call_timing',
                                   'call_timing_in_part',
                                   'extraordinary_make_whole_call',
                                   # 'extraordinary_redemption',
                                   'make_whole_call',
                                   'next_call_date',
                                   'next_call_price',
                                   # 'call_redemption_id',
                                   # 'first_optional_redemption_code',
                                   # 'second_optional_redemption_code',
                                   # 'third_optional_redemption_code',
                                   # 'first_mandatory_redemption_code',
                                   # 'second_mandatory_redemption_code',
                                   # 'third_mandatory_redemption_code',
                                   'par_call_date',
                                   'par_call_price',
                                   # 'maximum_call_notice_period',
                                   'called_redemption_type',
                                   # 'muni_issue_type',
                                   'refund_date',
                                   'refund_price',
                                   'redemption_cav_flag',
                                   # 'max_notification_days',
                                   # 'min_notification_days',
                                   # 'next_put_date',
                                   # 'put_end_date',
                                   # 'put_feature_price',
                                   # 'put_frequency',
                                   # 'put_start_date',
                                   # 'put_type',
                                   'maturity_date',
                                   'sp_long',
                                   # 'sp_stand_alone',
                                   # 'sp_icr_school',
                                   # 'sp_prelim_long',
                                   # 'sp_outlook_long',
                                   # 'sp_watch_long',
                                   # 'sp_Short_Rating',
                                   # 'sp_Credit_Watch_Short_Rating',
                                   # 'sp_Recovery_Long_Rating',
                                   'moodys_long',
                                   # 'moodys_short',
                                   # 'moodys_Issue_Long_Rating',
                                   # 'moodys_Issue_Short_Rating',
                                   # 'moodys_Credit_Watch_Long_Rating',
                                   # 'moodys_Credit_Watch_Short_Rating',
                                   # 'moodys_Enhanced_Long_Rating',
                                   # 'moodys_Enhanced_Short_Rating',
                                   # 'moodys_Credit_Watch_Long_Outlook_Rating',
                                   'has_sink_schedule',
                                   'next_sink_date',
                                   'sink_indicator',
                                   # 'sink_amount_type_text',
                                   # 'sink_amount_type_type',
                                   'sink_frequency',
                                   # 'sink_defeased',
                                   # 'additional_next_sink_date',
                                   'sink_amount_type',
                                   # 'additional_sink_frequency',
                                   'min_amount_outstanding',
                                   'max_amount_outstanding',
                                   'default_exists',
                                   'has_unexpired_lines_of_credit',
                                   # 'years_to_loc_expiration',
                                   'escrow_exists',
                                   # 'escrow_obligation_percent',
                                   # 'escrow_obligation_agent',
                                   # 'escrow_obligation_type',
                                   # 'child_linkage_exists',
                                   # 'put_exists',
                                   # 'floating_rate_exists',
                                   # 'bond_insurance_exists',
                                   'is_general_obligation',
                                   'has_zero_coupons',
                                   'delivery_date',
                                   'issue_price',
                                   # 'primary_market_settlement_date',
                                   # 'issue_date',
                                   'outstanding_indicator',
                                   # 'federal_tax_status',
                                   'maturity_amount',
                                   # 'available_denom',
                                   # 'denom_increment_amount',
                                   # 'min_denom_amount',
                                   'accrual_date',
                                   # 'bond_insurance',
                                   'coupon_type',
                                   # 'current_coupon_rate',
                                   'daycount_basis_type',
                                   # 'debt_type',
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
                                   # 'settlement_type',
                                   # 'additional_project_txt',
                                   # 'asset_claim_code',
                                   # 'additional_state_code',
                                   # 'backed_underlying_security_id',
                                   # 'bank_qualified',
                                   # 'capital_type',
                                   # 'conditional_call_date',
                                   # 'conditional_call_price',
                                   # 'designated_termination_date',
                                   # 'DTCC_status',
                                   # 'first_execution_date',
                                   # 'formal_award_date',
                                   'maturity_description_code',
                                   # 'muni_security_type',
                                   # 'mtg_insurance',
                                   # 'orig_cusip_status',
                                   # 'orig_instrument_enhancement_type',
                                   # 'other_enhancement_type',
                                   # 'other_enhancement_company',
                                   'pac_bond_indicator',
                                   # 'project_name',
                                   'purpose_class',
                                   # 'purpose_sub_class',
                                   # 'refunding_issue_key',
                                   # 'refunding_dated_date',
                                   # 'sale_date',
                                   # 'sec_regulation',
                                   # 'secured',
                                   'series_name',
                                   # 'sink_fund_redemption_method',
                                   'state_tax_status',
                                   # 'tax_credit_frequency',
                                   # 'tax_credit_percent',
                                   'use_of_proceeds',
                                   # 'use_of_proceeds_supplementary',
                                   # 'rating_downgrade',
                                   # 'rating_upgrade',
                                   # 'rating_downgrade_to_junk',
                                   # 'min_sp_rating_this_year',
                                   # 'max_sp_rating_this_year',
                                   # 'min_moodys_rating_this_year', 
                                   # 'max_moodys_rating_this_year',
                                  ]
    return ', '.join(reference_features_in_query)


def get_reference_data_query(up_until_datetime, cusip_list: list = None) -> str:
    '''The following query returns the reference data that is valid as of `up_until_datetime`.'''
    return f'''SELECT {get_reference_features()} FROM `reference_data_v1.reference_data_flat` WHERE cusip IS NOT NULL
               {create_cusip_in_cusip_list_clause(cusip_list)}
               AND ref_valid_from_date <= "{up_until_datetime}" AND "{up_until_datetime}" <= ref_valid_to_date'''


def join_trade_history_to_reference_data_query(up_until_datetime, cusip_list: list = None) -> str:
    '''The following query joins the reference data to the trade history that is valid as of 
    `up_until_datetime`.'''
    reference_data_table_alias = 'ref_data'
    return f'''SELECT {get_reference_features(reference_data_table_alias)},
                latest.* EXCEPT(cusip)
        FROM `reference_data_v1.reference_data_flat` {reference_data_table_alias} LEFT JOIN ({create_trade_history_query(up_until_datetime, cusip_list)}) latest
        ON latest.cusip = {reference_data_table_alias}.cusip   
        WHERE {reference_data_table_alias}.cusip IS NOT NULL
            {create_cusip_in_cusip_list_clause(cusip_list, table_identifier=reference_data_table_alias)}
              AND {reference_data_table_alias}.ref_valid_from_date <= "{up_until_datetime}" AND "{up_until_datetime}" <= {reference_data_table_alias}.ref_valid_to_date    -- need to have the WHERE clause statements in this order otherwise there is Google BigQuery error'''


def get_table_string(datetime_of_interest: datetime) -> str:
    return datetime_of_interest.strftime('%Y_%m_%d_%H_%M_%S')


def get_results_of_query_from_pickle_if_exists(query: str, file_path: str) -> pd.DataFrame:
    '''See if the results of the `query` are stored in `file_path`. If so, then return the dataframe. 
    Otherwise, call the query and save the results, i.e., (query, dataframe) in `file_path`.'''
    if VERBOSE: print('query:', query)
    using_pickle_filepath = False
    if os.path.exists(file_path):
        with open(file_path, 'rb') as pickle_file:
            query_from_pickle, df_from_pickle = pickle.load(pickle_file)
        if query == query_from_pickle:
            df = df_from_pickle
            print(f'Using saved dataframe from {file_path}')
            using_pickle_filepath = True
    if not using_pickle_filepath:
        df = sqltodf(query, bq_client)
        with open(file_path, 'wb') as pickle_file:
            pickle.dump((query, df), pickle_file)
        print(f'Saving (query, dataframe) to {file_path}')
    return df


def get_point_in_time_reference_data_from_deque(reference_data_deque: deque, datetime_of_interest: datetime = None) -> np.ndarray:
    '''Select the reference data snapshot that is current as of `datetime_of_interest`, i.e., has a `ref_valid_from_date` that 
    is before `datetime_of_interest` and `ref_valid_to_date` that is after `datetime_of_interest`. If `datetime_of_interest` is 
    `None`, then get the most recent snapshot of the reference data.'''
    most_recent_snapshot = reference_data_deque[0]    # index 0 indicates the most recent snapshot of the reference data
    if datetime_of_interest is None: return most_recent_snapshot
    
    ref_valid_from_date_idx = REFERENCE_DATA_FEATURE_TO_INDEX['ref_valid_from_date']
    valid_from_date_of_most_recent_snapshot = most_recent_snapshot[ref_valid_from_date_idx].tz_localize(None)    # remove timezone to avoid `TypeError: Cannot compare tz-naive and tz-aware timestamps`
    if datetime_of_interest >= valid_from_date_of_most_recent_snapshot: return most_recent_snapshot
    
    # after the most recent snapshot, each of the rest of the items in the dictionary is a dictionary containing the differences from the current snapshot to the immediately more recent snapshot
    reference_data_index = 1
    differences_dicts = [reference_data_deque[reference_data_index]]
    while datetime_of_interest < reference_data_deque[reference_data_index][ref_valid_from_date_idx].tz_localize(None):    # remove timezone to avoid `TypeError: Cannot compare tz-naive and tz-aware timestamps`
        reference_data_index += 1    # should not go out of bounds because if `datetime_of_interest` is before `DATE_FROM_WHICH_PAST_REFERENCE_DATA_IS_STORED_IN_REDIS` then we will use BigQuery to create the reference data
        differences_dicts.append(reference_data_deque[reference_data_index])    # add the differences dict of the next snapshot because entering the `while` loop means that the current snapshot is not the one that is before the `datetime_of_interest`
    cumulative_differences = {feature: value for differences in differences_dicts for feature, value in differences.items()}    # last occurrence of each key takes precedence which is desired since this is the snapshot which is the one right before the `datetime_of_interest`
    
    # override the features in the most recent snapshot
    for feature_idx, value in cumulative_differences.items():
        most_recent_snapshot[feature_idx] = value
    return most_recent_snapshot


@run_five_times_before_raising_redis_connector_error
def get_reference_data_from_redis_using_mget(cusip_list: list):
    '''Created this one line function solely to use the decorator: `run_five_times_before_raising_redis_connector_error`.'''
    return REFERENCE_DATA_REDIS_CLIENT.mget(cusip_list)


def get_point_in_time_reference_data_from_cusip_list(cusip_list: list, datetime_of_interest: datetime = None) -> list:
    reference_data_for_each_cusip = get_reference_data_from_redis_using_mget(cusip_list)
    return [get_point_in_time_reference_data_from_deque(pickle.loads(reference_data_pickle), datetime_of_interest) if reference_data_pickle is not None else None for reference_data_pickle in reference_data_for_each_cusip]


def get_trade_history_data_from_cusip_list(cusip_list: list, real_time: bool = False) -> list:
    '''`real_time` is a boolean flag that denotes whether the data is to be gotten as recently as possible. This 
    means that we can immediately perform truncation when getting the data and not have to do it further downstream. 
    I.e., truncation of the trade history to the datetime of interest will take place in `trade_dict_to_list(...)`'''
    trade_history_data_for_each_cusip = TRADE_HISTORY_REDIS_CLIENT.mget(cusip_list)
    trade_history_data_for_each_cusip = [pickle.loads(trade_history_data_pickle) if trade_history_data_pickle is not None else None for trade_history_data_pickle in trade_history_data_for_each_cusip]
    if real_time: [trade_history_data[:MAX_NUM_TRADES_IN_HISTORY_TO_DISPLAY_ON_UI, :] if trade_history_data is not None else None for trade_history_data in trade_history_data_for_each_cusip]
    return trade_history_data_for_each_cusip


def join_reference_data_with_trade_history(cusip_list: list, datetime_of_interest: datetime, multiprocessing: bool = MULTIPROCESSING):
    table_name_addendum = f'_{cusip_list[0]}' if multiprocessing else ''
    pickle_directory = '/tmp'    # f'{os.path.dirname(__file__)}/tmp'    # use `/tmp` as the pickle file directory since this is guaranteed to exist and will get cleared and is the default place to put files that are only temporary
    if datetime_of_interest < DATE_FROM_WHICH_ALL_PAST_TRADES_ARE_STORED_IN_REDIS:
        print(f'Since {datetime_of_interest} is before {DATE_FROM_WHICH_ALL_PAST_TRADES_ARE_STORED_IN_REDIS.date()}, we are using BigQuery to create the trade history and the reference data')
        trade_history_joined_to_reference_data_table_query = join_trade_history_to_reference_data_query(datetime_of_interest, cusip_list)
        pickle_filepath = f'{pickle_directory}/trade_history_latest_ref_data_minimal_exclusions_{get_table_string(datetime_of_interest)}{table_name_addendum}.pkl'
        trade_history_joined_to_reference_data_table_df = get_results_of_query_from_pickle_if_exists(trade_history_joined_to_reference_data_table_query, pickle_filepath)
    else:
        trade_history_data_for_each_cusip = get_trade_history_data_from_cusip_list(cusip_list)
        trade_history_data_for_each_cusip = pd.DataFrame(list(zip(cusip_list, trade_history_data_for_each_cusip)), columns=['cusip', 'recent'])    # create dataframe with two columns where the first column is the CUSIP and the second column is the trade history; using a dataframe to make merging easier
        if datetime_of_interest < DATE_FROM_WHICH_PAST_REFERENCE_DATA_IS_STORED_IN_REDIS:
            print(f'Since {datetime_of_interest} is after {DATE_FROM_WHICH_ALL_PAST_TRADES_ARE_STORED_IN_REDIS.date()}, but before {DATE_FROM_WHICH_PAST_REFERENCE_DATA_IS_STORED_IN_REDIS.date()} we are using the redis to create the trade history and BigQuery to create the reference data')
            pickle_filepath = f'{pickle_directory}/ref_data_minimal_exclusions_{get_table_string(datetime_of_interest)}{table_name_addendum}.pkl'
            reference_data_query = get_reference_data_query(datetime_of_interest, cusip_list)
            reference_data_df = get_results_of_query_from_pickle_if_exists(reference_data_query, pickle_filepath)
        else:
            print(f'Since {datetime_of_interest} is after {DATE_FROM_WHICH_PAST_REFERENCE_DATA_IS_STORED_IN_REDIS.date()} we are using the redis to create the trade history and the reference data')
            reference_data_for_each_cusip = get_point_in_time_reference_data_from_cusip_list(cusip_list, datetime_of_interest)
            reference_data_df = pd.DataFrame(np.vstack(reference_data_for_each_cusip), columns=REFERENCE_DATA_FEATURES) if reference_data_for_each_cusip != [] else pd.DataFrame()
        trade_history_joined_to_reference_data_table_df = reference_data_df.merge(trade_history_data_for_each_cusip, on='cusip', how='left')

    cusip_set = set(cusip_list)
    trade_history_joined_to_reference_data_table_df_cusip_set = set(trade_history_joined_to_reference_data_table_df['cusip'])
    in_cusip_list_but_not_in_trade_history_joined_to_reference_data_table_df = cusip_set - trade_history_joined_to_reference_data_table_df_cusip_set
    in_trade_history_joined_to_reference_data_table_df_but_not_in_cusip_list = trade_history_joined_to_reference_data_table_df_cusip_set - cusip_set
    if len(in_cusip_list_but_not_in_trade_history_joined_to_reference_data_table_df) > 0 or len(in_trade_history_joined_to_reference_data_table_df_but_not_in_cusip_list) > 0: print(f'CUSIPs in `cusip_list` not present in `trade_history_joined_to_reference_data_table_df`: {in_cusip_list_but_not_in_trade_history_joined_to_reference_data_table_df}.\nCUSIPs in `trade_history_joined_to_reference_data_table_df` not present in `cusip_list`: {in_trade_history_joined_to_reference_data_table_df_but_not_in_cusip_list}')
    return trade_history_joined_to_reference_data_table_df


def join_reference_data_with_similar_trade_history(cusip_list: list, reference_data_df: pd.DataFrame, datetime_of_interest: datetime, multiprocessing: bool = MULTIPROCESSING):
    many_cusips = len(cusip_list) > MAX_NUMBER_OF_CUSIPS_TO_USE_GROUPS_FOR_SIMILAR_TRADE_HISTORY_QUERY_OPTIMIZATION    # used for further downstream optimization in deciding whether to pass `groups` into `create_similar_trade_history_query(...)`

    missing_values_for_similar_trade_history_group = reference_data_df[FEATURES_NECESSARY_FOR_CREATING_SIMILAR_TRADE_HISTORY_GROUPS].isnull().any(axis=1)    # boolean pd.Series that indicates whether each row contains at least one null value in `FEATURES_NECESSARY_FOR_CREATING_SIMILAR_TRADE_HISTORY_GROUPS`
    reference_data_df_missing_values_for_similar_trade_history_group = reference_data_df[missing_values_for_similar_trade_history_group]
    reference_data_df_missing_values_for_similar_trade_history_group['recent_similar'] = np.nan
    reference_data_df = reference_data_df[~missing_values_for_similar_trade_history_group]

    if datetime_of_interest < DATE_FROM_WHICH_ALL_PAST_TRADES_ARE_STORED_IN_REDIS:
        print(f'Since {datetime_of_interest} is before {DATE_FROM_WHICH_ALL_PAST_TRADES_ARE_STORED_IN_REDIS.date()}, we are using BigQuery to create the similar trade history')
        similar_trades_column_prefix = 'similar_trade_history_'
        original_similar_trades_columns = ['issue_key', 'maturity_bucket', 'coupon_bucket']
        similar_trades_columns = [f'{similar_trades_column_prefix}{original_column}' for original_column in original_similar_trades_columns]

        temp = reference_data_df.apply(lambda row: get_similar_trade_history_group(row, datetime_of_interest.date()), axis=1)
        reference_data_df[similar_trades_columns] = pd.DataFrame(temp.tolist(), index=reference_data_df.index)    # followed the same format as `process_trade_history.py::get_recent_similar_trades_and_last_similar_trade_features(...)`
        groups = [] if many_cusips else list(zip(*[reference_data_df[column] for column in similar_trades_columns]))    # gets the values from each column in `similar_trades_columns` and converts it into a list of tuples (the length of the tuple is the number of columns in `similar_trades_columns`)

        similar_trade_history_df = get_similar_trade_history_df_from_bigquery(datetime_of_interest, groups)
        similar_trade_history_df = similar_trade_history_df.rename(columns=dict(zip(original_similar_trades_columns, similar_trades_columns)))
        
        similar_trade_history_joined_to_reference_data_table_df = reference_data_df.merge(similar_trade_history_df, on=similar_trades_columns, how='left')
        similar_trade_history_joined_to_reference_data_table_df = similar_trade_history_joined_to_reference_data_table_df.drop(columns=similar_trades_columns)    # remove temporary columns used for getting similar trades
    else:
        print(f'Since {datetime_of_interest} is after {DATE_FROM_WHICH_ALL_PAST_TRADES_ARE_STORED_IN_REDIS.date()}, we are using the redis to create the similar trade history')
        similar_trade_history_data_for_each_cusip = [get_similar_trade_history_data(reference_data_df[reference_data_df['cusip'] == cusip].head(1).squeeze(), cusip, datetime_of_interest.date()) for cusip in cusip_list]    # keep only one row with `.head(1)` for each CUSIP since we are using this to extract the reference data for each CUSIP and convert this one item dataframe to a series with `.squeeze()`
        similar_trade_history_data_for_each_cusip = pd.DataFrame(list(zip(cusip_list, similar_trade_history_data_for_each_cusip)), columns=['cusip', 'recent_similar'])
        similar_trade_history_joined_to_reference_data_table_df = reference_data_df.merge(similar_trade_history_data_for_each_cusip, on='cusip', how='left')
    
    similar_trade_history_joined_to_reference_data_table_df = pd.concat([similar_trade_history_joined_to_reference_data_table_df, reference_data_df_missing_values_for_similar_trade_history_group]).sort_index()    # `.sort_index()` preserves the original row order
    cusip_set = set(cusip_list)
    similar_trade_history_joined_to_reference_data_table_df_cusip_set = set(similar_trade_history_joined_to_reference_data_table_df['cusip'])
    in_cusip_list_but_not_in_similar_trade_history_joined_to_reference_data_table_df = cusip_set - similar_trade_history_joined_to_reference_data_table_df_cusip_set
    in_similar_trade_history_joined_to_reference_data_table_df_but_not_in_cusip_list = similar_trade_history_joined_to_reference_data_table_df_cusip_set - cusip_set
    if len(in_cusip_list_but_not_in_similar_trade_history_joined_to_reference_data_table_df) > 0 or len(in_similar_trade_history_joined_to_reference_data_table_df_but_not_in_cusip_list) > 0: print(f'CUSIPs in `cusip_list` not present in `similar_trade_history_joined_to_reference_data_table_df`: {in_cusip_list_but_not_in_similar_trade_history_joined_to_reference_data_table_df}.\nCUSIPs in `similar_trade_history_joined_to_reference_data_table_df` not present in `cusip_list`: {in_similar_trade_history_joined_to_reference_data_table_df_but_not_in_cusip_list}')
    return similar_trade_history_joined_to_reference_data_table_df


@function_timer
def trade_history_and_similar_trade_history_joined_to_reference_data_table(df_with_cusip_column: pd.DataFrame, datetime_of_interest: datetime, multiprocessing: bool = MULTIPROCESSING) -> pd.DataFrame:
    cusip_list = df_with_cusip_column['cusip'].unique().tolist()    # used to filter the query to consider only CUSIPs that we will price later; prefer to do it in the query so we are working with a lot less data and have lower memory usage
    trade_history_joined_to_reference_data_table_df = join_reference_data_with_trade_history(cusip_list, datetime_of_interest, multiprocessing)
    if USE_SIMILAR_TRADES_MODEL_FOR_YIELD_SPREAD_PREDICTIONS_FOR_POINT_IN_TIME_PRICING: trade_history_joined_to_reference_data_table_df = join_reference_data_with_similar_trade_history(cusip_list, trade_history_joined_to_reference_data_table_df, datetime_of_interest, multiprocessing)
    return trade_history_joined_to_reference_data_table_df


def create_reference_data_and_trade_history(to_be_priced_df: pd.DataFrame, datetime_of_interest, use_multiprocessing: bool = MULTIPROCESSING, save_data: bool = False, reset_index: bool = True) -> pd.DataFrame:
    num_cpus = os.cpu_count()
    num_line_items = len(to_be_priced_df)
    use_multiprocessing = use_multiprocessing and (num_line_items > num_cpus)

    to_be_priced_df_pickle_file = 'to_be_priced_df.pkl'
    trade_history_and_similar_trade_history_joined_to_reference_data_table_pickle_file = 'trade_history_and_similar_trade_history_joined_to_reference_data_table_df.pkl'
    if save_data and os.path.isfile(trade_history_and_similar_trade_history_joined_to_reference_data_table_pickle_file):
        print(f'Found `{trade_history_and_similar_trade_history_joined_to_reference_data_table_pickle_file}`, so attempting to use it instead of calling `trade_history_and_similar_trade_history_joined_to_reference_data_table(...)`')
        if os.path.isfile(trade_history_and_similar_trade_history_joined_to_reference_data_table_pickle_file) and to_be_priced_df.equals(pd.read_pickle(to_be_priced_df_pickle_file)): 
            trade_history_and_similar_trade_history_joined_to_reference_data_table_df = pd.read_pickle(trade_history_and_similar_trade_history_joined_to_reference_data_table_pickle_file)
            return trade_history_and_similar_trade_history_joined_to_reference_data_table_df
        else:
            print(f'Cannot use `{trade_history_and_similar_trade_history_joined_to_reference_data_table_pickle_file}` since the `to_be_priced_df` is different from the last time this file was created; calling `trade_history_and_similar_trade_history_joined_to_reference_data_table(...)`')

    if (USE_SIMILAR_TRADES_MODEL_FOR_YIELD_SPREAD_PREDICTIONS_FOR_POINT_IN_TIME_PRICING 
        and num_line_items > MAX_NUMBER_OF_CUSIPS_TO_USE_GROUPS_FOR_SIMILAR_TRADE_HISTORY_QUERY_OPTIMIZATION 
        and datetime_of_interest < DATE_FROM_WHICH_ALL_PAST_TRADES_ARE_STORED_IN_REDIS):
        get_similar_trade_history_df_from_bigquery(datetime_of_interest)    # running this function simply to populate the dataframe in Google Cloud storage if it does not already exist; TODO: need to maintain the GCP bucket that stores all of these dataframes
    if use_multiprocessing and num_line_items > MAX_NUM_OF_CUSIPS_FOR_BIGQUERY:
        create_trade_history_and_similar_trade_history_joined_to_reference_data_table_df_caller = lambda df: trade_history_and_similar_trade_history_joined_to_reference_data_table(df, datetime_of_interest, True)
        assert num_line_items <= MAX_NUM_OF_CUSIPS_FOR_BIGQUERY * num_cpus, f'Number of CUSIPs is too large: {num_line_items}; must be less than {MAX_NUM_OF_CUSIPS_FOR_BIGQUERY * num_cpus}'
        print(f'Using multiprocessing inside price_cusips_point_in_time(...) with num_cpus={num_cpus}')
        with mp.Pool() as pool_object:
            trade_history_and_similar_trade_history_joined_to_reference_data_table_df = pool_object.map(create_trade_history_and_similar_trade_history_joined_to_reference_data_table_df_caller, create_df_chunks(to_be_priced_df, num_cpus))
    else:
        print('Not using multiprocessing to create reference data and trade history table')
        create_trade_history_and_similar_trade_history_joined_to_reference_data_table_df_caller = lambda df: trade_history_and_similar_trade_history_joined_to_reference_data_table(df, datetime_of_interest, False)
        to_be_priced_df_chunks = create_df_chunks(to_be_priced_df, int(np.ceil(num_line_items / MAX_NUM_OF_CUSIPS_FOR_BIGQUERY)))    # always have at most 1 chunk
        trade_history_and_similar_trade_history_joined_to_reference_data_table_df = [create_trade_history_and_similar_trade_history_joined_to_reference_data_table_df_caller(chunk) for chunk in tqdm(to_be_priced_df_chunks, disable=len(to_be_priced_df_chunks) == 1)]    # disable tqdm progress bar if the number of chunks is 1
    trade_history_and_similar_trade_history_joined_to_reference_data_table_df = pd.concat(trade_history_and_similar_trade_history_joined_to_reference_data_table_df)
    if reset_index: trade_history_and_similar_trade_history_joined_to_reference_data_table_df = trade_history_and_similar_trade_history_joined_to_reference_data_table_df.reset_index(drop=True)
    
    if save_data:
        to_be_priced_df.to_pickle(to_be_priced_df_pickle_file)
        trade_history_and_similar_trade_history_joined_to_reference_data_table_df.to_pickle(trade_history_and_similar_trade_history_joined_to_reference_data_table_pickle_file)
    return trade_history_and_similar_trade_history_joined_to_reference_data_table_df


@function_timer
def get_similar_trade_history_df_from_bigquery(datetime_of_interest: datetime, groups: list = []) -> pd.DataFrame:
    similar_trade_history_gcp_bucket_name = 'similar_trade_history_point_in_time'
    similar_trade_history_query = create_similar_trade_history_query(datetime_of_interest, groups)
    similar_trade_history_point_in_time_file_name = datetime_of_interest.strftime(YEAR_MONTH_DAY + '_' + HOUR_MIN_SEC) + '.pkl'
    similar_trade_history_point_in_time_file_name = similar_trade_history_point_in_time_file_name.replace(':', '-')
    similar_trade_history_point_in_time_file = download_pickle_file(storage_client, similar_trade_history_gcp_bucket_name, similar_trade_history_point_in_time_file_name)
    if VERBOSE: print('similar trade history query:', similar_trade_history_query)
    similar_trade_history_df = None
    if similar_trade_history_point_in_time_file is not None:    # file exists
        query, df = similar_trade_history_point_in_time_file
        if query == similar_trade_history_query:
            similar_trade_history_df = df
    if similar_trade_history_df is None:    # either pickle file does not exist in Google Cloud Storage or the query did not match
        similar_trade_history_df = sqltodf(similar_trade_history_query, bq_client)
        similar_trade_history_df = similar_trade_history_df.dropna()    # necessary to drop rows with null values since some of the rows have a null value for one of the features necessary to create the similar trade history group
        if len(groups) == 0:    # do not store the dataframe if there are groups in the similar trades query since this does not generalize and is not useful for cusips that do not belong to this group
            with open(similar_trade_history_point_in_time_file_name, 'wb') as pickle_file:    # create a temporary file to store the similar trade history dataframe to upload to GCP storage
                pickle.dump((similar_trade_history_query, similar_trade_history_df), pickle_file)
            upload_data(storage_client, similar_trade_history_gcp_bucket_name, similar_trade_history_point_in_time_file_name)
            os.remove(similar_trade_history_point_in_time_file_name)    # delete the temporary file
    return similar_trade_history_df
