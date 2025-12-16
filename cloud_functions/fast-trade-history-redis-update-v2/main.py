import functions_framework

import os
from datetime import datetime
import pickle

import numpy as np
import pandas as pd

from google.cloud import bigquery

from auxiliary_variables import FILENAME_ADDENDUM, \
                                MSRB_INTRADAY_FILES_BUCKET_NAME, \
                                DATETIME_FAR_INTO_FUTURE, \
                                ALL_TRADE_MESSAGES_FILENAME, \
                                EASTERN, \
                                YEAR_MONTH_DAY, \
                                HOUR_MIN_SEC, \
                                FEATURES_FOR_EACH_TRADE_IN_HISTORY, \
                                LOGGING_PRECISION, \
                                DATA_TYPE_DICT, \
                                REFERENCE_DATA_FEATURES, \
                                NUM_OF_DAYS_IN_YEAR, \
                                MAX_TRADES_USED_IN_MODEL, \
                                MAX_NUM_DAYS_FOR_TRADES_IN_HISTORY, \
                                MAX_NUM_TRADES_IN_HISTORY, \
                                MAX_NUM_TRADES_IN_SIMILAR_TRADE_HISTORY, \
                                reference_data_redis_client, \
                                trade_history_redis_client, \
                                similar_trade_history_redis_client, \
                                future_processing_file_bucket_name, \
                                future_processing_file_name
from auxiliary_functions import function_timer, run_five_times_before_raising_redis_connector_error, download_pickle_file, delete_file, upload_to_storage, df_to_json_dict, diff_in_days_two_dates, upload_to_bigquery, upload_to_redis_using_mset
from price import compute_price
from bond_characteristics import get_frequency


pd.options.mode.chained_assignment = None    # default='warn'; suppresses `SettingWithCopyWarning` in Pandas

# # TODO: comment the below line out when not running the function locally
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/user/ficc/mitas_creds.json'


@function_timer
def typecast_reference_data(df):
    '''This function takes the dataframe from the bigquery and updates certain 
    fields to be the right type. Note that this function mutates the fields 
    passed in dataframe, so the function itself has no return value.'''
    df['coupon'] = df['coupon'].astype(float)
    df['next_call_price'] = df['next_call_price'].astype(float)
    return df


def typecast_yield(df):
    df['yield'] = df['yield'].astype(float)
    # df['deferred'] = (df.interest_payment_frequency == 0) | df.coupon == 0
    return df


def get_latest_sequence_number_filename_with_folder(timestamp):
    timestamp_wo_hour_min_sec = timestamp[:10]
    latest_sequence_number_filename = 'latest_sequence_number' + FILENAME_ADDENDUM
    return f'{timestamp_wo_hour_min_sec}/{latest_sequence_number_filename}.pkl'


@function_timer
def get_latest_sequence_number(timestamp):
    '''Get the latest sequence number from a pickle file that contains it. Each day has a new file.'''
    latest_sequence_number_filename_with_folder = get_latest_sequence_number_filename_with_folder(timestamp)
    latest_sequence_number = download_pickle_file(MSRB_INTRADAY_FILES_BUCKET_NAME, latest_sequence_number_filename_with_folder)
    latest_sequence_number = -1 if latest_sequence_number is None else latest_sequence_number
    print(f'Latest sequence number: {latest_sequence_number}')
    return latest_sequence_number


@function_timer
def update_latest_sequence_number(timestamp, sequence_number):
    '''Update the latest sequence number in the pickle file that contains it. Each day has a new file.'''
    latest_sequence_number_filename_with_folder = get_latest_sequence_number_filename_with_folder(timestamp)
    upload_to_storage(latest_sequence_number_filename_with_folder, pickle.dumps(sequence_number))
    print(f'Updated the latest sequence number to {sequence_number} in {latest_sequence_number_filename_with_folder} in Google Cloud Storage bucket: {MSRB_INTRADAY_FILES_BUCKET_NAME}')
    return sequence_number


def update_latest_sequence_number_and_return_success(timestamp, sequence_number):
    '''Used to terminate the `hello_http(...)` function.'''
    update_latest_sequence_number(timestamp, sequence_number)
    return 'SUCCESS'


def concatenate_date_and_time_objects_into_datetime_object(df):
    '''Concatenate a date object with a time object to create a datetime object. We assume that the date 
    field is a date object (converted from string to date in `convert_date_object_to_date_type`) and the 
    time field is a string.
    NOTE: `msrb_valid_to_date` is assigned to `publish_datetime` and `msrb_valid_from_date` is a dummy field 
    assigned to a far away datetime. Even for point in time, the most recent data should be used for `calc_date`.'''
    date_and_time_objects = {('trade_date', 'time_of_trade'): 'trade_datetime', 
                             ('publish_date', 'publish_time'): 'publish_datetime'}
    for (date_field, time_field), datetime_field in date_and_time_objects.items():
        df[datetime_field] = df.apply(lambda row: datetime.combine(row[date_field], datetime.strptime(row[time_field], HOUR_MIN_SEC).time()), axis=1)
    df['msrb_valid_from_date'] = df['publish_datetime']
    df['msrb_valid_to_date'] = DATETIME_FAR_INTO_FUTURE
    return df


@function_timer
def upload_calculation_date_and_price_to_bigquery(df: pd.DataFrame, current_timestamp):
    df = df.rename(columns={'calc_day_cat': 'calc_date_selection',    # `auxiliary_views_v2.calculation_date_and_price_v2` BigQuery table has the name `calc_date_selection` for `calc_day_cat`
                            'conduit_obligor_name': 'obligor_id'})
    column_to_dtype = {'rtrs_control_number': 'INTEGER', 
                       'trade_datetime': 'DATETIME', 
                       'cusip': 'STRING', 
                       'calc_price': 'FLOAT', 
                       'price_to_next_call': 'FLOAT', 
                       'price_to_par_call': 'FLOAT', 
                       'price_to_maturity': 'FLOAT', 
                       'calc_date': 'DATE', 
                       'next_call_date': 'DATE', 
                       'par_call_date': 'DATE', 
                       'maturity_date': 'DATE', 
                       'refund_date': 'DATE', 
                       'price_delta': 'FLOAT', 
                       'publish_datetime': 'DATETIME', 
                       'when_issued': 'BOOLEAN', 
                       'calc_date_selection': 'INTEGER', 
                       'issue_key': 'STRING', 
                       'sequence_number': 'INTEGER', 
                       'par_traded': 'INTEGER', 
                       'series_name': 'STRING', 
                       'series_id': 'STRING', 
                       'msrb_valid_to_date': 'DATETIME', 
                       'msrb_valid_from_date': 'DATETIME', 
                       'obligor_id': 'STRING', 
                       'upload_datetime': 'DATETIME'}    # 'brokers_broker': 'STRING', 'assumed_settlement_date': 'DATE', 'unable_to_verify_dollar_price': 'BOOLEAN'; these columns were not in auxiliary_views_v2.calculation_date_and_price_v2
    df['upload_datetime'] = current_timestamp    # when there are duplicate RTRS control numbers with the same `publish_datetime` and `sequence_number` (but some miscellaneous feature is different such as `obligor_id` due to weirdness with reference data files) we use the row that corresponds to the most recent `upload_datetime`
    df = df[list(column_to_dtype.keys())]    # keep only the columns in `columns_to_dtype`
    schema = [bigquery.SchemaField(column, dtype) for column, dtype in column_to_dtype.items()]
    upload_to_bigquery('auxiliary_views_v2.calculation_date_and_price_v2', schema, df, 'upload_calculation_date_and_price_to_bigquery')    # TODO: remove final argument for procedure to be asynchronous


@function_timer
def get_all_trade_messages(timestamp):
    '''Each day has its own dataframe pickle file with all trades, so we retrieve it for the 
    day from the specified timestamp.'''
    timestamp_wo_hour_min_sec = timestamp[:10]    # assumes that the first 10 characters of `timestamp` contain the year, month, and day (e.g., 2024-01-01 is 10 characters)
    filename_with_directory = f'{timestamp_wo_hour_min_sec}/{ALL_TRADE_MESSAGES_FILENAME}.pkl'
    return download_pickle_file(MSRB_INTRADAY_FILES_BUCKET_NAME, filename_with_directory), filename_with_directory


@function_timer
def get_msrb_trade_messages(beginning_sequence_number, timestamp):
    '''MSRB API is no longer called in this function. See the cloud function `get_msrb_trade_messages` for more details.'''
    trade_messages, filename = get_all_trade_messages(timestamp)
    if trade_messages is None:
        print(f'{filename} not found in Google Cloud Storage bucket: {MSRB_INTRADAY_FILES_BUCKET_NAME}, meaning that are no trade messages yet for today. If this is in error, refer to cloud function `get_msrb_trade_messages`.')
        return pd.DataFrame(), beginning_sequence_number - 1
    trade_messages_after_beginning_sequence_number = trade_messages[trade_messages.index >= beginning_sequence_number]
    if len(trade_messages_after_beginning_sequence_number) == 0:
        print(f'No new trade messages since sequence number: {beginning_sequence_number}')
        return pd.DataFrame(), beginning_sequence_number - 1
    print(f'{len(trade_messages_after_beginning_sequence_number)} trade messages since sequence number: {beginning_sequence_number}')
    trade_messages_after_beginning_sequence_number = trade_messages_after_beginning_sequence_number.reset_index(drop=True)    # `.reset_index(drop=True)` ensures that `sequence_number` is not the index, as it is when getting trades from `get_all_trade_messages(...)`, since this causes problems downstream since the index is called `sequence_number` and there is a column `sequence_number` (e.g., `.sort_values(by='sequence_number')` throws an error because of ambiguity)
    latest_sequence_number = trade_messages_after_beginning_sequence_number['sequence_number'].max()
    print('Latest sequence number:', latest_sequence_number)
    return trade_messages_after_beginning_sequence_number, latest_sequence_number


@function_timer
def remove_open_and_close_messages(trade_messages):
    return trade_messages if len(trade_messages) == 0 else trade_messages.dropna(subset=['transaction_type'])    # drop the opening and closing messages for the day


@function_timer
def process_msrb_data(msrb_data):
    '''Make the following modifications (that were previously done in SQL) to `par_traded` and `settlement_date`:
    CASE
        WHEN a.par_traded IS NULL AND is_trade_with_a_par_amount_over_5MM IS TRUE THEN 5000000
    ELSE
        a.par_traded
    END AS par_traded, 
    CASE
        WHEN a.settlement_date IS NULL AND a.assumed_settlement_date IS NOT NULL THEN a.assumed_settlement_date
    ELSE
        a.settlement_date
    END AS settlement_date

    Additionally, check for multiple trade messages with the same RTRS control number. If those exist, order by publish_datetime, sequence_number 
    descending and take the most recent (the reason why we have to use sequence number is because sometimes publish datetime is not unique). It is 
    possibly the case that you only need sequence number because `msrb_data` only contains rows that are in the same day.'''
    if len(msrb_data) == 0: return msrb_data
    msrb_data = msrb_data.sort_values(by='sequence_number', ascending=False)
    par_traded_is_null_and_par_amount_over_5MM = msrb_data['par_traded'].isnull() & msrb_data['is_trade_with_a_par_amount_over_5MM']
    msrb_data.loc[par_traded_is_null_and_par_amount_over_5MM, 'par_traded'] = 5000000
    msrb_data = msrb_data[msrb_data['par_traded'] >= 10000]    # remove all trades with `par_traded` less than 10000
    msrb_data['settlement_date'].fillna(msrb_data['assumed_settlement_date'], inplace=True)    # pandas replace NaN in one column with value from corresponding row of second column: https://stackoverflow.com/questions/29177498/python-pandas-replace-nan-in-one-column-with-value-from-corresponding-row-of-sec
    return msrb_data


@function_timer
def get_trades_for_future_processing():
    '''Only keep trades that are 7 days old or newer. From a team member: "If you keep trades for a month, you get 99% of the 
    trades that eventually have reference data. If you keep trades for 2 days you get 70%. If you keep trades for 7 
    days you get 97.5%. In almost all these cases where reference data is not available within a day, the issue is 
    the same as when the reference data never shows up: these are short maturity securities (like coml paper) that 
    have different reporting requirements."'''
    trades = download_pickle_file(future_processing_file_bucket_name, future_processing_file_name)
    if trades is None or len(trades) == 0: return pd.DataFrame()
    trades = pd.DataFrame(trades, columns=DATA_TYPE_DICT.keys())
    at_most_7_days_old = trades['upload_date'].apply(lambda date_string: diff_in_days_two_dates(datetime.now(), pd.to_datetime(date_string), convention='exact')) <= 7    # mark trades if it has been at most 7 days since we first saw it
    return trades[at_most_7_days_old]    # TODO: make this faster by filtering only once a day


@function_timer
def store_trades_for_future_processing_and_return_trades_with_cusips_found_in_reference_data(trades, cusips_not_found):
    '''Upload `trades` where the CUSIP for each trade is in `cusips_not_found` to the bucket if it is not empty. 
    NOTE: If you upload a file with the same name as an existing object in your Cloud Storage bucket, the existing 
    object is overwritten (https://cloud.google.com/storage/docs/uploads-downloads). This is the desired behavior.'''
    if len(cusips_not_found) == 0:
        delete_file(future_processing_file_bucket_name, future_processing_file_name)
        return trades

    print(f'CUSIPs not found in reference data redis: {cusips_not_found}')
    cusips_not_found_condition = trades['cusip'].isin(cusips_not_found)
    trades_with_cusips_not_found = trades[cusips_not_found_condition]
    print(f'{len(trades_with_cusips_not_found)} trades have CUSIPs that were not found in reference data redis')
    upload_to_storage(future_processing_file_name, pickle.dumps(trades_with_cusips_not_found), bucket_name=future_processing_file_bucket_name)
    return trades[~cusips_not_found_condition]


@run_five_times_before_raising_redis_connector_error
def get_pickled_cusips_data(cusips: list) -> list:
    return reference_data_redis_client.mget(cusips)


@function_timer
def get_reference_data_from_reference_data_redis(cusips: list):
    '''Get reference data for each CUSIP in `cusips` from reference_data_redis. Return the result as a dataframe.'''
    pickled_cusips_data = get_pickled_cusips_data(cusips)
    reference_data = []
    cusips_not_found = []
    for cusip_idx, pickled_cusip_data in enumerate(pickled_cusips_data):    # takes less than 0.5 seconds, so no need to parallelize this with `multiprocess`
        if pickled_cusip_data is None:
            cusip = cusips[cusip_idx]
            cusips_not_found.append(cusip)
        else:
            reference_data.append(pd.Series(pickle.loads(pickled_cusip_data)[0], index=REFERENCE_DATA_FEATURES))    # index 0 indicates the most recent snapshot of the reference data
    return (pd.concat(reference_data, axis=1).T, cusips_not_found) if reference_data != [] else (pd.DataFrame(), cusips_not_found)    # list of series to dataframe: https://stackoverflow.com/questions/55478191/list-of-series-to-dataframe


@function_timer
def left_join_on_cusip(table1, table2):
    '''Perform a left join of `table1` to `table2` on `cusip`. Assume that `table1` is MSRB data and 
    `table2` is reference data, so that we keep the features from the reference data with handling of 
    `suffixes` when calling `merge(...)`.'''
    duplicate_column_suffix = '_remove_after_merge_to_keep_the_column_from_reference_data_provider'
    merged = table1.merge(table2, on='cusip', how='left', suffixes=(duplicate_column_suffix, None))    # `suffixes` argument specifies which suffix to put at the end of the column name for each table if the column name is the same; if a feature is in MSRB data and reference data, then we want to keep the one from the reference data
    return merged.drop(columns=[column for column in merged.columns if column.endswith(duplicate_column_suffix)])    # keep the column name from the first table


@function_timer
def add_restrictions_on_interest_payment_frequency(df):
    '''Make sure that `df` has `interest_payment_frequency` values that are one of 1, 2, 3, 5, 16.'''
    return df[df['interest_payment_frequency'].isin([1, 2, 3, 5, 16])]


@function_timer
def add_calc_date(df):
    '''Add the calc date field for each row to the dataframe `df`.'''
    df['calc_price'], df['calc_date'], df['price_to_next_call'], df['price_to_par_call'], df['price_to_maturity'], df['calc_day_cat'] = zip(*df.apply(lambda row: compute_price(row), axis=1))    # no need to parallelize this since although it is the bottleneck in this procedure, the entire procedure takes less than 0.5 seconds
    df['price_delta'] = np.abs(df.calc_price - df.dollar_price)
    df['price_delta'] = df['price_delta'].astype(float)    # manually convert to type `float` to avoid the following error in the next line: `TypeError: loop of ufunc does not support argument 0 of type float which has no callable rint method`
    df['price_delta'] = np.round(df['price_delta'], LOGGING_PRECISION)    # need to be rounded to LOGGING_PRECISION (can perhaps do more but not too many more digits) decimal places otherwise will be detected as an invalid numerical value
    return df


@function_timer
def upload_trade_history_to_trade_history_bigquery(cusip_trade_history_pairs):
    '''Upload the trade history to a bigquery table for easy monitoring.'''
    def create_trade_history_record(trade_history_array):
        '''`trade_history_array` is a numpy array for a single trade.'''
        columns = list(FEATURES_FOR_EACH_TRADE_IN_HISTORY.keys())
        trade_history_df = pd.DataFrame(trade_history_array, columns=columns)
        return df_to_json_dict(trade_history_df)
    
    schema = [bigquery.SchemaField('cusip', 'STRING'), 
              bigquery.SchemaField('upload_datetime', 'DATETIME'), 
              bigquery.SchemaField('recent', 'RECORD', mode='REPEATED', fields=[bigquery.SchemaField(feature, schema_type) for feature, schema_type in FEATURES_FOR_EACH_TRADE_IN_HISTORY.items()])]    # `recent` corresponds to trade history
    now = datetime.now(EASTERN).strftime(YEAR_MONTH_DAY + 'T' + HOUR_MIN_SEC)
    cusip_trade_history_dict = [{'cusip': cusip, 'upload_datetime': now, 'recent': create_trade_history_record(trade_history)} for cusip, trade_history in cusip_trade_history_pairs]    # no need to parallelize this since this procedure is not a bottleneck
    upload_to_bigquery('demo_monitoring.trade_history_only' + FILENAME_ADDENDUM, schema, cusip_trade_history_dict)    #, 'upload_trade_history_to_trade_history_bigquery')    # removed the final argument for procedure to be asynchronous


@run_five_times_before_raising_redis_connector_error
def key_exists_in_redis(redis_client, key):
    '''Created this one line function solely to use the decorator: `run_five_times_before_raising_redis_connector_error`.'''
    return redis_client.exists(key)


@run_five_times_before_raising_redis_connector_error
def get_value_for_key_in_redis(redis_client, key):
    '''Created this one line function solely to use the decorator: `run_five_times_before_raising_redis_connector_error`.'''
    return redis_client.get(key)


def create_trade_history_numpy_array(trade_history_df, max_num_trades):
    trade_history_df = trade_history_df.drop_duplicates(subset='rtrs_control_number', keep='first')    # keep the most recently published `rtrs_control_number` which we can assume is in descending order of 'publish_datetime' and 'sequence_number' due to the `.sort_values(...)` statement above
    trade_history_df = trade_history_df[trade_history_df['transaction_type'] != 'C']    # drop all cancelled trades
    trade_history_df = trade_history_df.sort_values(by=['trade_datetime', 'publish_datetime', 'sequence_number'], ascending=False)

    current_date_as_datetime = datetime.now(EASTERN).replace(hour=23, minute=59, second=59, microsecond=999999).replace(tzinfo=None)    # set every field not pertaining to the day to be the end of day; set `tzinfo=None` so that we can compute the difference with `trade_datetime` otherwise the following error occurs: `TypeError: Cannot subtract tz-naive and tz-aware datetime-like objects.`
    older_than_max_num_days = diff_in_days_two_dates(current_date_as_datetime, trade_history_df['trade_datetime'], convention='exact') > MAX_NUM_DAYS_FOR_TRADES_IN_HISTORY + 1    # add 1 since the current_date_as_datetime sets all of the values to be the end of the day for the current date
    idx_of_most_recent_trade_older_than_max_num_days = np.argmax(older_than_max_num_days.values) if older_than_max_num_days.any() else len(trade_history_df)    # `np.argmax(series.values)` returns the position of the first `True` value: https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value
    return trade_history_df.head(max(idx_of_most_recent_trade_older_than_max_num_days + MAX_TRADES_USED_IN_MODEL, max_num_trades)).to_numpy()    # keep a trade if the trade is before `MAX_NUM_DAYS_FOR_TRADES_IN_HISTORY` or if the trade is one of the `max_num_trades` most recent trades; also ensures that for a trade that is within `MAX_NUM_DAYS_FOR_TRADES_IN_HISTORY`, we have the `MAX_TRADES_USED_IN_MODEL` most recent trades from that point to be used in the model if we were to do point in time pricing from that point


def upload_trade_history_to_redis(key, trade_history, redis_client):
    '''Add `trade_history` to `redis_client` for a corresponding `key`. If we are in 
    testing mode, then we should wipe the redis before using it for production.
    NOTE: If the only new trade_message is a cancellation message and there is only one trade in the history 
    (for example), we will upload a trade_history with nothing in it. This is good and desirable, because this 
    will overwrite/replace a key/CUSIP with a trade_message that has subsequently been cancelled.'''
    trade_history = pickle.dumps(trade_history)
    redis_client.set(key, trade_history)


def get_key_trade_history_pair(key, trade_history, redis_client, max_num_trades, key_transform_func=None, verbose=False, keep_cusip_in_trade_history=False):
    '''`key_transform_func` is helpful in turning a tuple into a primitive type (e.g. string) that can be 
    used as a key for Redis. Redis does not allow tuples to be used as keys.'''
    if key_transform_func is not None: key = key_transform_func(key)
    if verbose: print(f'Calling get_key_trade_history_pair(...) with key={key} and trade_history:\n{trade_history.to_markdown()}')
    features_for_each_trade_in_history = list(FEATURES_FOR_EACH_TRADE_IN_HISTORY.keys())
    if keep_cusip_in_trade_history: features_for_each_trade_in_history.append('cusip')
    trade_history = trade_history[features_for_each_trade_in_history]    # this procedure cannot be done outside of this function since it removes the `cusip` field
    if key_exists_in_redis(redis_client, key):
        old_trade_history = get_value_for_key_in_redis(redis_client, key)
        try:
            old_trade_history = pd.DataFrame(pickle.loads(old_trade_history), columns=features_for_each_trade_in_history)
        except Exception as e:
            print(f'Unable to load old trade history for key: {key} from redis. {type(e)}: {e}')
            print('old_trade_history:\n', pd.DataFrame(pickle.loads(old_trade_history)))
            raise e
        trade_history = pd.concat([trade_history, old_trade_history], ignore_index=True)
    return key, create_trade_history_numpy_array(trade_history, max_num_trades)


@function_timer
def update_trade_history_redis(new_trades, verbose=False):
    '''Update the redis corresponding to the trade history with the rows from `new_trades`. If the CUSIP does not exist 
    in the redis, then create the trade history starting from this trade(s). If the CUSIP does exist, then check if 
    there are new messages for old RTRS control numbers and substitute those new messages for the old ones. If the 
    `transaction_type` is 'C', remove the trade, otherwise, replace the old message with the newest message. Add 
    new trades to the dataframe in descending order of `trade_datetime`.
    NOTE: 'I' is an instruction or the first trade message. 'C' is to cancel the trade. We see here the trade messages 
    have the same information. 'M' and 'R' both indicate modification. 'R' is an MSRB modification (e.g., to fill in 
    par_traded when that value is initially null because of the `par_traded` over $5M rule).
    NOTE: for a particular RTRS control number, there is a specific `trade_datetime`. A more recent message for that  
    RTRS control number, such as a modify or a cancellation, would correspond to a more recent `publish_datetime`.'''
    get_cusip_trade_history_pair_caller = lambda cusip, df: get_key_trade_history_pair(cusip, df, trade_history_redis_client, MAX_NUM_TRADES_IN_HISTORY, verbose=verbose)
    # if MULTIPROCESSING:    # this procedure takes less than 1 second during testing, so no need to parallelize this with `multiprocess`
    #     with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
    #         cusip_trade_history_pairs = pool_object.starmap(get_cusip_trade_history_pair_caller, new_trades.groupby('cusip'))    # need to use starmap since `get_cusip_trade_history_pair_caller` has multiple arguments: https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
    # else:
    cusip_trade_history_pairs = [get_cusip_trade_history_pair_caller(cusip, df_for_cusip) for cusip, df_for_cusip in new_trades.groupby('cusip')]

    # upload_trade_history_to_trade_history_redis = lambda cusip, trade_history: upload_trade_history_to_redis(cusip, trade_history, trade_history_redis_client)
    # upload_to_redis_from_upload_function(cusip_trade_history_pairs, upload_trade_history_to_trade_history_redis)
    upload_to_redis_using_mset(cusip_trade_history_pairs, trade_history_redis_client, 'trade history redis')
    return cusip_trade_history_pairs


def remove_negative_and_missing_yields(trades_df: pd.DataFrame) -> pd.DataFrame:
    num_trades_before_removal = len(trades_df)
    trades_df = trades_df[~pd.isna(trades_df['yield'])]    # remove trades that have missing yields
    trades_df = trades_df[trades_df['yield'] >= 0]    # remove trades that have negative yields
    num_trades_after_removal = len(trades_df)
    if num_trades_before_removal != num_trades_after_removal: print(f'Removed {num_trades_before_removal - num_trades_after_removal} trades for having negative or missing yields, leaving {num_trades_after_removal} trades')
    return trades_df


def add_features_for_definition_of_similar(trades_df: pd.DataFrame) -> pd.DataFrame:
    '''Add the following features which are needed to define similarity: `years_to_maturity_date_by_5`, `coupon_by_1`. The 
    definition of similar is one that matches on `issue_key`, `maturity_year_by_5`, and `coupon_by_1`, where `maturity_year_by_5` 
    takes the `maturity_year` and floor divides it by 5 and `coupon_by_1` takes the coupon and floor divides it by 1.'''
    trades_df['years_to_maturity_date_by_5'] = ((trades_df['maturity_date'] - trades_df['trade_date']).dt.days // NUM_OF_DAYS_IN_YEAR) // 5
    trades_df['coupon_by_1'] = np.nan    # initialize the column
    is_zero_coupon = trades_df['coupon'] == 0
    trades_df.loc[is_zero_coupon, 'coupon_by_1'] = -1    # zero coupon has its own bucket
    trades_df.loc[~is_zero_coupon, 'coupon_by_1'] = trades_df.loc[~is_zero_coupon, 'coupon'] // 1
    trades_df = trades_df.astype({'years_to_maturity_date_by_5': int, 'coupon_by_1': int})
    return trades_df


@function_timer
def update_similar_trade_history_redis(new_trades, verbose=False):
    '''Update the redis corresponding to the similar trade history with the rows from `new_trades`. If the feature set 
    defining the related trade does not exist in the redis, then create the similar trade history starting from this 
    trade(s). If the feature set does exist, then check if there are new messages for old RTRS control numbers 
    and substitute those new messages for the old ones. If the `transaction_type` is 'C', remove the trade, 
    otherwise, replace the old message with the newest message. Add new trades to the dataframe in descending 
    order of `trade_datetime`. The definition of similar is one that matches on `issue_key`, `maturity_year_by_5`, 
    and `coupon_by_1`, where `maturity_year_by_5` takes the maturity_year and floor divides it by 5 and `coupon_by_1` 
    takes the coupon and floor divides it by 1.
    NOTE: 'I' is an instruction or the first trade message. 'C' is to cancel the trade. We see here the trade messages 
    have the same information. 'M' and 'R' both indicate modification. 'R' is an MSRB modification (e.g., to fill in 
    par_traded when that value is initially null because of the `par_traded` over $5M rule).
    NOTE: for a particular RTRS control number, there is a specific `trade_datetime`. A more recent message for that  
    RTRS control number, such as a modify or a cancellation, would correspond to a more recent `publish_datetime`.
    NOTE: Setting `verbose` to `True` provides detailed print output and is helpful for testing.'''
    new_trades = remove_negative_and_missing_yields(new_trades)    # only keep trades with nonnegative yields
    new_trades = new_trades.dropna(subset=['issue_key', 'maturity_date', 'trade_date', 'coupon'])    # remove trades that have null values for features that we need to determine similarity
    if len(new_trades) == 0:
        print('No trades to add to the similar trade history redis after removing trades with negative yields, and trades with null values for yield, issue_key, maturity_date, trade_date, or coupon.')
        return None
    new_trades = add_features_for_definition_of_similar(new_trades)
    if verbose: print(f'new_trades:\n{new_trades.drop(columns=["recent"]).to_markdown()}')    # drop `recent` column because it has a lot of data that makes it difficult to read the output

    features_to_string = lambda features: '_'.join([str(feature) for feature in features])    # `features` should be a tuple or list; NOTE: this lambda function is identical to `similar_group_to_similar_key(...)` in `app_engine/demo/server/modules/finance.py`
    get_features_similar_trade_history_pair_caller = lambda features, df: get_key_trade_history_pair(features, df, similar_trade_history_redis_client, MAX_NUM_TRADES_IN_SIMILAR_TRADE_HISTORY, features_to_string, verbose=verbose, keep_cusip_in_trade_history=True)
    features_trade_history_pairs = [get_features_similar_trade_history_pair_caller(features, df_for_features) for features, df_for_features in new_trades.groupby(['issue_key', 'years_to_maturity_date_by_5', 'coupon_by_1'])]

    # upload_similar_trade_history_to_similar_trade_history_redis = lambda features, trade_history: upload_trade_history_to_redis(features, trade_history, similar_trade_history_redis_client)
    # upload_to_redis_from_upload_function(features_trade_history_pairs, upload_similar_trade_history_to_similar_trade_history_redis)
    upload_to_redis_using_mset(features_trade_history_pairs, similar_trade_history_redis_client, 'similar trade history redis')
    return features_trade_history_pairs    # return value is unused, but perhaps can be used later to store these values into bigquery for testing


@functions_framework.http
def hello_http(request):
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
    
    Step 1: Get MSRB data. Step 2: Process MSRB data. Step 3: Get reference data from reference_data_redis. 
    Step 4: Update trade history in trade_history_redis. Step 5: Update similar trade history in 
    similar_trade_history_redis.
    NOTE: updating the reference data redis is done in this cloud function: `update-new-pipeline-ice-data`'''
    current_timestamp = datetime.now(EASTERN).replace(microsecond=0)    # remove microseconds to reduce distraction
    current_timestamp_string = current_timestamp.strftime(f'{YEAR_MONTH_DAY}_{HOUR_MIN_SEC}')
    latest_sequence_number = get_latest_sequence_number(current_timestamp_string)
    # TODO: this could be a place where we could try to find trades that happened when we don't expect them to if we run this cloud function once at the end of the day of every day, then we will always get all of the trades that day. hard case: trade occurs on sunday
    # if TESTING: latest_sequence_number = 0    # TODO: remove after testing; testing API from https://rtrsbetasubscription.msrb.org does not provide any values when `latest_sequence_number` is large
    msrb_data, latest_sequence_number = get_msrb_trade_messages(latest_sequence_number + 1, current_timestamp_string)
    msrb_data = remove_open_and_close_messages(msrb_data)
    if len(msrb_data) == 0: print('No trade messages that are not the opening and closing messages')

    msrb_data = process_msrb_data(msrb_data)
    trades_for_future_processing = get_trades_for_future_processing()
    if len(trades_for_future_processing) > 0:
        msrb_data = pd.concat([msrb_data, trades_for_future_processing], ignore_index=True) if len(msrb_data) > 0 else trades_for_future_processing
        msrb_data = msrb_data.drop_duplicates(keep='first', ignore_index=True)    # prevents duplicate trades being stored for future processing, if there are duplicate trades coming in due to re-running the function after upstream failures; the `keep` argument can be either `first` or `last`, but leaving it empty will result in all duplicates to be dropped; `ignore_index=True` will re-label the index as 0, 1, ..., n-1

    if len(msrb_data) == 0: return update_latest_sequence_number_and_return_success(current_timestamp_string, latest_sequence_number)
    cusip_list_from_msrb_data = msrb_data['cusip'].unique().tolist()    # .unique() prevents same CUSIP being queried in Redis when there are multiple trades with the same CUSIP
    reference_data, cusips_not_found = get_reference_data_from_reference_data_redis(cusip_list_from_msrb_data)
    msrb_data = store_trades_for_future_processing_and_return_trades_with_cusips_found_in_reference_data(msrb_data, cusips_not_found)
    if len(reference_data) == 0: return update_latest_sequence_number_and_return_success(current_timestamp_string, latest_sequence_number)    # do not perform any processing if no reference data is found

    reference_data = typecast_reference_data(reference_data)
    all_data = left_join_on_cusip(msrb_data, reference_data)
    all_data_after_restrictions = add_restrictions_on_interest_payment_frequency(all_data)

    if len(all_data_after_restrictions) == 0:    # do not perform any processing if there are no trades after applying restrictions
        print(f'No trades left after applying restrictions. Before restrictions:')
        print(all_data.to_markdown())
        return update_latest_sequence_number_and_return_success(current_timestamp_string, latest_sequence_number)
    
    all_data = all_data_after_restrictions
    all_data['interest_payment_frequency'] = all_data['interest_payment_frequency'].apply(get_frequency)
    all_data = typecast_yield(all_data)
    all_data = add_calc_date(all_data)
    all_data = concatenate_date_and_time_objects_into_datetime_object(all_data)
    upload_calculation_date_and_price_to_bigquery(all_data, current_timestamp)
    all_data = all_data.sort_values(by=['publish_datetime', 'sequence_number'], ascending=False)    # `publish_datetime` and `sequence_number` are ONLY for getting the most recent trade message for a given trade (RTRS control number)
    cusip_trade_history_pairs = update_trade_history_redis(all_data)
    update_similar_trade_history_redis(all_data)
    upload_trade_history_to_trade_history_bigquery(cusip_trade_history_pairs)
    return update_latest_sequence_number_and_return_success(current_timestamp_string, latest_sequence_number)
