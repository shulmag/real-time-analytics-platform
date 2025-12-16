'''
'''
from functools import wraps
import time
import logging as python_logging    # to not confuse with google.cloud.logging
from datetime import timedelta, datetime
import pickle

import redis
import multiprocess as mp    # using `multiprocess` instead of `multiprocessing` because function to be called in `map` is in the same file as the function which is calling it: https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror

import urllib3
import requests

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from google.cloud import storage, bigquery

from auxiliary_variables import MULTIPROCESSING, YEAR_MONTH_DAY, HOUR_MIN_SEC, MSRB_INTRADAY_FILES_BUCKET_NAME, PROJECT_ID, LOCATION


def function_timer(function_to_time):
    '''This function is to be used as a decorator. It will print out the execution time of `function_to_time`.'''
    @wraps(function_to_time)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
    def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
        print(f'BEGIN {function_to_time.__name__}')
        start_time = time.time()
        result = function_to_time(*args, **kwargs)
        end_time = time.time()
        print(f'END {function_to_time.__name__}. Execution time: {timedelta(seconds=end_time - start_time)}')
        return result
    return wrapper


def run_multiple_times_before_raising_error(errors, max_runs):    # using the same formatting from https://stackoverflow.com/questions/10176226/how-do-i-pass-extra-arguments-to-a-python-decorator
    '''This function customizes the returned decorator for a custom list of `errors` and a number of `max_runs`.'''
    def run_multiple_times_before_failing(function):
        '''This function is to be used as a decorator. It will run `function` over and over again until it does not 
        raise an Exception for a maximum of `max_runs` (specified below) times. `max_runs = 1` is the same functionality 
        as not having this decorator. It solves the following problems: (1) GCP limits how quickly files can be 
        downloaded from buckets and raises an `SSLError` or a `KeyError` when the buckets are accessed too quickly in 
        succession, (2) redis infrequently fails due to connectionError which succeeds upon running the function again.'''
        @wraps(function)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
        def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
            nonlocal max_runs    # `max_runs` belongs to the outer scope
            while max_runs > 0:
                try:
                    return function(*args, **kwargs)
                except tuple(errors) as e:
                    max_runs -= 1
                    if max_runs == 0: raise e
                    python_logging.warning(f'{function.__name__} raise error: {e}, but we will re-attempt it {max_runs} more times before failing')    # raise warning of error instead of error itself
                    time.sleep(1)    # have a one second delay to prevent overloading the number of calls
        return wrapper
    return run_multiple_times_before_failing


def run_five_times_before_raising_redis_connector_error(function):
    return run_multiple_times_before_raising_error((redis.exceptions.ConnectionError, redis.exceptions.TimeoutError), 5)(function)


def convert_to_date(date):
    '''Converts an object, either of type pd.Timestamp or datetime.datetime to a 
    datetime.date object.'''
    if isinstance(date, pd.Timestamp): date = date.to_pydatetime()
    if isinstance(date, datetime): date = date.date()
    return date    # assumes the type is datetime.date


def compare_dates(date1, date2):
    '''This function compares two date objects whether they are in Timestamp or datetime.date. 
    The different types are causing a future warning. If date1 occurs after date2, return 1. 
    If date1 equals date2, return 0. Otherwise, return -1.'''
    return (convert_to_date(date1) - convert_to_date(date2)).total_seconds()


def dates_are_equal(date1, date2):
    '''This function directly calls `compare_dates` to check if two dates are equal.'''
    return compare_dates(date1, date2) == 0


def _diff_in_days_two_dates_360_30(end_date, start_date):
    '''This function calculates the difference in days using the 360/30 
    convention specified in MSRB Rule Book G-33, rule (e).'''
    Y2 = end_date.year
    Y1 = start_date.year
    M2 = end_date.month
    M1 = start_date.month
    D2 = end_date.day
    D1 = start_date.day
    D1 = min(D1, 30)
    if D1 == 30: D2 = min(D2, 30)
    return (Y2 - Y1) * 360 + (M2 - M1) * 30 + (D2 - D1)


def _diff_in_days_two_dates_exact(end_date, start_date):
    diff = end_date - start_date
    if isinstance(diff, pd.Series): return diff.dt.days    # https://stackoverflow.com/questions/60879982/attributeerror-timedelta-object-has-no-attribute-dt
    else: return diff.days


ACCEPTED_CONVENTIONS = {'360/30': _diff_in_days_two_dates_360_30, 
                        'exact': _diff_in_days_two_dates_exact}


def diff_in_days_two_dates(end_date, start_date, convention='360/30'):
    if convention not in ACCEPTED_CONVENTIONS:
        print('unknown convention', convention)
        return None
    return ACCEPTED_CONVENTIONS[convention](end_date, start_date)


def trunc(x, decimal_places):
    '''This file truncations an input to a specified number of decimal places.

    >>> trunc(3.33333, 3)
    3.333
    >>> trunc(3.99499, 3)
    3.994
    >>> trunc(30.99499, 3)
    30.994
    '''
    ten_places = 10 ** decimal_places
    return ((x * ten_places) // 1) / ten_places


def trunc_and_round_price(price):
    '''This function rounds the final price according to MSRB Rule Book G-33, rule (d).'''
    return trunc(price, 3)


def upload_to_storage(file_name, file_text, bucket_name=MSRB_INTRADAY_FILES_BUCKET_NAME):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(file_text)
    print(f'File {file_name} uploaded to in {bucket_name}')


def _get_file_from_storage_bucket(storage_client, bucket_name, file_name):
    '''Return blob object of the file with name `file_name` from the GCP storage bucket 
    with name `bucket_name`.'''
    bucket = storage_client.bucket(bucket_name)    # use `.bucket(...)` since we know that the bucket exists: https://stackoverflow.com/questions/65310422/difference-between-bucket-and-get-bucket-on-google-storage
    blob = bucket.blob(file_name)
    if blob.exists():
        print(f'File {file_name} found in {bucket_name}/{file_name}')
        return blob
    else:
        print(f'{file_name} not found in {bucket_name}')


@run_multiple_times_before_raising_error((KeyError, urllib3.exceptions.SSLError, requests.exceptions.SSLError), 50)    # catches KeyError: 'email', KeyError: 'expires_in', urllib3.exceptions.SSLError: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad, requests.exceptions.SSLError: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad record mac 
def download_pickle_file(bucket_name, file_name):
    '''Download a pickle file `file_name` from the GCP storage bucket with name `bucket_name`.'''
    client = storage.Client()
    blob = _get_file_from_storage_bucket(client, bucket_name, file_name)
    if blob is None: return None
    pickle_in = blob.download_as_string()
    data = pickle.loads(pickle_in)
    print(f'Pickle file {file_name} downloaded from {bucket_name}/{file_name}')
    return data


def delete_file(bucket_name, file_name):
    '''Delete `file_name` from the GCP storage bucket with name `bucket_name`.'''
    client = storage.Client()
    blob = _get_file_from_storage_bucket(client, bucket_name, file_name)
    if blob is not None: blob.delete()


def upload_to_bigquery(table_name, schema, df_or_dict, function_name=None):
    '''Uploads `df_or_dict` (after converting the object to a dictionary if it is passed in as a pandas DataFrame) to 
    `table_name` (with `schema`) using the function `load_table_from_json(...)`. If `function_name` is not `None`, then 
    we wait until the job is completed and print the job status (success or failure).'''
    if isinstance(df_or_dict, pd.DataFrame): df_or_dict = df_to_json_dict(df_or_dict)
    client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition='WRITE_APPEND')
    table_name = f'{PROJECT_ID}.{table_name}'    # NOTE: table will be automatically created the first time that this function is run
    job = client.load_table_from_json(df_or_dict, table_name, job_config=job_config)
    if function_name is not None:
        try:    # removing this try...catch would make `client.load_table_from_dataframe(...)` asynchronous since `job.result()` would be removed
            job.result()    # waits for job to complete
            print(f'Upload Successful in `{function_name}`')
        except Exception as e:
            print(f'Upload Failed in `{function_name}`')
            print('Dataframe')
            num_rows_to_print_at_once = 50    # if the dataframe is too large, then it gets truncated in the logs, so we print chunks of it so that it can all be viewed; 50 is chosen since the last time this happened, we were able to see 84 rows in the logs before truncation
            for start_idx in range(0, len(df_or_dict), num_rows_to_print_at_once):
                end_idx = min(start_idx + num_rows_to_print_at_once, len(df_or_dict))
                print(f'Rows {start_idx} to {end_idx - 1}')
                print(df_or_dict[start_idx : end_idx])
            print('Schema')
            print(schema)
            python_logging.warning(f'{function_name} was not able to upload to {table_name} due to {type(e)}: {e}')    # raise warning of error instead of error itself


def typecast_for_bigquery(df, column_to_dtype):
    '''Typecast each column in `column_to_dtype` for `df` to its corresponding dtype. Sometimes the numerical data 
    that is supposed to be an integer comes in as a float causing an error when attempting to upload to bigquery; use 
    `Int64` instead of `int` to allow conversion when there are `None` values: https://stackoverflow.com/questions/26614465/python-pandas-apply-function-if-a-column-value-is-not-null
    TODO: figure out why certain columns come in as float when they otherwise mostly come in as integer (e.g. `issue_key`).'''
    for column, dtype in column_to_dtype.items():
        if column in df.columns: df[column] = df[column].astype(dtype)
    return df


def df_to_json_dict(df):
    '''Convert a dataframe into a json serializable dict. Avoids `TypeError: Object of type ___ is not JSON serializable`.'''
    df = typecast_for_bigquery(df, {'rtrs_control_number': int, 
                                    'par_traded': int, 
                                    'sequence_number': 'Int64', 
                                    'calc_day_cat': 'Int64'})    # sometimes the numerical data that is supposed to be an integer comes in as a float causing an error when attempting to upload to BigQuery; use `Int64` instead of `int` to allow conversion when there are `None` values: https://stackoverflow.com/questions/26614465/python-pandas-apply-function-if-a-column-value-is-not-null
    for column_name in df.columns:
        column = df[column_name]
        if is_datetime(column):
            df[column_name] = column.dt.strftime(YEAR_MONTH_DAY + 'T' + HOUR_MIN_SEC)
        elif column_name.endswith('_date'):
            df[column_name] = column.astype('string')
    df = df.mask(df == float('inf'), pd.NA)    # replaces `inf`, which may appear for certain prices as a result of `compute_price`; the reason that we do not use `df.replace(float('inf'), pd.NA)` is because of it sometimes raises `TypeError: boolean value of NA is ambiguous` which occurs because Pandas is trying to interpret the pd.NA value as a boolean in the process of replacing the float('inf') values in the DataFrame
    df = df.astype(object).where(pd.notnull(df), None)    # need to replace all NaN values with `None` in order to properly upload to BigQuery: https://stackoverflow.com/questions/14162723/replacing-pandas-or-numpy-nan-with-a-none-to-use-with-mysqldb
    return df.to_dict('records')


def upload_to_redis_from_upload_function(pairs, upload_function):
    '''Upload each pair from `pairs` to the redis using the `upload_function`.'''
    if MULTIPROCESSING:
        with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
            pool_object.starmap(upload_function, pairs)    # need to use starmap since `upload_function` has multiple arguments: https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
    else:
        [upload_function(key, trade_history) for key, trade_history in pairs]
    return pairs    # unused return value


@run_five_times_before_raising_redis_connector_error
def set_pairs_in_redis(redis_client, pairs: dict):
    '''Created this one line function solely to use the decorator: `run_five_times_before_raising_redis_connector_error`.'''
    return redis_client.mset(pairs)


@function_timer
def upload_to_redis_using_mset(pairs: list, redis_client, redis_name: str = 'redis'):
    '''Add each item in `pairs` to `redis_client`, by first converting it to a dictionary and then using `.mset(...)` 
    Using `.mset(...)` instead of `.set(...)` uploads all the values at once and does not require multiple round trips 
    to the redis server. If we are in testing mode, then we should wipe the redis before using it for production. 
    `redis_name` is used only for printing.
    NOTE: If the only new trade_message is a cancellation message and there is only one trade in the history 
    (for example), we will upload a trade_history with nothing in it. This is good and desirable, because this 
    will overwrite/replace a key/CUSIP with a trade_message that has subsequently been cancelled.'''
    pairs_to_upload_to_redis = {key: pickle.dumps(unpickled_value) for key, unpickled_value in pairs}
    set_pairs_in_redis(redis_client, pairs_to_upload_to_redis)
    print(f'Uploaded {len(pairs_to_upload_to_redis)} keys to {redis_name}')
    return pairs_to_upload_to_redis    # unused return value
