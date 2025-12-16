'''
'''
import urllib3
import requests
import time
from datetime import timedelta
from functools import wraps
import pickle
import logging as python_logging    # to not confuse with google.cloud.logging

import numpy as np
import pandas as pd

from auxiliary_variables import REFERENCE_DATA_FEATURE_TO_INDEX


def function_timer(function_to_time):
    '''This function is to be used as a decorator. It will print out the execution time of `function_to_time`. 
    It is very similar to the decorator by the same name in `app_engine/demo/server/modules/ficc/utils/auxiliary_functions.py`.'''
    @wraps(function_to_time)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
    def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
        python_logging.info(f'BEGIN {function_to_time.__name__}')
        print(f'BEGIN {function_to_time.__name__}')    # need this for local testing since python_logging does not print to the terminal which is the only display for text that is visible during local testing
        start_time = time.time()
        result = function_to_time(*args, **kwargs)
        end_time = time.time()
        python_logging.info(f'END {function_to_time.__name__}. Execution time: {timedelta(seconds=end_time - start_time)}')
        print(f'END {function_to_time.__name__}. Execution time: {timedelta(seconds=end_time - start_time)}')    # need this for local testing since python_logging does not print to the terminal which is the only display for text that is visible during local testing
        return result
    return wrapper


def run_multiple_times_before_failing(error_types: tuple, max_runs: int):
    '''This function returns a decorator. It will run `function` over and over again until it does not 
    raise an Exception for a maximum of `max_runs` times.
    NOTE: max_runs = 1 is the same functionality as not having this decorator.
    NOTE: this is taken directly from `app_engine/demo/server/modules/ficc/utils/auxiliay_functions.py::run_multiple_times_before_failing(...)`'''
    def decorator(function):
        @wraps(function)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
        def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
            runs_so_far = 0
            while runs_so_far < max_runs:
                try:
                    return function(*args, **kwargs)
                except error_types as e:
                    runs_so_far += 1
                    if runs_so_far >= max_runs:
                        python_logging.warning(f'Already caught {type(e)}: {e}, {max_runs} times in {function.__name__}, so will now raise the error')
                        raise e
                    python_logging.warning(f'Caught {type(e)}: {e}, and will retry {function.__name__} {max_runs - runs_so_far} more times')
                    time.sleep(1)    # have a one second delay to prevent overloading the server
        return wrapper
    return decorator


def upload_data(storage_client, bucket_name, file_name, path_to_file=None):
    '''This function is used to upload data to the cloud bucket.
    NOTE: taken directly from app_engine/demo/server/modules/ficc/utils/gcp_storage_functions.py::upload_data(...).'''
    if path_to_file is None: path_to_file = file_name
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(path_to_file)
    print(f'File from {path_to_file} uploaded to Google cloud storage: {bucket_name}/{file_name}.')


@run_multiple_times_before_failing((KeyError, urllib3.exceptions.SSLError, requests.exceptions.SSLError), 10)    # catches KeyError: 'email', KeyError: 'expires_in', urllib3.exceptions.SSLError: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad, requests.exceptions.SSLError: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad record mac 
def download_pickle_file(storage_client, bucket_name, file_name):
    '''This function is used to download the data from the GCP storage bucket.
    It is assumed that we will be downloading a pickle file. The decorator solves 
    the following problem: GCP limits how quickly files can be downloaded from 
    buckets and raises an `SSLError` or a `KeyError` when the buckets are accessed 
    too quickly in succession.'''
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    if not blob.exists():    # `file_name` in `bucket_name` does not exist
        print(f'File {file_name} does not exist in {bucket_name}.')
        return None
    pickle_in = blob.download_as_string()
    data = pickle.loads(pickle_in)
    print(f'File {file_name} downloaded from {bucket_name}.')
    return data


def get_feature_value(single_cusip_data: pd.Series | np.ndarray, feature: str):
    '''`single_cusip_data` may be a `pd.Series` if the data is coming from point-in-time pricing, or it 
    may be a `np.array` if it is coming directly from the redis.
    NOTE: this is identical to `app_engine/demo/server/modules/auxiliary_functions.py::get_feature_value(...)`.'''
    if isinstance(single_cusip_data, np.ndarray): return single_cusip_data[REFERENCE_DATA_FEATURE_TO_INDEX[feature]]
    if isinstance(single_cusip_data, pd.Series): return single_cusip_data[feature]
    raise ValueError(f'single_cusip_data is of type: {type(single_cusip_data)} which is neither a `pd.Series` nor a `np.array`')
