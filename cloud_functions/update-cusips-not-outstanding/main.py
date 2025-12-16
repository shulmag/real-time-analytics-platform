'''
Description: Stores all CUSIPs that are not outstanding in a file. This file is used to speed up pricing of the entire universe since we can disregard all CUSIPs that are not outstanding.
'''
import functions_framework

import os
import math    # used for `math.ceil(...)` when supplying the `total` argument for `tqdm`
import pickle
import multiprocess as mp
from tqdm import tqdm

from auxiliary_variables import REFERENCE_DATA_REDIS_CLIENT, GOOGLE_CLOUD_BUCKET, LOCAL_NOT_OUTSTANDING_CUSIPS_FILENAME, NOT_OUTSTANDING_CUSIPS_PICKLE_FILENAME, STORAGE_CLIENT, MULTIPROCESSING, TESTING
from auxiliary_functions import upload_data, function_timer
from filtering import get_not_outstanding_cusips_set, remove_not_outstanding_cusips, is_outstanding


TEST_CUSIPS_IN_REFERENCE_DATA_REDIS = ['XXXXXXXXX', 'YYYYYYYYY', 'ZZZZZZZZZ', 'test', 'test_neg_yield_cusip']


def remove_test_cusips_from_reference_data_redis(cusips: list = TEST_CUSIPS_IN_REFERENCE_DATA_REDIS) -> None:
    '''Remove each CUSIP in `cusips` from the reference data redis.'''
    for cusip in cusips:
        if REFERENCE_DATA_REDIS_CLIENT.exists(cusip):
            print(f'Removing {cusip} from reference data redis')
            REFERENCE_DATA_REDIS_CLIENT.delete(cusip)
        else:
            print(f'{cusip} does not exist in reference data redis')


@function_timer
def get_all_cusips():
    all_cusips = REFERENCE_DATA_REDIS_CLIENT.keys('*')
    if MULTIPROCESSING and len(all_cusips) > os.cpu_count():
        print(f'Using multiprocessing with {os.cpu_count()} cores inside `get_all_cusips(...)`')
        with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
            all_cusips = pool_object.map(lambda cusip: cusip.decode('utf-8'), all_cusips)
    else:
        all_cusips = [cusip.decode('utf-8') for cusip in tqdm(all_cusips, total=len(all_cusips), disable=not TESTING)]
    return sorted(all_cusips)    # in case the procedure fails midway, sorting beforehand helps us start the procedure from a particular CUSIP since the order will be the same (unless new CUSIPs are added)


@function_timer
def get_reference_data_one_at_a_time(all_cusips: list):
    '''Use `.get(...)` on the redis client to get the reference data one at a time for each CUSIP in `all_cusips`.'''
    if MULTIPROCESSING and len(all_cusips) > os.cpu_count():
        print(f'Using multiprocessing with {os.cpu_count()} cores inside `get_reference_data_one_at_a_time(...)`')
        with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
            all_reference_data = pool_object.map(lambda cusip: pickle.loads(REFERENCE_DATA_REDIS_CLIENT.get(cusip))[0], all_cusips)    # index 0 indicates the most recent snapshot of the reference data
    else:
        all_reference_data = [pickle.loads(REFERENCE_DATA_REDIS_CLIENT.get(cusip))[0] for cusip in tqdm(all_cusips, total=len(all_cusips), disable=not TESTING)]    # index 0 indicates the most recent snapshot of the reference data
    return dict(zip(all_cusips, all_reference_data))


@function_timer
def get_pickled_reference_data_with_mget(all_cusips: list) -> list:
    '''Using `.mget(...)` on all 1 million CUSIPs was causing `redis.exceptions.ConnectionError: Connection closed by server` 
    error perhaps due to hitting the maximum network throughput of 1250 MB/s (as configured in the redis).'''
    BATCH_SIZE_FOR_MGET = 10000
    all_pickled_reference_data = []
    for cusip_idx in tqdm(range(0, len(all_cusips), BATCH_SIZE_FOR_MGET), total=math.ceil(len(all_cusips) / BATCH_SIZE_FOR_MGET), disable=not TESTING):
        all_pickled_reference_data.extend(REFERENCE_DATA_REDIS_CLIENT.mget(all_cusips[cusip_idx : cusip_idx + BATCH_SIZE_FOR_MGET]))
    return all_pickled_reference_data


@function_timer
def get_reference_data_all_at_once(all_cusips: list) -> dict:
    '''Use `.mget(...)` on the redis client to get the reference data all at once for each CUSIP in `all_cusips`. 
    This makes only one network call to the redis to reduce latency.'''
    all_pickled_reference_data = get_pickled_reference_data_with_mget(all_cusips)
    if MULTIPROCESSING and len(all_pickled_reference_data) > os.cpu_count():
        print(f'Using multiprocessing with {os.cpu_count()} cores inside `get_reference_data_all_at_once(...)`')
        with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
            all_reference_data = pool_object.map(lambda pickled_reference_data: pickle.loads(pickled_reference_data)[0], all_pickled_reference_data)    # index 0 indicates the most recent snapshot of the reference data
    else:
        all_reference_data = [pickle.loads(pickled_reference_data)[0] for pickled_reference_data in tqdm(all_pickled_reference_data, total=len(all_pickled_reference_data), disable=not TESTING)]    # index 0 indicates the most recent snapshot of the reference data
    return dict(zip(all_cusips, all_reference_data))


@function_timer
def get_all_not_outstanding_cusips(cusip_to_reference_data: dict) -> set:
    if MULTIPROCESSING and len(cusip_to_reference_data) > os.cpu_count():
        print(f'Using multiprocessing with {os.cpu_count()} cores inside `price_batches(...)`')
        with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
            is_cusip_not_outstanding = pool_object.map(lambda reference_data: not is_outstanding(reference_data), cusip_to_reference_data.values())
        not_outstanding_cusips = [cusip for cusip, cusip_not_outstanding in tqdm(zip(cusip_to_reference_data.keys(), is_cusip_not_outstanding), len(is_cusip_not_outstanding), disable=not TESTING) if cusip_not_outstanding]
    else:
        not_outstanding_cusips = [cusip for cusip, reference_data in tqdm(cusip_to_reference_data.items(), len(cusip_to_reference_data), disable=not TESTING) if not is_outstanding(reference_data)]
    return set(not_outstanding_cusips)


@function_timer
def write_outstanding_cusip_to_file_one_at_a_time_get_all_not_outstanding_cusips(all_cusips: list) -> set:
    local_filename = f'/tmp/{LOCAL_NOT_OUTSTANDING_CUSIPS_FILENAME}'
    # create an empty file if it does not exist
    if not os.path.exists(local_filename):
        with open(local_filename, 'w') as file:
            file.write('')
    
    all_pickled_reference_data = get_pickled_reference_data_with_mget(all_cusips)
    # first storing the CUSIPs into a file line by line before reading from this file so that in case there is an issue midway, we can recover and start from that point onwards
    for cusip, pickled_reference_data in tqdm(zip(all_cusips, all_pickled_reference_data), total=len(all_cusips), disable=not TESTING):
        # reference_data = pickle.loads(REFERENCE_DATA_REDIS_CLIENT.get(cusip))
        if not is_outstanding(pickle.loads(pickled_reference_data)[0]):    # index 0 indicates the most recent snapshot of the reference data
            with open(local_filename, 'a') as file:
                file.write(cusip)
                file.write('\n')
    
    not_outstanding_cusips_set = set()
    with open(local_filename, 'r') as file:
        for cusip in file:
            not_outstanding_cusips_set.add(cusip.strip())
    if os.path.exists(local_filename): os.remove(local_filename)    # remove this file to avoid confusion if it later becomes stale
    return not_outstanding_cusips_set


@functions_framework.http
def hello_http(request) -> str:
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
    
    Price a list of CUSIPs with corresponding quantities for trade type `TRADE_TYPE` and 
    store the result in a CSV with name `output_csv_filename` in Google Cloud bucket with  
    name `GOOGLE_CLOUD_BUCKET`. Return the location of the output file.'''
    # remove_test_cusips_from_reference_data_redis()
    all_cusips = get_all_cusips()
    all_cusips = remove_not_outstanding_cusips(all_cusips)
    not_outstanding_cusips_set = write_outstanding_cusip_to_file_one_at_a_time_get_all_not_outstanding_cusips(all_cusips)
    print(f'{len(not_outstanding_cusips_set)} CUSIPs added to the not outstanding CUSIPs set')
    not_outstanding_cusips_set = not_outstanding_cusips_set.union(get_not_outstanding_cusips_set())    # combine the two sets: (1) the new not outstanding CUSIPs, (2) the old not outstanding CUSIPs
    
    print(f'{len(not_outstanding_cusips_set)} CUSIPs in `not_outstanding_cusips_set`')
    if not TESTING:
        pickle_filepath = f'/tmp/{NOT_OUTSTANDING_CUSIPS_PICKLE_FILENAME}'
        with open(pickle_filepath, 'wb') as pickle_file:
            pickle.dump(not_outstanding_cusips_set, pickle_file)
        upload_data(STORAGE_CLIENT, GOOGLE_CLOUD_BUCKET, NOT_OUTSTANDING_CUSIPS_PICKLE_FILENAME, pickle_filepath)
        if os.path.exists(pickle_filepath): os.remove(pickle_filepath)    # remove this file to avoid confusion if it later becomes stale
    return 'SUCCESS'
