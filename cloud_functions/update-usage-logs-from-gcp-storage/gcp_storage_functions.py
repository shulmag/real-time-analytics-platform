'''
'''
import urllib3
import requests
import pickle
from tqdm import tqdm

from google.api_core.exceptions import ServiceUnavailable

from auxiliary_variables import STORAGE_CLIENT
from auxiliary_functions import run_multiple_times_before_raising_error


def _get_file_from_storage_bucket(storage_client, bucket_name, file_name, verbose: bool = False):
    '''Taken directly from ficc/cloud_functions/fast_trade_history_redis_update/main.py.
    Return blob object of the file with name `file_name` from the GCP storage bucket 
    with name `bucket_name`. `using_tqdm` boolean argument determines whether we use a 
    print statement or not to not disrupt the progress bar.'''
    bucket = storage_client.bucket(bucket_name)    # use `.bucket(...)` since we know that the bucket exists: https://stackoverflow.com/questions/65310422/difference-between-bucket-and-get-bucket-on-google-storage
    blob = bucket.blob(file_name)
    if blob.exists():
        if verbose: print(f'File {file_name} found in {bucket_name}')
        return blob
    else:
        if verbose: print(f'{file_name} not found in {bucket_name}')


@run_multiple_times_before_raising_error((KeyError, urllib3.exceptions.SSLError, requests.exceptions.SSLError), 50)    # catches `KeyError: 'email'`, `KeyError: 'expires_in'`, `urllib3.exceptions.SSLError: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad, requests.exceptions.SSLError: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad record mac`
def download_pickle_file(bucket_name, file_name, verbose: bool = False):
    '''Taken directly from ficc/cloud_functions/fast_trade_history_redis_update/main.py.
    Download a pickle file `file_name` from the GCP storage bucket with name `bucket_name`. The 
    `using_tqdm` boolean argument determines whether we use a print statement or not to not 
    disrupt the progress bar.'''
    blob = _get_file_from_storage_bucket(STORAGE_CLIENT, bucket_name, file_name, False)
    if blob is None: return None
    pickle_in = blob.download_as_string()
    data = pickle.loads(pickle_in)
    if verbose: print(f'Pickle file {file_name} downloaded from {bucket_name}')
    return data


@run_multiple_times_before_raising_error((ServiceUnavailable,), 50)    # catches `google.api_core.exceptions.ServiceUnavailable: 503 DELETE ... We encountered an internal error. Please try again.`
def delete_pickle_file(bucket, bucket_name: str, file_name: str, verbose: bool = False) -> None:
    '''This function was created only to rerun the procedure in case of transient failure. 
    `bucket_name` is used solely for printing.'''
    bucket.blob(file_name).delete()
    if verbose: print(f'Deleted {file_name} from {bucket_name}')


def delete_list_of_filenames(bucket_name: str, list_of_filenames: list, using_tqdm: bool = False) -> None:
    '''Remove `list_of_filenames` from `bucket_name`.'''
    bucket = STORAGE_CLIENT.get_bucket(bucket_name)
    print(f'Deleting {len(list_of_filenames)} files from {bucket_name}')
    for filename in tqdm(list_of_filenames, total=len(list_of_filenames), disable=not using_tqdm):
        delete_pickle_file(bucket, bucket_name, filename, not using_tqdm)
    print(f'Successfully deleted {len(list_of_filenames)} files from {bucket_name}')
