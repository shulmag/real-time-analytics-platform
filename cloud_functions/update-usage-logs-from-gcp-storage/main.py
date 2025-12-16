'''
Description: Details: https://www.notion.so/Log-Usage-for-Product-aed87e70670941eca59a532f4e9f3283
             This cloud function takes the usage logs stored as pickle files in the `server_logging` Google Cloud 
             bucket and uploads them to the BigQuery table `eng-reactor-287421.api_calls_tracker.usage_data`. Only the first 
             MAX_NUMBER_OF_LOG_FILES_TO_PROCESS log files are processed because this ensures that the entire procedure takes 
             at most 15 minutes to complete and also because this means that with high likelihood (tested empirically), the 
             entire procedure takes at most 8 GB of memory which is what is allocated to the cloud function. Before uploading 
             to the BigQuery table, the log files are combined into a single log file that has all of the log entries, and a 
             single upload call is made to the BigQuery table. This avoids the following error: 
             <class 'google.api_core.exceptions.Forbidden'>: 403 Quota exceeded: Your table exceeded quota for imports or query appends per table. For more information, see https://cloud.google.com/bigquery/docs/troubleshoot-quotas; reason: quotaExceeded, location: load_job_per_table.long, message: Quota exceeded: Your table exceeded quota for imports or query appends per table. For more information, see https://cloud.google.com/bigquery/docs/troubleshoot-quotas
'''
# # TODO: comment the below line out when not running the function locally
# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/user/ficc/ficc/mitas_creds.json'


import functions_framework

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday    # used to create a business day defined on the US federal holiday calendar that can be added or subtracted to a datetime
from datetime import datetime, timedelta
from pytz import timezone
from tqdm import tqdm

from auxiliary_variables import YEAR_MONTH_DAY, STORAGE_CLIENT, GCP_BUCKET_NAME, MAX_NUMBER_OF_LOG_FILES_TO_PROCESS, MAX_NUMBER_OF_LOG_ITEMS_TO_INSERT_AT_ONCE
from gcp_storage_functions import download_pickle_file, delete_list_of_filenames
from gcp_bigquery_functions import upload_usage_dicts_to_bigquery_stream


class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]
BUSINESS_DAY = CustomBusinessDay(calendar=USHolidayCalendarWithGoodFriday())    # used to skip over holidays when adding or subtracting business days


def get_all_usage_log_filenames(date_as_string: str, internal_usage_only: bool = None) -> list:
    '''Get all of the files inside the folder with name `date_as_string` inside `GCP_BUCKET_NAME`, 
    sorted by creation time ascending (earliest first). `internal_usage_only` is a boolean (default 
    value `None`) which determines whether we should only return filenames with '@ficc.ai' in the 
    filename or to exclude all filenames with '@ficc.ai' in the filename. If `internal_usage_only` 
    is `None`, then return all filenames without any exclusions.'''
    if internal_usage_only is None:
        condition = lambda filename: True    # default condition to return all filenames
    else:
        if internal_usage_only:
            condition = lambda filename: '@ficc.ai' in filename
            print(f'Keeping only filenames with "@ficc.ai" in the filename (i.e., internal usage)')    # for debugging purposes
        else:
            condition = lambda filename: '@ficc.ai' not in filename
            print(f'Keeping only filenames without "@ficc.ai" in the filename (i.e., external usage)')    # for debugging purposes

    bucket = STORAGE_CLIENT.get_bucket(GCP_BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=date_as_string))    # call `list(...)` on the iterator to be able to call `len(...)` on the object
    # print(f'Found {len(blobs)} total usage log files (pre filtering) in {GCP_BUCKET_NAME}/{date_as_string}')    # for debugging purposes
    blobs_without_folders = [blob for blob in blobs if not blob.name.endswith('/')]    # ignore all folders
    sorted_blobs = sorted(blobs_without_folders, key=lambda blob: blob.time_created)    # in ascending order of time created (prioritizes uploading older logs first)
    return [blob.name for blob in sorted_blobs if condition(blob.name)]    # return the names of the blobs that satisfy `condition`


def combine_usage_log_filenames_into_list_of_dicts(list_of_filenames: list, use_tqdm: bool = False) -> list:
    '''Take a `list_of_filenames` and unpickle all of the files located in `folder_name` in `GCP_STORAGE_BUCKET`. 
    Each filename should correspond to a list of dictionaries, so combine each of the lists together.'''
    list_of_dicts = []
    for filename in tqdm(list_of_filenames, total=len(list_of_filenames), disable=not use_tqdm):
        list_of_dicts.extend(download_pickle_file(GCP_BUCKET_NAME, filename, not use_tqdm))
    print(f'Combining the usage produced {len(list_of_dicts)} log entries')
    return list_of_dicts


def remove_previous_day_folder(current_date_as_string: str) -> None:
    '''NOTE: function is unused due to realization that Google Cloud storage "automatically" removes a 
    folder with nothing in it due to the flat namespace (there is no file that exists if a folder is empty). 
    NOTE: should decrement by a calendar day instead of business day because we now have weekend usage.
    Get the previous day from `current_date_as_string` and remove this folder if it empty.'''
    current_date = pd.to_datetime(datetime.strptime(current_date_as_string, YEAR_MONTH_DAY))
    previous_business_date = current_date - (BUSINESS_DAY * 1)
    previous_business_date_as_string = previous_business_date.strftime(YEAR_MONTH_DAY)

    bucket = STORAGE_CLIENT.get_bucket(GCP_BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=previous_business_date_as_string))    # call `list(...)` on the iterator to be able to call `len(...)` on the object
    folder_exists = False
    folder_is_empty = False
    if len(blobs) == 0:    # folder does not exist and is considered empty
        folder_is_empty = True
    elif len(blobs) == 1 and blobs[0].name == previous_business_date_as_string:    # folder exists and is empty
        folder_exists = True
        folder_is_empty = True
    else:    # folder exists and is not empty
        folder_exists = True

    if folder_exists and folder_is_empty:
        blobs[0].delete()
        print(f'Deleted the empty folder: {GCP_BUCKET_NAME}/{previous_business_date_as_string}')


def get_argument_value_from_request(request, argument_name: str) -> bool:
    request_json = request.get_json(silent=True)    # `silent=True` will not raise an error if the JSON parsing fails (e.g., due to invalid JSON in the request body), and will instead return `None`
    request_args = request.args

    argument_found_in_request_json = request_json and argument_name in request_json
    argument_found_in_request_args = request_args and argument_name in request_args
    
    if not argument_found_in_request_json and not argument_found_in_request_args:
        print(f'Did not receive `{argument_name}` from request JSON or request args so returning `False` by default')    # for debugging purposes
        return False

    if argument_found_in_request_json:
        argument_value = request_json[argument_name]
        location_where_argument_found = 'JSON'
    elif argument_found_in_request_args:
        argument_value = request_args[argument_name]
        location_where_argument_found = 'args'
    print(f'Received `{argument_name}` from request {location_where_argument_found}: {argument_value}; {type(argument_value)}')    # for debugging purposes
    return argument_value == 'True'    # `argument_value` will be passed in as a string


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
    '''
    internal_usage = get_argument_value_from_request(request, 'internal_usage')    # this will determine whether to log to `usage_data_internal` or `usage_data`
    current_datetime = datetime.now(timezone('US/Eastern'))
    if current_datetime.hour < 5: current_datetime = current_datetime - timedelta(days=1)    # if before 5 am, then use yesterday's date
    current_date_as_string = current_datetime.strftime(YEAR_MONTH_DAY)
    print(f'Inspecting folder with name: {current_date_as_string}')
    all_usage_log_filenames = get_all_usage_log_filenames(current_date_as_string, internal_usage_only=internal_usage)    # sorted in ascending order of time created
    print(f'Processing {len(all_usage_log_filenames)} usage log files (post filtering)')
    if len(all_usage_log_filenames) > 0:
        if len(all_usage_log_filenames) > MAX_NUMBER_OF_LOG_FILES_TO_PROCESS:
            print(f'Only processing the first {MAX_NUMBER_OF_LOG_FILES_TO_PROCESS} log files to finish upload and delete within 30 minutes')
            all_usage_log_filenames = all_usage_log_filenames[:MAX_NUMBER_OF_LOG_FILES_TO_PROCESS]
        usage_log_list_of_dicts = combine_usage_log_filenames_into_list_of_dicts(all_usage_log_filenames)
        upload_usage_dicts_to_bigquery_stream(usage_log_list_of_dicts, internal_usage_table=internal_usage, chunk_size=MAX_NUMBER_OF_LOG_ITEMS_TO_INSERT_AT_ONCE)
        delete_list_of_filenames(GCP_BUCKET_NAME, all_usage_log_filenames)
    return 'SUCCESS'
