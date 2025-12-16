'''
Description: Convert the S&P XML files loaded into the Google Cloud bucket `sp_ref_data` to 
`sp_nested` BigQuery tables and upload these tables to BigQuery.
'''
import os

# # TODO: comment the below line out when not running the function locally
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/user/ficc/ficc/mitas_creds.json'

import functions_framework
# import warnings
import logging as python_logging    # to not confuse with google.cloud.logging
from datetime import datetime
import multiprocess as mp

from auxiliary_functions import function_timer
from parse_xml import EASTERN, \
                      STORAGE_CLIENT, \
                      SP_REF_DATA_GOOGLE_CLOUD_BUCKET_NAME, \
                      BQ_SCHEMA_TYPES, \
                      BQ_SCHEMA_TO_UPLOAD_TO_BIGQUERY, \
                      SP_REFERENCE_DATA_NESTED_TABLE_NAME, \
                      parse_xml, \
                      convert_parsed_xml_to_bigquery_data, \
                      upload_to_bigquery

from google.cloud import logging


# set up logging client; https://cloud.google.com/logging/docs/setup/python
logging_client = logging.Client()
logging_client.setup_logging()


MULTIPROCESSING = False    # uploading to BigQuery does not work with the `multiprocess` module in the cloud function


def get_current_year_month_day():
    year_month_day = datetime.now(EASTERN).strftime('%Y-%m-%d')
    return year_month_day.split('-')


@function_timer
def get_xml_files_from_directory(directory: str):
    bucket = STORAGE_CLIENT.get_bucket(SP_REF_DATA_GOOGLE_CLOUD_BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=directory)
    return [blob.name for blob in blobs if not blob.name.endswith('/')]


def download_xml_file_from_gcp_storage_as_bytes(bucket_name: str, file_path: str):
    bucket = STORAGE_CLIENT.bucket(bucket_name)
    blob = bucket.blob(file_path)
    return blob.download_as_bytes()


@function_timer
def upload_filename_to_bigquery(file_name: str, 
                                bucket_name: str = SP_REF_DATA_GOOGLE_CLOUD_BUCKET_NAME, 
                                schema: list = BQ_SCHEMA_TO_UPLOAD_TO_BIGQUERY, 
                                schema_types: dict = BQ_SCHEMA_TYPES, 
                                table_name: str = SP_REFERENCE_DATA_NESTED_TABLE_NAME):
    xml_as_bytes = download_xml_file_from_gcp_storage_as_bytes(bucket_name, file_name)
    parsed_xml = parse_xml(xml_as_bytes, file_name)
    upload_to_bigquery(convert_parsed_xml_to_bigquery_data(schema_types, parsed_xml), schema, table_name)


@function_timer
def upload_list_of_filenames_to_bigquery(list_of_filenames: list) -> None:    
    if MULTIPROCESSING and len(list_of_filenames) >= os.cpu_count():
        print(f'Using multiprocessing with {os.cpu_count()} CPUs')
        with mp.Pool() as pool_object:
            pool_object.map(upload_filename_to_bigquery, list_of_filenames)
    else:
        for file_name in list_of_filenames:
            upload_filename_to_bigquery(file_name)


@functions_framework.http
def main(request):
    '''HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.'''
    current_year, current_month, current_day = get_current_year_month_day()
    directory = f'{current_year}/{current_month}/{current_day}/'
    xml_files = get_xml_files_from_directory(directory)
    if len(xml_files) == 0:
        no_xml_files_message = f'No XML files in directory: {directory}'
        python_logging.warning(no_xml_files_message)    # warnings.warn(no_xml_files_message, RuntimeWarning)
        return no_xml_files_message
    print(f'{len(xml_files)} present in directory: {directory}. Files are: {", ".join(xml_files[:3])} ... {", ".join(xml_files[-3:])}')
    upload_list_of_filenames_to_bigquery(xml_files)
    return f'{len(xml_files)} files from {directory} uploaded to sp_nested BigQuery table'
