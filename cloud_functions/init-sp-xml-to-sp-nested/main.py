'''
Description: Convert the S&P XML files loaded into the Google Cloud bucket `sp_ref_data` to 
`sp_nested` BigQuery tables and upload these tables to BigQuery, just for the init master file
'''

import os
import functions_framework
import logging as python_logging
from datetime import datetime
import pytz
from google.cloud import storage, bigquery, logging

# Import your existing modules
from auxiliary_functions import function_timer
from parse_xml import (
    EASTERN,
    STORAGE_CLIENT,
    SP_REF_DATA_GOOGLE_CLOUD_BUCKET_NAME,
    BQ_SCHEMA_TYPES,
    BQ_SCHEMA_TO_UPLOAD_TO_BIGQUERY,
    SP_REFERENCE_DATA_NESTED_TABLE_NAME,
    parse_xml,
    convert_parsed_xml_to_bigquery_data,
    upload_to_bigquery
)

# Set up logging client
logging_client = logging.Client()
logging_client.setup_logging()

# Initialize BigQuery client
bigquery_client = bigquery.Client()

# Constants
PROCESSED_FILES_TABLE = 'eng-reactor-287421.sp_reference_data.processed_files'
TIMEZONE = pytz.timezone('America/New_York')

def is_file_already_processed(file_name):
    query = f"""
    SELECT COUNT(1) as file_count
    FROM `{PROCESSED_FILES_TABLE}`
    WHERE file_name = @file_name
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter('file_name', 'STRING', file_name)
        ]
    )
    query_job = bigquery_client.query(query, job_config=job_config)
    result = query_job.result()
    count = next(result).file_count
    return count > 0

def mark_file_as_processed(file_name):
    table_id = PROCESSED_FILES_TABLE
    rows_to_insert = [
        {'file_name': file_name}
    ]
    errors = bigquery_client.insert_rows_json(table_id, rows_to_insert)
    if errors:
        python_logging.error(f'Failed to insert {file_name} into processed_files table: {errors}')
        raise Exception(f'Failed to mark file {file_name} as processed.')
    else:
        python_logging.info(f'File {file_name} marked as processed.')

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

@functions_framework.http
def main(request):
    """HTTP Cloud Function to process a given XML file."""
    request_json = request.get_json(silent=True)
    file_name = request_json.get('file_name') if request_json else None

    if not file_name:
        return 'No file name provided.', 400

    # Check if the file has already been processed
    if is_file_already_processed(file_name):
        message = f'File {file_name} has already been processed.'
        python_logging.info(message)
        return message, 200

    try:
        upload_filename_to_bigquery(file_name)
        # Mark the file as processed
        mark_file_as_processed(file_name)
        success_message = f'File {file_name} processed successfully.'
        python_logging.info(success_message)
        return success_message, 200
    except Exception as e:
        python_logging.error(f'Error processing file {file_name}: {e}')
        return f'Error processing file {file_name}: {e}', 500
