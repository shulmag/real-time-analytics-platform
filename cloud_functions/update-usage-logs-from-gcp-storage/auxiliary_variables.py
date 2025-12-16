'''
'''
from google.cloud import storage, bigquery, logging

MAX_NUMBER_OF_LOG_FILES_TO_PROCESS = 1500
MAX_NUMBER_OF_LOG_ITEMS_TO_INSERT_AT_ONCE = 100    # prevents memory issues when calling `upload_usage_dicts_to_bigquery_stream(...)`
YEAR_MONTH_DAY = '%Y-%m-%d'

# set up logging client; https://cloud.google.com/logging/docs/setup/python
logging_client = logging.Client()
logging_client.setup_logging()

GCP_BUCKET_NAME = 'server_logging'
STORAGE_CLIENT = storage.Client()
BQ_CLIENT = bigquery.Client()

# specify the target tables for logging usage data in BigQuery
DATASET_NAME = 'eng-reactor-287421.api_calls_tracker'
USAGE_DATA_LOGGING_TABLE = f'{DATASET_NAME}.usage_data'
USAGE_DATA_INTERNAL_LOGGING_TABLE = f'{DATASET_NAME}.usage_data_internal'

## taken directly from `ficc/app_engine/demo/server/modules/auxiliary_variables.py`
# used to create the scheme for logging
LOGGING_FEATURES = {'user': 'string', 
                    'api_call': 'BOOLEAN', 
                    'time': 'timestamp', 
                    'cusip': 'string', 
                    'direction': 'string', 
                    'quantity': 'integer', 
                    'ficc_price': 'numeric', 
                    'ficc_ytw': 'numeric', 
                    'yield_spread': 'numeric', 
                    'ficc_ycl': 'numeric', 
                    'calc_date': 'string', 
                    'daily_schoonover_report': 'BOOLEAN', 
                    'real_time_yield_curve': 'BOOLEAN', 
                    'batch': 'BOOLEAN', 
                    'show_similar_bonds': 'BOOLEAN', 
                    'error': 'BOOLEAN', 
                    'model_used': 'string'}
RECENT_FEATURES = ['yield_spread', 'yield_spread2', 'yield_spread3', 'yield_spread4', 'yield_spread5']

## taken directly from `ficc/app_engine/demo/server/modules/logging_functions.py`
# create schema
logging_schema = [bigquery.SchemaField(feature, dtype) for feature, dtype in LOGGING_FEATURES.items()]
recent_schema = bigquery.SchemaField('recent', 'RECORD', mode='REPEATED', fields=[bigquery.SchemaField(feature, 'numeric') for feature in RECENT_FEATURES])
logging_schema.append(recent_schema)
LOGGING_JOB_CONFIG = bigquery.LoadJobConfig(schema=logging_schema)
