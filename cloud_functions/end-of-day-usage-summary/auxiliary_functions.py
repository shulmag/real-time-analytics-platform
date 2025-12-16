'''
'''
import time
from functools import wraps
from datetime import timedelta

import pandas as pd

from google.cloud import bigquery, secretmanager


MAX_NUMBER_OF_RESULTS = 1000    # determines how many entries to display; if there is high volume usage, then it does not make sense to display every single line item
SEED = 1    # ensures reproducability when performing random sample

SMTP_SERVER = 'smtp.gmail.com'
PORT = 587
EMAIL_RECIPIENTS = ['ficc-eng@ficc.ai', 'myles@ficc.ai', 'jon@ficc.ai']

YEAR_MONTH_DAY_HOUR_MIN_SEC = '%Y-%m-%d' + ' ' + '%H:%M:%S'


def access_secret_version(secret_id: str, project_id: str = 'eng-reactor-287421', version_id='latest'):
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
    response = secretmanager.SecretManagerServiceClient().access_secret_version(request={'name': name})
    payload = response.payload.data.decode('UTF-8')
    return payload


SENDER_EMAIL = access_secret_version('notifications_username')
SENDER_PASSWORD = access_secret_version('notifications_password')


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


def sqltodf(sql) -> pd.DataFrame:
    bq_client = bigquery.Client()
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()


def remove_trailing_zeros(df: pd.DataFrame, column_names: list) -> pd.DataFrame:
    '''Remove the trailing zeros in `df.column_name`.'''
    for column in column_names:
        df[column] = pd.to_numeric(df[column])    # converting to numeric removes the 0's
    return df
