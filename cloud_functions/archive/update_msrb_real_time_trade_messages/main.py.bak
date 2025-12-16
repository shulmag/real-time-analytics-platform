'''The only difference between this cloud function and `update_msrb_real_time_trade_messages_v2` is that this one is deployed 
in region us-east-4, and that one is deployed in region us-west-1. Deploying in us-west-1 fixed issues we were having with 
calling the MSRB API. See the docstring for the function `get_msrb_trade_messages(...)` in this file for more details.'''
from functools import wraps
import time
from datetime import timedelta, datetime
from urllib.error import URLError  # used in `get_msrb_trade_messages(...)` to raise an error when MSRB API is failing
import json
import requests
from datetime import datetime
from pytz import timezone
from google.cloud import bigquery, storage, secretmanager
from send_email import send_error_email

# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS']='GOOGLE_APPLICATION_CREDENTIALS.json'

PROJECT_ID = 'eng-reactor-287421'

# Construct a BigQuery client object.
bq_client = bigquery.Client()
# ET time zone:
eastern = timezone('US/Eastern')

client = storage.Client()
bucket = client.get_bucket('msrb_intraday_real_time_trade_files')


def function_timer(function_to_time):
    '''This function is to be used as a decorator. It will print out the execution time of `function_to_time`.'''
    @wraps(function_to_time)  # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
    def wrapper(*args, **kwargs):  # using the same formatting from https://docs.python.org/3/library/functools.html
        print(f'BEGIN {function_to_time.__name__}')
        start_time = time.time()
        result = function_to_time(*args, **kwargs)
        end_time = time.time()
        print(f'END {function_to_time.__name__}. Execution time: {timedelta(seconds=end_time - start_time)}')
        return result

    return wrapper


def _get_credentials(secret_id, version_id, project_id=PROJECT_ID):
    client = secretmanager.SecretManagerServiceClient()  # create the Secret Manager client
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'  # build the resource name of the secret version
    response = client.access_secret_version(request={'name': name})
    payload = response.payload.data.decode('UTF-8')
    return payload


def get_latest_username():
    '''Get the latest username from SecretManagerServiceClient.'''
    return _get_credentials('msrb_username', 'latest')


def get_latest_password():
    '''Get the latest password from SecretManagerServiceClient.'''
    return _get_credentials('msrb_password', 'latest')


def get_msrb_trade_messages(beginning_sequence_number):
    '''If this function  has errors, refer to https://docs.google.com/document/d/1RPeXWpOehtTcR98B3l5LgR2f6MxxpRKQRDOpYhGroPU/edit?usp=sharing.'''
    base_url = 'https://rtrsprodsubscription.msrb.org'  # if not TESTING else 'https://rtrsbetasubscription.msrb.org'    # the testing URL is so that we can query MSRB as frequently as possible; it may not have all of the trade messages but can still be used for testing
    url = f'{base_url}/rtrssubscriptionwebservice/api/Subscription.GetNext?beginSequence={beginning_sequence_number}'
    headers = {
        'credentials': get_latest_username() + ',' + get_latest_password(),
        'accept': 'application/json',
    }
    response = requests.request('GET', url, headers=headers)
    if not response.ok:
        raise URLError(f'Getting MSRB trade messages using URL {url} failed with status code: {response.status_code} and reason: {response.reason}, so we do not proceed further. Headers used when making API call: {headers}. Headers from the returned response: {response.headers}')
        # return pd.DataFrame()    # used if we want to make progress even in the presence of MSRB API errors
    response = json.loads(response.text)
    return response


def trade_message_to_full_db_record(trade_message):
    names = [
        'message_type',
        'sequence_number',
        'delete',
        'rtrs_control_number',
        'trade_type',
        'transaction_type',
        'cusip',
        'security_description',
        'dated_date',
        'coupon',
        'maturity_date',
        'when_issued',
        'assumed_settlement_date',
        'trade_date',
        'time_of_trade',
        'settlement_date',
        'par_traded',
        'dollar_price',
        'yield',
        'brokers_broker',
        'is_weighted_average_price',
        'is_lop_or_takedown',
        'publish_date',
        'publish_time',
        'version',
        'unable_to_verify_dollar_price',
        'is_alternative_trading_system',
        'is_non_transaction_based_compensation',
        'is_trade_with_a_par_amount_over_5MM',
    ]
    parsed_messages = trade_message.split(',')
    temp_dict = dict([s.split('=', 1) for s in parsed_messages])
    dict_record = {int(key): temp_dict[key] for key in temp_dict.keys()}
    dict_record_with_headers = {
        names[i]: dict_record[i + 1] if i + 1 in dict_record else None for i in range(len(names))
    }
    del dict_record_with_headers['delete']
    return data_type_casting(dict_record_with_headers)


def data_type_casting(dict_trade_message):
    # The 'except ValueError:' place null value for every value retuened that does not fit the data type - we lose data that way, but perhaps not useful data. Let's discuss.
    dict_data_type = {
        'upload_date': 'date',
        'message_type': 'string',
        'sequence_number': 'integer',
        'rtrs_control_number': 'integer',
        'trade_type': 'string',
        'transaction_type': 'string',
        'cusip': 'string',
        'security_description': 'string',
        'dated_date': 'date',
        'coupon': 'numeric',
        'maturity_date': 'date',
        'when_issued': 'boolean',
        'assumed_settlement_date': 'date',
        'trade_date': 'date',
        'time_of_trade': 'time',
        'settlement_date': 'date',
        'par_traded': 'numeric',
        'dollar_price': 'float',
        'yield': 'float',
        'brokers_broker': 'string',
        'is_weighted_average_price': 'boolean',
        'is_lop_or_takedown': 'boolean',
        'publish_date': 'date',
        'publish_time': 'time',
        'version': 'numeric',
        'unable_to_verify_dollar_price': 'boolean',
        'is_alternative_trading_system': 'boolean',
        'is_non_transaction_based_compensation': 'boolean',
        'is_trade_with_a_par_amount_over_5MM': 'boolean',
    }
    for k, v in dict_trade_message.items():
        try:
            if dict_data_type[k] == 'date':
                if v != None:
                    if len(v) == 8:
                        dict_trade_message[k] = datetime.strptime(v, '%Y%m%d').strftime('%Y-%m-%d')
                    elif len(v) == 6:
                        dict_trade_message[k] = datetime.strptime(v, '%m%d%y').strftime('%Y-%m-%d')
                    else:
                        dict_trade_message[k] = None
            if dict_data_type[k] == 'numeric' or dict_data_type[k] == 'float':
                if v == 'MM+':
                    dict_trade_message[k] = None
                    dict_trade_message['is_trade_with_a_par_amount_over_5MM'] = 'Y'
                elif v != '' and v != None:
                    dict_trade_message[k] = float(v)
                else:
                    dict_trade_message[k] = None
            if dict_data_type[k] == 'string':
                dict_trade_message[k] = v
            if dict_data_type[k] == 'integer' and v != None:
                dict_trade_message[k] = int(v)
            if dict_data_type[k] == 'boolean':
                dict_trade_message[k] = v == 'Y'
            elif dict_data_type[k] == 'time':
                if v != None:
                    dict_trade_message[k] = datetime.strftime(datetime.strptime(v, '%H%M%S'), '%H:%M:%S')
        except Exception:
            dict_trade_message[k] = None

    dict_trade_message['upload_date'] = datetime.now(eastern).strftime('%Y-%m-%d')
    return dict_trade_message


def add_to_bigquery(rows_to_insert):
    table_id = 'eng-reactor-287421.MSRB.msrb_trade_messages'
    errors = bq_client.insert_rows_json(table_id, rows_to_insert)  # Make an API request.
    if errors != []:
        send_error_email('msrb_api.py bq_client.insert_rows_json(table_id, rows_to_insert) Error', str(errors))


def get_latest_sequence_number():
    query = '''SELECT MAX(sequence_number) AS current_sequence_number
               FROM MSRB.msrb_trade_messages WHERE upload_date = CURRENT_DATE('US/Eastern')'''
    query_job = bq_client.query(query)
    query_job.result()
    destination = query_job.destination
    table = bq_client.get_table(destination)
    # Download rows:
    rows = bq_client.list_rows(table, max_results=1)
    for row in rows:
        return row['current_sequence_number']


def get_project_id():
    url = 'http://metadata.google.internal/computeMetadata/v1/project/project-id'
    headers = {'Metadata-Flavor': 'Google'}
    response = requests.request('GET', url, headers=headers)
    return response.text


def upload_to_storage(file_name, file_text):
    blob = bucket.blob(file_name)
    blob.upload_from_string(file_text)


def main(args):
    latest_sequence_number = get_latest_sequence_number()
    beginning_sequence_number = 0 if latest_sequence_number is None else latest_sequence_number + 1
    data = get_msrb_trade_messages(beginning_sequence_number)

    subscription_data = data.get('Subscription')
    if subscription_data is None:
        print('No subscription data found in the data')
        return 'SUCCESS'

    records = subscription_data.get('Records')
    if records is None:
        print('No "Records" found in the subscription data')
        return 'SUCCESS'

    timestamp = datetime.now(eastern).strftime('%Y-%m-%d_%H:%M:%S')
    upload_to_storage('real_time_msrb_file_%s.json' % timestamp, json.dumps(records))
    num_messages = subscription_data.get('RecordCount')
    print(f'Number of messages from MSRB: {num_messages}')

    if num_messages > 0:
        rows = [trade_message_to_full_db_record(records[idx]['Message']) for idx in range(len(records))]
        if rows:
            add_to_bigquery(rows)
            return 'SUCCESS'
        else:
            send_error_email('msrb_api.py trade_message_to_full_db_record Error', str(records))
            return 'ERROR'
    else:
        print(f'No new trade messages since sequence number: {beginning_sequence_number}')
        return 'SUCCESS'
