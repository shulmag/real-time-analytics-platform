#import os
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="GOOGLE_APPLICATION_CREDENTIALS.json"

import re
import time
import json
import requests
from datetime import datetime
from pytz import timezone
from google.cloud import bigquery,storage
from send_email import send_error_email
from google.cloud import secretmanager
# Construct a BigQuery client object.
bq_client = bigquery.Client()
# ET time zone: 
eastern = timezone('US/Eastern')

client = storage.Client()
bucket = client.get_bucket('msrb_intraday_real_time_trade_files')

def access_secret_version(project_id, secret_id, version_id):
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()
    # Build the resource name of the secret version.
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    # Access the secret version.
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    return(payload)

#Port: 443
def get_msrb_trade_messages(beginSequence):
    #  rtrsprodsubscription.msrb.org - the URL's for the production site.
    # https://rtrsbetasubscription.msrb.org - BETA
    url = 'https://rtrsprodsubscription.msrb.org/rtrssubscriptionwebservice/api/Subscription.GetNext?beginSequence=' + str(beginSequence)
    #print(access_secret_version('eng-reactor-287421','msrb_username','latest') + ',' + access_secret_version('eng-reactor-287421','msrb_password','latest'))
    headers = {
        'credentials': access_secret_version('eng-reactor-287421','msrb_username','latest') + ',' + access_secret_version('eng-reactor-287421','msrb_password','latest'),
        #'credentials': 'RTDSFICCAI,BinduReddy0317&',
        'accept': 'application/json'
        }
    #print(get_project_id())
    #if get_project_id() == 'eng-reactor-287421':
        #time.sleep(11)
    response = requests.request("GET", url, headers=headers)
    return(response.text)

def trade_message_to_full_db_record(trade_message): 
    names = ['message_type',	'sequence_number','delete',	'rtrs_control_number',	'trade_type',	'transaction_type',	'cusip',	'security_description',	'dated_date',	'coupon',	'maturity_date',	'when_issued',	'assumed_settlement_date',	'trade_date',	'time_of_trade',	'settlement_date',	'par_traded',	'dollar_price',	'yield',	'brokers_broker',	'is_weighted_average_price',	'is_lop_or_takedown',	'publish_date',	'publish_time',	'version',	'unable_to_verify_dollar_price',	'is_alternative_trading_system',	'is_non_transaction_based_compensation',	'is_trade_with_a_par_amount_over_5MM']
    parsed_messages = trade_message.split(',')
    temp_dict = dict([s.split('=',1) for s in parsed_messages])
    dict_record = {int(key):temp_dict[key] for key in temp_dict.keys()}
    dict_record_with_headers = ({names[i]:dict_record[i+1] if i+1 in dict_record else None for i in range(len(names))})
    del dict_record_with_headers['delete']
    return(data_type_casting(dict_record_with_headers))

def data_type_casting(dict_trade_message):
    # The 'except ValueError:' place null value for every value retuened that does not fit the data type - we lose data that way, but perhaps not useful data. Let's discuss. 
    dict_data_type = {'upload_date':'date',	'message_type':'string',	'sequence_number':'integer',	'rtrs_control_number':'integer',	'trade_type':'string',	'transaction_type':'string',	'cusip':'string',	'security_description':'string',	'dated_date':'date',	'coupon':'numeric',	'maturity_date':'date',	'when_issued':'boolean',	'assumed_settlement_date':'date',	'trade_date':'date',	'time_of_trade':'time',	'settlement_date':'date',	'par_traded':'numeric',	'dollar_price':'float',	'yield':'float',	'brokers_broker':'string',	'is_weighted_average_price':'boolean',	'is_lop_or_takedown':'boolean',	'publish_date':'date',	'publish_time':'time',	'version':'numeric',	'unable_to_verify_dollar_price':'boolean',	'is_alternative_trading_system':'boolean',	'is_non_transaction_based_compensation':'boolean',	'is_trade_with_a_par_amount_over_5MM':'boolean'}
    for k,v in dict_trade_message.items():
        try:
            if(dict_data_type[k]=='date'):
                if v != None:
                    if len(v) == 8:
                        dict_trade_message[k] = datetime.strptime(v, '%Y%m%d').strftime('%Y-%m-%d')
                    elif len(v)==6:
                        dict_trade_message[k] = datetime.strptime(v, '%m%d%y').strftime('%Y-%m-%d')
                    else:
                        dict_trade_message[k] = None
            if(dict_data_type[k] == 'numeric' or dict_data_type[k] == 'float'):
                if v == 'MM+':
                    dict_trade_message[k] = None
                    dict_trade_message['is_trade_with_a_par_amount_over_5MM'] = "Y"  
                elif v != '' and v != None:
                    dict_trade_message[k] = float(v)
                else: dict_trade_message[k] = None
            if(dict_data_type[k]=='string'):
                dict_trade_message[k] = v   
            if(dict_data_type[k]=='integer' and v != None):
                dict_trade_message[k] = int(v) 
            if(dict_data_type[k]=='boolean'):
                dict_trade_message[k] = (v == 'Y')                   
            elif(dict_data_type[k]=='time'):
                if v != None:
                    dict_trade_message[k] = datetime.strftime(datetime.strptime(v,'%H%M%S'),'%H:%M:%S')
        except Exception:
            dict_trade_message[k] = None

    dict_trade_message['upload_date'] = datetime.now(eastern).strftime('%Y-%m-%d')
    return(dict_trade_message)

def add_to_bigquery(rows_to_insert):
    table_id = "eng-reactor-287421.MSRB.msrb_trade_messages"
    errors = bq_client.insert_rows_json(table_id, rows_to_insert)  # Make an API request.
    if errors != []:
        send_error_email('msrb_api.py bq_client.insert_rows_json(table_id, rows_to_insert) Error',str(errors))

def get_last_Sequence():
    query = """
        SELECT MAX(sequence_number) AS current_sequence_number
        FROM MSRB.msrb_trade_messages WHERE upload_date = CURRENT_DATE("US/Eastern")
    """
    query_job = bq_client.query(query)
    query_job.result()
    destination = query_job.destination
    table = bq_client.get_table(destination)
    # Download rows:
    rows = bq_client.list_rows(table, max_results=1)
    for row in rows:
        return(row["current_sequence_number"])
        break

def get_project_id():
    url = 'http://metadata.google.internal/computeMetadata/v1/project/project-id'
    headers = {
        'Metadata-Flavor':'Google'
        }
    response = requests.request("GET", url, headers=headers)
    return(response.text)

def upload_to_storage(file_name, file_text):
    blob = bucket.blob(file_name)
    blob.upload_from_string(file_text)

def main(args):
    sequence = get_last_Sequence()
    beginSequence = 0 if sequence == None else sequence + 1
    json_data = get_msrb_trade_messages(beginSequence)
    data = json.loads(json_data)
    trade_messages = data.get('Subscription').get('Records')
    timestamp = datetime.now(eastern).strftime('%Y-%m-%d_%H:%M:%S')
    upload_to_storage('real_time_msrb_file_%s.json' % timestamp,json.dumps(trade_messages))
    if data.get('Subscription').get('RecordCount') > 0:
        rows = [trade_message_to_full_db_record(trade_messages[idx]['Message']) for idx in range(len(trade_messages))]
        if rows != []:
            add_to_bigquery(rows)
        else:
            send_error_email('msrb_api.py trade_message_to_full_db_record Error',str(trade_messages))