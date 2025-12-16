import os
credential_path = "/Users/Gil/git/ficc/data_loader/Cusip Global Service Importer-2fdcdfc4edba.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= credential_path

import re
import time
import json
import requests
from datetime import datetime,timedelta
from pytz import timezone
from google.cloud import bigquery,storage
from send_email import send_error_email
from google.cloud import secretmanager

# Construct a BigQuery client object.
bq_client = bigquery.Client()

# use ET time zone: 
eastern = timezone('US/Eastern')

# Construct a Storage client object.
client = storage.Client()
bucket = client.get_bucket('msrb_trade_history')

# wrapper for google cloud platform secret manager, to get usernames and passwords
def access_secret_version(project_id, secret_id, version_id):
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()
    # Build the resource name of the secret version.
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    # Access the secret version.
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    return(payload)

# fetch a replay file for daily trades from MSRB
def get_msrb_replay_file(date):
    #  rtrsprodsubscription.msrb.org - the URL's for the production site.
    # https://rtrsbetasubscription.msrb.org - BETA
    url = 'https://rtrsprodsubscription.msrb.org/rtrssubscriptionfilewebservice/api/Subscription.GetFile?filetype=RTRSReplay&dt='\
          + datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
    headers = {
        'credentials': access_secret_version('eng-reactor-287421','msrb_username','latest') + ','\
                       + access_secret_version('eng-reactor-287421','msrb_password','latest'),
        'accept': 'application/json'
        }
    response = requests.request("GET", url, headers=headers)
    return(response.text)

# parse a single trade message into BQ ready data
def trade_message_to_full_db_record(trade_message): 
    names = ['message_type',	'sequence_number','delete',	'rtrs_control_number',	'trade_type',	'transaction_type',	'cusip',\
        	 'security_description',	'dated_date',	'coupon',	'maturity_date',	'when_issued',	'assumed_settlement_date',\
             'trade_date',	'time_of_trade',	'settlement_date',	'par_traded',	'dollar_price',	'yield',	'brokers_broker',\
             'is_weighted_average_price',	'is_lop_or_takedown',	'publish_date',	'publish_time',	'version',	'unable_to_verify_dollar_price',\
             'is_alternative_trading_system',	'is_non_transaction_based_compensation',	'is_trade_with_a_par_amount_over_5MM']
    parsed_messages = trade_message.split(',')
    temp_dict = dict([s.split('=',1) for s in parsed_messages])
    dict_record = {int(key):temp_dict[key] for key in temp_dict.keys()}
    dict_record_with_headers = ({names[i]:dict_record[i+1] if i+1 in dict_record else None for i in range(len(names))})
    del dict_record_with_headers['delete']
    return(data_type_casting(dict_record_with_headers))

# cast each MSRB string into the BQ schema data type
def data_type_casting(dict_trade_message):
    # The 'except ValueError:' place null value for every value retuened that does not fit the data type - we lose data that way, but perhaps not useful data. Let's discuss. 
    dict_data_type = {'upload_date':'date',	'message_type':'string',	'sequence_number':'integer',	'rtrs_control_number':'integer',\
        	'trade_type':'string',	'transaction_type':'string',	'cusip':'string',	'security_description':'string',	'dated_date':'date',\
            'coupon':'numeric',	'maturity_date':'date',	'when_issued':'boolean',	'assumed_settlement_date':'date',	'trade_date':'date',\
            'time_of_trade':'time',	'settlement_date':'date',	'par_traded':'numeric',	'dollar_price':'float',	'yield':'float',\
            'brokers_broker':'string',	'is_weighted_average_price':'boolean',	'is_lop_or_takedown':'boolean',	'publish_date':'date',\
            'publish_time':'time',	'version':'numeric',	'unable_to_verify_dollar_price':'boolean',	'is_alternative_trading_system':'boolean',\
            'is_non_transaction_based_compensation':'boolean',	'is_trade_with_a_par_amount_over_5MM':'boolean'}
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
                if v == 'MM':
                    dict_trade_message[k] = None
                    dict_trade_message['is_trade_with_a_par_amount_over_5MM'] = True  
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

# push the data to BQ
def add_to_bigquery(rows_to_insert):
    table_id = "eng-reactor-287421.MSRB.msrb_daily_replay_files"
    #test DB: 
    #table_id = "eng-reactor-287421.MSRB.msrb_daily_replay_files_gil_test"
    
    #Create schema
    trade_messages_schema = [bigquery.SchemaField("upload_date","date"),	bigquery.SchemaField("message_type","string"),	bigquery.SchemaField("sequence_number","integer"),	bigquery.SchemaField("rtrs_control_number","integer"),	bigquery.SchemaField("trade_type","string"),	bigquery.SchemaField("transaction_type","string"),	bigquery.SchemaField("cusip","string"),	bigquery.SchemaField("security_description","string"),	bigquery.SchemaField("dated_date","date"),	bigquery.SchemaField("coupon","numeric"),	bigquery.SchemaField("maturity_date","date"),	bigquery.SchemaField("when_issued","boolean"),	bigquery.SchemaField("assumed_settlement_date","date"),	bigquery.SchemaField("trade_date","date"),	bigquery.SchemaField("time_of_trade","time"),	bigquery.SchemaField("settlement_date","date"),	bigquery.SchemaField("par_traded","numeric"),	bigquery.SchemaField("dollar_price","float"),	bigquery.SchemaField("yield","float"),	bigquery.SchemaField("brokers_broker","string"),	bigquery.SchemaField("is_weighted_average_price","boolean"),	bigquery.SchemaField("is_lop_or_takedown","boolean"),	bigquery.SchemaField("publish_date","date"),	bigquery.SchemaField("publish_time","time"),	bigquery.SchemaField("version","numeric"),	bigquery.SchemaField("unable_to_verify_dollar_price","boolean"),	bigquery.SchemaField("is_alternative_trading_system","boolean"),	bigquery.SchemaField("is_non_transaction_based_compensation","boolean"),	bigquery.SchemaField("is_trade_with_a_par_amount_over_5MM","boolean")]
        
    job_config = bigquery.LoadJobConfig(
        schema=trade_messages_schema,
        writeDisposition="WRITE_APPEND",
        maxBadRecords=5,
        allowJaggedRows=True,
        ignore_unknown_values=True
    )

    load_job = bq_client.load_table_from_json(rows_to_insert, table_id, job_config=job_config)# Make an API request.
    from google.api_core.exceptions import BadRequest

    try:
        result = load_job.result()  # Waits for the job to complete.
    except BadRequest as ex:
        send_error_email("failed to update BQ: msrb_daily_replay_files",str(ex))

# get the day before the day param you send: 
def previous_day(day):
    previous = day - timedelta(1)
    return(previous)

# get the last date for which we have data
def get_last_data_entry():
    query = """
        SELECT MAX(publish_date) AS current_date
        FROM MSRB.msrb_daily_replay_files
    """
    query_job = bq_client.query(query)
    query_job.result()
    destination = query_job.destination
    table = bq_client.get_table(destination)
    # Download rows:
    rows = bq_client.list_rows(table, max_results=1)
    for row in rows:
        return(row["current_date"])
        break

# upoad the MSRB data to GCS
def upload_to_storage(file_name, file_text):
    blob = bucket.blob(file_name)
    blob.upload_from_string(file_text)

def main(args):
    time.sleep(7)
    data = ''
    wanted_date = get_last_data_entry() + timedelta(1) #datetime.strptime('2021-03-11', '%Y-%m-%d').date()
    while wanted_date < datetime.date(datetime.now(eastern)):
        data = get_msrb_replay_file(wanted_date.strftime('%Y-%m-%d'))
        upload_to_storage('2021/daily_replay_file_{}.txt'.format(wanted_date) ,data)
        trade_messages = data.splitlines()
        rows = [trade_message_to_full_db_record(trade_messages[idx]) for idx in range(len(trade_messages))]
        if rows != []:
            add_to_bigquery(rows)
        wanted_date += timedelta(1)
        time.sleep(6)

main("g")