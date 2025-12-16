'''Last updated by Developer on 2025-04-02.'''
import paramiko
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logging as python_logging

from google.cloud import secretmanager, bigquery, logging


logging_client = logging.Client()
logging_client.setup_logging()

# Constants
PROJECT_ID = 'eng-reactor-287421'
DATASET = 'cusip_global_services'
TABLE_ID = 'cusip_issuer'
COLUMN_HEADER = [
    'issuer_num', 'issuer_check', 'issuer_name', 'issuer_adl', 'issuer_type',
    'issuer_status', 'domicile', 'state_cd', 'cabre_id', 'cabre_status',
    'lei_gmei', 'legal_entity_name', 'previous_name', 'issuer_entry_date',
    'cp_institution_type_desc', 'issuer_transaction', 'issuer_update_date',
    'reserved_1', 'reserved_2', 'reserved_3', 'reserved_4', 'reserved_5',
    'reserved_6', 'reserved_7', 'reserved_8', 'reserved_9', 'reserved_10'
]


def access_secret_version(project_id, secret_id, version_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def process_data(file_path, upload_date):
    try:
        df = pd.read_csv(file_path, sep="|", names=COLUMN_HEADER, header=None)
        df["upload_date"] = pd.to_datetime(upload_date)

        df = df[~pd.isna(df["issuer_check"])]
        df["issuer_check"] = df["issuer_check"].astype(int)

        date_columns = ["issuer_entry_date", "issuer_update_date"]
        for column in date_columns:
            df[column] = pd.to_datetime(df[column], format="%Y-%m-%d", exact=True, errors='coerce').dt.date

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')

        return df

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise


def get_bigquery_client():
    return bigquery.Client(project=PROJECT_ID)


def get_last_data_entry(bq_client):
    query = f"SELECT MAX(upload_date) AS current_date FROM `{PROJECT_ID}.{DATASET}.{TABLE_ID}`"
    query_job = bq_client.query(query)
    results = query_job.result()
    for row in results:
        return row['current_date']
    return None


def load_data_to_bigquery(bq_client, data):
    job_config = bigquery.LoadJobConfig(
        schema=get_table_schema(),
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )
    table_ref = bq_client.dataset(DATASET).table(TABLE_ID)

    job = bq_client.load_table_from_dataframe(data, table_ref, job_config=job_config)
    job.result()  # Wait for the job to complete
    print(f"Loaded {job.output_rows} rows to {PROJECT_ID}.{DATASET}.{TABLE_ID}")
    return "SUCCESS"


def get_table_schema():
    return [
        bigquery.SchemaField("issuer_num", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("issuer_check", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("issuer_name", "STRING"),
        bigquery.SchemaField("issuer_adl", "STRING"),
        bigquery.SchemaField("issuer_type", "STRING"),
        bigquery.SchemaField("issuer_status", "STRING"),
        bigquery.SchemaField("domicile", "STRING"),
        bigquery.SchemaField("state_cd", "STRING"),
        bigquery.SchemaField("cabre_id", "STRING"),
        bigquery.SchemaField("cabre_status", "STRING"),
        bigquery.SchemaField("lei_gmei", "STRING"),
        bigquery.SchemaField("legal_entity_name", "STRING"),
        bigquery.SchemaField("previous_name", "STRING"),
        bigquery.SchemaField("issuer_entry_date", "DATE"),
        bigquery.SchemaField("cp_institution_type_desc", "STRING"),
        bigquery.SchemaField("issuer_transaction", "STRING"),
        bigquery.SchemaField("issuer_update_date", "DATE"),
        bigquery.SchemaField("reserved_1", "STRING"),
        bigquery.SchemaField("reserved_2", "STRING"),
        bigquery.SchemaField("reserved_3", "STRING"),
        bigquery.SchemaField("reserved_4", "STRING"),
        bigquery.SchemaField("reserved_5", "STRING"),
        bigquery.SchemaField("reserved_6", "STRING"),
        bigquery.SchemaField("reserved_7", "STRING"),
        bigquery.SchemaField("reserved_8", "STRING"),
        bigquery.SchemaField("reserved_9", "STRING"),
        bigquery.SchemaField("reserved_10", "STRING"),
        bigquery.SchemaField("upload_date", "DATE"),
    ]


def cusip_daily_issuer(args):
    EASTERN = pytz.timezone('America/New_York')
    host = 'sftp.cusip.com'
    port = 22
    user = access_secret_version(PROJECT_ID, 'cusip_username', 'latest')
    pwd = access_secret_version(PROJECT_ID, 'cusip_password', 'latest')

    transport = paramiko.Transport((host, port))
    transport.connect(username=user, password=pwd)
    sftp = paramiko.SFTPClient.from_transport(transport)

    if sftp.getcwd() != '/Inbox':
        sftp.chdir('/Inbox')

    bq_client = get_bigquery_client()
    current_date = datetime.now(EASTERN).date()

    # Look back window of 3 days
    for i in range(4):
        new_date = current_date - timedelta(days=i)
        if new_date.weekday() > 4: continue    # do not look for files on weekends; 0 = Mon, 1 = Tue, ..., 4 = Fri, 5 = Sat, 6 = Sun
        new_date_string = new_date.strftime('%Y-%m-%d')
        file_name = f"ACUD{new_date.strftime('%m-%d')}R.PIP"

        try:
            file_path = sftp.open(file_name)
            data = process_data(file_path, new_date_string)

            if data.empty:
                python_logging.warning("CUSIP Daily Issuer: Empty Data", f'The processed data is empty for {new_date_string}. Please check the input file.')    # f"<html><body><p>The processed data is empty for {new_date_string}. Please check the input file.</p></body></html>")
                continue

            last_entry_date = get_last_data_entry(bq_client)
            print(f"Last entry date: {last_entry_date}")

            if last_entry_date is None or new_date > last_entry_date:
                print(f"Uploading new data for date: {new_date}")
                result = load_data_to_bigquery(bq_client, data)
                print(f"Upload result: {result}")

                if result == "SUCCESS":
                    message = f"New {len(data)} issuers data was uploaded to {PROJECT_ID}.{DATASET}.{TABLE_ID} for {new_date_string}."
                    print("CUSIP Daily Issuer: Success", message)    # f"<html><body><p>{message}</p></body></html>")
                else:
                    message = f"Data upload failed for {new_date_string}."
                    python_logging.warning("CUSIP Daily Issuer: Failure", message)    # f"<html><body><p>{message}</p></body></html>")
            else:
                print(f"No new data to upload for {new_date_string}. Current data is up to date.")
                break    # no need to search for files further back in time when the data is up to date

        except IOError as e:
            python_logging.warning(f"File not found for date: {new_date}, {str(e)}")
            continue    # if file is not found, continue to the next day

    return "Process completed successfully."
