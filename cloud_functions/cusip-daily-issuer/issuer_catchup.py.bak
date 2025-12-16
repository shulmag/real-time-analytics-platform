'''Last updated by Developer on 2025-04-02.'''
import io
import paramiko
import pandas as pd
from datetime import datetime
import pytz

from google.cloud import secretmanager, bigquery

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# # Set your Google Cloud credentials
# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'

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

def process_data(file_content, upload_date):
    try:
        # Read the file into a DataFrame
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), sep="|", names=COLUMN_HEADER, header=None)
        
        # Add upload_date column
        df["upload_date"] = pd.to_datetime(upload_date)

        # Convert issuer_check to integer after filtering out nulls
        df = df[~pd.isna(df["issuer_check"])]
        df["issuer_check"] = df["issuer_check"].astype(int)

        # Convert specified columns to datetime
        date_columns = ["issuer_entry_date", "issuer_update_date"]
        for column in date_columns:
            df[column] = pd.to_datetime(df[column], format="%Y-%m-%d", exact=True, errors='coerce').dt.date

        # Handle potential NaN values in other columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')  # Fill string NaN with empty string

        return df

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise

def get_bigquery_client():
    return bigquery.Client(project=PROJECT_ID)

def send_email(subject, message):
    sender_email = access_secret_version('notifications_username')
    recipients = ['ficc-eng@ficc.ai']
    password = access_secret_version('notifications_password')

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = ', '.join(recipients)

    body = MIMEText(message, 'html')
    msg.attach(body)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, recipients, msg.as_string())

def get_existing_upload_dates(bq_client):
    query = f"""
    SELECT upload_date
    FROM `{PROJECT_ID}.{DATASET}.{TABLE_ID}`
    GROUP BY upload_date
    ORDER BY upload_date DESC
    LIMIT 50
    """
    query_job = bq_client.query(query)
    results = query_job.result()
    return [row['upload_date'] for row in results]

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

def catch_up_cusip_daily_issuer(event, context):
    my_timezone = pytz.timezone('America/New_York')
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
    
    # Get existing upload dates
    existing_dates = set(get_existing_upload_dates(bq_client))
    
    # Get list of PIP files on SFTP


    # Get list of PIP files on SFTP
    sftp_files = sftp.listdir()
    pip_files = [f for f in sftp_files if f.startswith('ACUD') and f.endswith('R.PIP')]
    
    for file_name in pip_files:
        # Extract date from file name
        file_date_str = file_name[4:9]  # Extract "MM-DD" part
        current_year = datetime.now().year
        file_date = datetime.strptime(f"{current_year}-{file_date_str}", "%Y-%m-%d").date()
        
        if file_date not in existing_dates:
            try:
                with sftp.open(file_name) as file:
                    file_content = file.read()
                data = process_data(file_content, file_date)

                if data.empty:
                    send_email("CUSIP Daily Issuer Catch-up: Empty Data", f"<html><body><p>The processed data is empty for {file_date}. Please check the input file.</p></body></html>")
                    continue

                print(f"Uploading data for date: {file_date}")
                result = load_data_to_bigquery(bq_client, data)
                print(f"Upload result: {result}")

                if result == "SUCCESS":
                    message = f"Catch-up: {len(data)} issuers data was uploaded to {PROJECT_ID}.{DATASET}.{TABLE_ID} for {file_date}."
                    send_email("CUSIP Daily Issuer Catch-up: Success", f"<html><body><p>{message}</p></body></html>")
                else:
                    message = f"Catch-up: Data upload failed for {file_date}."
                    send_email("CUSIP Daily Issuer Catch-up: Failure", f"<html><body><p>{message}</p></body></html>")

            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")
                send_email("CUSIP Daily Issuer Catch-up: Error", f"<html><body><p>Error processing file {file_name}: {str(e)}</p></body></html>")
                continue

    sftp.close()
    transport.close()

    return "Catch-up process completed successfully."

if __name__ == "__main__":
    catch_up_cusip_daily_issuer(None, None)