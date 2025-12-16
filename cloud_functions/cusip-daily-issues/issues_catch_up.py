'''Last updated by Developer on 2025-04-02.'''
import paramiko
import pandas as pd
from datetime import datetime, timedelta
import pytz
from google.cloud import secretmanager, bigquery
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import io

# Constants
PROJECT_ID = 'eng-reactor-287421'
DATASET = 'cusip_global_services'
TABLE_ID = 'cusip_issues'

COLUMN_HEADER = [
    'issuer_num', 'issue_num', 'issue_check', 'issue_description',
    'issue_additional_info', 'issue_status', 'issue_type_code',
    'dated_date', 'maturity_date', 'partial_maturity', 'coupon_rate',
    'currency_code', 'security_type_description', 'fisn', 'issue_group',
    'isin', 'where_traded', 'ticker_symbol', 'us_cfi_code', 'iso_cfi_code',
    'issue_entry_date', 'alternative_minimum_tax', 'bank_qualified',
    'callable', 'first_coupon_date', 'initial_public_offering',
    'payment_frequency_code', 'closing_date', 'dtc_eligible',
    'pre_refunded', 'refundable', 'remarketed', 'sinking_fund', 'taxable',
    'bond_form', 'enhancements', 'fund_distribution_policy',
    'fund_investment_policy', 'fund_type', 'guarantee', 'income_type',
    'insured_by', 'ownership_restriction', 'payment_status',
    'preferred_type', 'putable', 'rate_type', 'redemption',
    'source_document', 'sponsoring', 'voting_rights', 'warrant_assets',
    'warrant_status', 'warrant_type', 'underwriter', 'auditor',
    'paying_agent', 'tender_agent', 'transfer_agent', 'bond_counsel',
    'financial_advisor', 'municipal_sale_date', 'sale_type',
    'offering_amount', 'offering_amount_code', 'issue_transaction',
    'issue_last_update_date', 'obligator_name',
    'obligor_cusip_issuer_number', 'co_obligor_name',
    'co_obligor_cusip_issuer_number', 'government_stimulus_program',
    'reserved_6', 'reserved_7', 'reserved_8', 'reserved_9', 'reserved_10'
]

def access_secret_version(project_id, secret_id, version_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def get_last_data_entry(bq_client):
    query = f"SELECT MAX(upload_date) AS current_date FROM `{PROJECT_ID}.{DATASET}.{TABLE_ID}`"
    query_job = bq_client.query(query)
    results = query_job.result()
    for row in results:
        return row['current_date']
    return None

def process_data(file_content, upload_date):
    # Read the file into a DataFrame
    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), sep="|", names=COLUMN_HEADER, header=None)

    # Normalize column names (strip and lowercase)
    df.columns = df.columns.str.strip().str.lower()

    # Remove any rows where 'issuer_num' is '999999' and ensure 'issue_num' and 'issue_check' are not null
    df = df[df['issuer_num'] != '999999']
    df = df[~pd.isnull(df['issue_num']) & ~pd.isnull(df['issue_check'])]

    # Ensure 'issuer_num' and 'issue_check' are converted to string/int
    df['issuer_num'] = df['issuer_num'].astype(str)
    df['issue_check'] = df['issue_check'].astype(int)

    # Add upload_date column
    df["upload_date"] = pd.to_datetime(upload_date)

    # Convert specified columns to datetime
    date_columns = [
        'dated_date',
        'maturity_date',
        'issue_entry_date',
        'first_coupon_date',
        'closing_date',
        'municipal_sale_date',
        'issue_last_update_date',
        'upload_date'
    ]

    # Process date columns
    for column in date_columns:
        # Convert all values in the column to strings
        df[column] = df[column].astype(str)

        # Attempt to convert to datetime format, coercing errors to NaT
        df[column] = pd.to_datetime(df[column], format='%Y-%m-%d', exact=True, errors='coerce')

        # Convert datetime to just date and replace NaT with np.nan
        df[column] = df[column].dt.date.replace({pd.NaT: np.nan})

    # Replace any 'nan' strings with actual NaN values
    df = df.replace('nan', np.nan)

    # Ensure other columns are properly formatted as string or float
    float_columns = ['offering_amount', 'coupon_rate']
    for col in df.columns:
        if col not in float_columns and col not in date_columns:  # Exclude date columns from being converted to string or float
            if df[col].dtype == 'float64' or df[col].dtype == 'object':
                df[col] = df[col].astype(str)

    # Optionally: Ensure the float columns remain as floats
    df[float_columns] = df[float_columns].astype(float)

    df = df.replace('nan', np.nan)  # Ensure 'nan' strings are replaced with np.nan

    return df

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

def get_last_data_entry(bq_client):
    query = f"SELECT MAX(upload_date) AS current_date FROM `{PROJECT_ID}.{DATASET}.{TABLE_ID}`"
    query_job = bq_client.query(query)
    results = query_job.result()
    for row in results:
        return row['current_date']
    return None

def get_table_schema():
    return [
        bigquery.SchemaField("issuer_num", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("issue_num", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("issue_check", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("issue_description", "STRING"),
        bigquery.SchemaField("issue_additional_info", "STRING"),
        bigquery.SchemaField("issue_status", "STRING"),
        bigquery.SchemaField("issue_type_code", "STRING"),
        bigquery.SchemaField("dated_date", "DATE"),
        bigquery.SchemaField("maturity_date", "DATE"),
        bigquery.SchemaField("partial_maturity", "STRING"),
        bigquery.SchemaField("coupon_rate", "FLOAT64"),
        bigquery.SchemaField("currency_code", "STRING"),
        bigquery.SchemaField("security_type_description", "STRING"),
        bigquery.SchemaField("fisn", "STRING"),
        bigquery.SchemaField("issue_group", "STRING"),
        bigquery.SchemaField("isin", "STRING"),
        bigquery.SchemaField("where_traded", "STRING"),
        bigquery.SchemaField("ticker_symbol", "STRING"),
        bigquery.SchemaField("us_cfi_code", "STRING"),
        bigquery.SchemaField("iso_cfi_code", "STRING"),
        bigquery.SchemaField("issue_entry_date", "DATE"),
        bigquery.SchemaField("alternative_minimum_tax", "STRING"),
        bigquery.SchemaField("bank_qualified", "STRING"),
        bigquery.SchemaField("callable", "STRING"),
        bigquery.SchemaField("first_coupon_date", "DATE"),
        bigquery.SchemaField("initial_public_offering", "STRING"),
        bigquery.SchemaField("payment_frequency_code", "STRING"),
        bigquery.SchemaField("closing_date", "DATE"),
        bigquery.SchemaField("dtc_eligible", "STRING"),
        bigquery.SchemaField("pre_refunded", "STRING"),
        bigquery.SchemaField("refundable", "STRING"),
        bigquery.SchemaField("remarketed", "STRING"),
        bigquery.SchemaField("sinking_fund", "STRING"),
        bigquery.SchemaField("taxable", "STRING"),
        bigquery.SchemaField("bond_form", "STRING"),
        bigquery.SchemaField("enhancements", "STRING"),
        bigquery.SchemaField("fund_distribution_policy", "STRING"),
        bigquery.SchemaField("fund_investment_policy", "STRING"),
        bigquery.SchemaField("fund_type", "STRING"),
        bigquery.SchemaField("guarantee", "STRING"),
        bigquery.SchemaField("income_type", "STRING"),
        bigquery.SchemaField("insured_by", "STRING"),
        bigquery.SchemaField("ownership_restriction", "STRING"),
        bigquery.SchemaField("payment_status", "STRING"),
        bigquery.SchemaField("preferred_type", "STRING"),
        bigquery.SchemaField("putable", "STRING"),
        bigquery.SchemaField("rate_type", "STRING"),
        bigquery.SchemaField("redemption", "STRING"),
        bigquery.SchemaField("source_document", "STRING"),
        bigquery.SchemaField("sponsoring", "STRING"),
        bigquery.SchemaField("voting_rights", "STRING"),
        bigquery.SchemaField("warrant_assets", "STRING"),
        bigquery.SchemaField("warrant_status", "STRING"),
        bigquery.SchemaField("warrant_type", "STRING"),
        bigquery.SchemaField("underwriter", "STRING"),
        bigquery.SchemaField("auditor", "STRING"),
        bigquery.SchemaField("paying_agent", "STRING"),
        bigquery.SchemaField("tender_agent", "STRING"),
        bigquery.SchemaField("transfer_agent", "STRING"),
        bigquery.SchemaField("bond_counsel", "STRING"),
        bigquery.SchemaField("financial_advisor", "STRING"),
        bigquery.SchemaField("municipal_sale_date", "DATE"),
        bigquery.SchemaField("sale_type", "STRING"),
        bigquery.SchemaField("offering_amount", "FLOAT64"),
        bigquery.SchemaField("offering_amount_code", "STRING"),
        bigquery.SchemaField("issue_transaction", "STRING"),
        bigquery.SchemaField("issue_last_update_date", "DATE"),
        bigquery.SchemaField("obligator_name", "STRING"),
        bigquery.SchemaField("obligor_cusip_issuer_number", "STRING"),
        bigquery.SchemaField("co_obligor_name", "STRING"),
        bigquery.SchemaField("co_obligor_cusip_issuer_number", "STRING"),
        bigquery.SchemaField("government_stimulus_program", "STRING"),
        bigquery.SchemaField("upload_date", "DATE"),
        bigquery.SchemaField("reserved_6", "STRING"),
        bigquery.SchemaField("reserved_7", "STRING"),
        bigquery.SchemaField("reserved_8", "STRING"),
        bigquery.SchemaField("reserved_9", "STRING"),
        bigquery.SchemaField("reserved_10", "STRING"),
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

def cusip_daily_issues(args):
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
    current_date = my_timezone.localize(datetime.now()).date()

    # Look back window of 3 days
    for i in range(4):
        new_date = current_date - timedelta(days=i)
        new_date_string = new_date.strftime('%Y-%m-%d')
        file_name = f"ACUD{new_date.strftime('%m-%d')}E.PIP"

        try:
            with sftp.open(file_name) as file:
                file_content = file.read()
            data = process_data(file_content, new_date_string)

            if data.empty:
                send_email("CUSIP Daily Issues: Empty Data", f"<html><body><p>The processed data is empty for {new_date_string}. Please check the input file.</p></body></html>")
                continue

            last_entry_date = get_last_data_entry(bq_client)
            print(f"Last entry date: {last_entry_date}")

            if last_entry_date is None or new_date > last_entry_date:
                print(f"Uploading new data for date: {new_date}")
                result = load_data_to_bigquery(bq_client, data)
                print(f"Upload result: {result}")

                if result == "SUCCESS":
                    message = f"New {len(data)} issues data was uploaded to {PROJECT_ID}.{DATASET}.{TABLE_ID} for {new_date_string}."
                    send_email("CUSIP Daily Issues: Success", f"<html><body><p>{message}</p></body></html>")
                else:
                    message = f"Data upload failed for {new_date_string}."
                    send_email("CUSIP Daily Issues: Failure", f"<html><body><p>{message}</p></body></html>")
            else:
                print(f"No new data to upload for {new_date_string}. Current data is up to date.")

        except IOError as e:
            print(f"File not found for date: {new_date}, {str(e)}")
            continue  # If file is not found, continue to the next day

    sftp.close()
    transport.close()

    return "Process completed successfully."