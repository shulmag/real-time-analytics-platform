'''
Description: Reads manually uploaded json files from Cloud Storage and take the maturities (the number of years to maturity) and yield to worst values for each index. The bucket is here: https://console.cloud.google.com/storage/browser/muni_index_jsons.  The json files are to be added by hand and once this procedure is complete this cloud function should run. 
             See [this Notion page](https://www.notion.so/Yield-Curve-0e9d3fb1a49a4789826083361257a962?pvs=4#189eb87466c280d9ad01dc717ba0c6ae) for more details on related cloud functions and procedures.
'''
import os
import requests
import traceback    # used to print error stack trace
import pandas as pd
import smtplib
import json

from datetime import datetime
from pytz import timezone
from google.cloud import bigquery, secretmanager,storage
from email.mime.text import MIMEText
from auxiliary_functions import function_timer, today_is_a_holiday,go_to_previous_weekday_if_weekend, EASTERN



# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'


RUN_LOCALLY = False
TESTING = False   # set this to `True` when testing the cloud function
PROJECT_ID = 'eng-reactor-287421'

# defining IDs using dictionaries so we don't need to rely on indexing for lists, etc, to remove any ambiguity; the IDs are unique identifiers for each S&P index to request their data from the S&P API; these values were scraped directly from S&P
INDEX_MATURITIES_TABLE_NAME_TO_INDEX_ID = {
    'sp_15plus_year_national_amt_free_index': 92346704, 
    'sp_12_22_year_national_amt_free_index': 946546, 
    'sp_7_12_year_national_amt_free_index': 946545, 
    'sp_high_quality_index': 10001818, 
    'sp_high_quality_intermediate_managed_amt_free_index': 92404510, 
    'sp_high_quality_short_intermediate_index': 10001820, 
    'sp_high_quality_short_index': 10001819, 
    'sp_long_term_national_amt_free_municipal_bond_index_yield': 946547, 
}
INDEX_MATURITIES_INDEX_IDS = list(INDEX_MATURITIES_TABLE_NAME_TO_INDEX_ID.values())

# defining IDs using dictionaries so we don't need to rely on indexing for lists, etc, to remove any ambiguity; the IDs are unique identifiers for each S&P index to request their data from the S&P API; these values were scraped directly from S&P
INDEX_YIELDS_TABLE_NAME_TO_INDEX_ID = {
    'sp_15plus_year_national_amt_free_index': 92346704,
    'sp_12_22_year_national_amt_free_index': 946546,
    'sp_7_12_year_national_amt_free_municipal_bond_index_yield': 946545,
    'sp_high_quality_intermediate_managed_amt_free_municipal_bond_index_yield': 92404510,
    'sp_high_quality_short_intermediate_municipal_bond_index_yield': 10001820,
    'sp_high_quality_short_municipal_bond_index_yield': 10001819,
    'sp_muni_high_quality_index_yield': 10001818,
    'sp_long_term_national_amt_free_municipal_bond_index_yield': 946547,
}
INDEX_YIELDS_INDEX_IDS = list(INDEX_YIELDS_TABLE_NAME_TO_INDEX_ID.values())
assert sorted(INDEX_MATURITIES_INDEX_IDS) == sorted(INDEX_YIELDS_INDEX_IDS), f'INDEX_MATURITIES_INDEX_IDS={INDEX_MATURITIES_INDEX_IDS} should equal INDEX_YIELDS_INDEX_IDS={INDEX_YIELDS_INDEX_IDS}'
INDEX_IDS = INDEX_MATURITIES_INDEX_IDS

INDEX_ID_TO_INDEX_MATURITIES_TABLE_NAME = {index_id: table_name for table_name, index_id in INDEX_MATURITIES_TABLE_NAME_TO_INDEX_ID.items()}
INDEX_ID_TO_INDEX_YIELDS_TABLE_NAME = {index_id: table_name for table_name, index_id in INDEX_YIELDS_TABLE_NAME_TO_INDEX_ID.items()}

def access_secret_version(secret_id, project_id=PROJECT_ID, version_id='latest'):
    client = secretmanager.SecretManagerServiceClient()     # create the Secret Manager client
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'    # create the resource name of the secret version
    response = client.access_secret_version(request={'name': name})    # access the secret version
    payload = response.payload.data.decode('UTF-8')
    return payload


def send_error_email(subject, error_message):
    sender_email = access_secret_version('notifications_username')
    password = access_secret_version('notifications_password')
    receiver_emails = ['eng@ficc.ai','gil@ficc.ai', 'ficc-eng@ficc.ai']  

    msg = MIMEText(error_message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = ', '.join(receiver_emails)

    smtp_server = 'smtp.gmail.com'
    port = 587

    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_emails, msg.as_string())
        except Exception as e:
            print(e)
        finally:
            server.quit()


@function_timer
def gcs_muni_jsons_to_dataframe(bucket_name: str, folder_path: str) -> pd.DataFrame:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=folder_path + '/'))

    # Step 1: Pre-check for expected JSON files
    found_index_ids = {
        int(blob.name.split('/')[-1].replace('.json', ''))
        for blob in blobs
        if blob.name.endswith('.json')
    }

    missing = [index_id for index_id in INDEX_MATURITIES_INDEX_IDS if index_id not in found_index_ids]
    assert not missing, f"JSON for index_id(s) {missing} not in Cloud Storage for date/folder_path {folder_path}"
    print(f"All indices are present for {folder_path}")

    records = []

    for blob in blobs:
        if not blob.name.endswith('.json'):
            continue

        index_id = blob.name.split('/')[-1].replace('.json', '')
        raw_data = blob.download_as_string()
        data = json.loads(raw_data)
        index_data = data["indicesCollection"].get(index_id, {})

        # Extract latest yield to worst
        index_levels = index_data.get("indexLevelsHolder", {}).get("indexLevels", [])
        df_levels = pd.DataFrame(index_levels)

        if not df_levels.empty:
            df_levels["effectiveDate"] = pd.to_datetime(df_levels["effectiveDate"], unit="ms")
            latest = df_levels.sort_values("effectiveDate", ascending=False).iloc[0]
            ytw = latest.get("yieldToWorst", None)
            eff_date = latest["effectiveDate"].date()
        else:
            ytw = None
            eff_date = None

        # Extract performance metrics
        perf_data = index_data.get("indexPerformanceHolder", {}).get("indexPerformance", {}).get("totalReturn", {})
        wam = perf_data.get("weightedAverageMaturity", None)
        duration = perf_data.get("weightedAverageDuration", None)

        records.append({
            "indexId": index_id,
            "effectiveDate": eff_date,
            "yieldToWorst": ytw,
            "weightedAverageMaturity": wam,
            "weightedAverageDuration": duration
        })

    return pd.DataFrame(records)


def get_schema_maturities_table():
    return [bigquery.SchemaField('effectivedate', bigquery.enums.SqlTypeNames.DATE),
            bigquery.SchemaField('weightedAverageMaturity', bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField('weightedAverageDuration', bigquery.enums.SqlTypeNames.FLOAT)]


def get_schema_yields_table():
    return [bigquery.SchemaField('date', 'DATE'),
            bigquery.SchemaField('ytw', 'FLOAT')]


GET_SCHEMA_FUNCTIONS = {'maturities': get_schema_maturities_table, 'yields': get_schema_yields_table}


def check_dates_in_tables(dates_to_check, tables_info) -> None:
    '''Checks if any dates from `dates_to_check` already exist in specified BigQuery tables in `tables_info`. 
    It queries each table for matches in the specified date column, and if a match is found, it raises an 
    `AssertionError` with details about the duplicate date and table.'''
    client = bigquery.Client(project=PROJECT_ID, location='US')

    for table_id, date_column in tables_info.items():
        dates_list = list(dates_to_check)
        query = f'SELECT {date_column} FROM `{table_id}` WHERE {date_column} IN UNNEST(@dates) LIMIT 1'    # `UNNEST(@dates)` expands the list of dates into rows for filtering; `LIMIT 1` stops the search as soon as a match is found to improve performance
        job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter('dates', 'DATE', dates_list)])
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        rows = list(results)
        assert len(rows) == 0, f'Data for date {rows[0][date_column]} already exists in table {table_id}'


@function_timer
def upload_df_to_bigquery(df: pd.DataFrame, table_id: str, yields_or_maturities: str):
    assert yields_or_maturities in ('yields', 'maturities')
    print(f'Uploading the following dataframe to {table_id} for {yields_or_maturities}')
    print(df.to_markdown())
    
    if TESTING:
        print(f'TESTING: Would upload the above dataframe to BigQuery table: {table_id}')
        print(f'TESTING: Schema used would be: {GET_SCHEMA_FUNCTIONS[yields_or_maturities]()}')
    else:
        client = bigquery.Client(project=PROJECT_ID, location='US')
        job_config = bigquery.LoadJobConfig(schema=GET_SCHEMA_FUNCTIONS[yields_or_maturities](), write_disposition='WRITE_APPEND')
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        try:
            job.result()
        except Exception as e:
            send_error_email('Error uploading S&P index maturities in cloud function: `update_sp_all_indices_and_maturities`',
                             f'This error means that there was an issue uploading data to BigQuery\nTable Name: {table_id}\n{type(e)}: {e}\n{traceback.format_exc()}')


def main(request):
    if today_is_a_holiday(): return 'SUCCESS'    # since S&P index data is not updated on national holidays, we do not need to run this cloud function, so this line will return 'SUCCESS' and exit early
    now = datetime.now(EASTERN)
    now = go_to_previous_weekday_if_weekend(now)    # if today is a Saturday or Sunday, we will use the previous Friday
    today = pd.Timestamp(now).tz_localize(None).normalize()    # `.tz_localize(None)` is to remove the time zone; `.normalize()` is used to remove the time component from the timestamp
    folder_path = today.strftime('%Y-%m-%d') 
    
    if TESTING:
        print('TESTING: Main function would process data and upload to BigQuery')
    try:    # wrap the entire procedure in a `try...except` so that we can send an email in the event of an issue
        # Step 1: Read data from GCS
        df = gcs_muni_jsons_to_dataframe(bucket_name='muni_index_jsons', folder_path=folder_path)
        df['effectivedate'] = pd.to_datetime(df['effectiveDate'])

        # Step 2: Collect dates to insert
        dates_to_check = set()
        for index_id in INDEX_IDS:
            index_data = df[df['indexId'] == str(index_id)]
            latest_index_data = index_data.sort_values(by='effectiveDate', ascending=False).iloc[0]    # sort by 'effectiveDate' and take the last row, the most recent record
            date_to_insert = latest_index_data['effectiveDate']#.date()
            dates_to_check.add(date_to_insert)

        # Step 3: Prepare table information
        tables_info = {}
        for index_id in INDEX_IDS:
            yields_table_name = INDEX_ID_TO_INDEX_YIELDS_TABLE_NAME[index_id]
            bigquery_yields_table_name = f'{PROJECT_ID}.spBondIndex.{yields_table_name}'
            tables_info[bigquery_yields_table_name] = 'date'

            maturities_table_name = INDEX_ID_TO_INDEX_MATURITIES_TABLE_NAME[index_id]
            bigquery_maturities_table_name = f'{PROJECT_ID}.spBondIndexMaturities.{maturities_table_name}'
            tables_info[bigquery_maturities_table_name] = 'effectivedate'

        # Step 4: Check if dates already exist in any table
        check_dates_in_tables(dates_to_check, tables_info)

        # Step 5: Proceed with data insertion
        for index_id in INDEX_IDS:
            yields_table_name = INDEX_ID_TO_INDEX_YIELDS_TABLE_NAME[index_id]
            bigquery_yields_table_name = f'{PROJECT_ID}.spBondIndex.{yields_table_name}'
            maturities_table_name = INDEX_ID_TO_INDEX_MATURITIES_TABLE_NAME[index_id]
            bigquery_maturities_table_name = f'{PROJECT_ID}.spBondIndexMaturities.{maturities_table_name}'
            print(f'Index ID: {index_id} corresponds to BigQuery Yields Table: {bigquery_yields_table_name} and BigQuery Maturities Table: {bigquery_maturities_table_name}')

            index_data = df[df['indexId'] == str(index_id)]
            latest_index_data = index_data.sort_values(by='effectiveDate', ascending=False).iloc[0]    # sort by 'effectiveDate' and take the last row, the most recent record
            latest_index_data_df = pd.DataFrame([latest_index_data])
            print('Latest index data:')
            print(latest_index_data_df.to_markdown())

            latest_maturities_df = latest_index_data_df[['effectivedate', 'weightedAverageMaturity', 'weightedAverageDuration']] #note effectivedate capitalization, which is not correct camelCase
            upload_df_to_bigquery(latest_maturities_df, bigquery_maturities_table_name, 'maturities')    # pass the DataFrame, not the original data

            latest_yields_df = latest_index_data_df[['effectiveDate', 'yieldToWorst']]
            latest_yields_df = latest_yields_df.rename(columns={'effectiveDate': 'date', 'yieldToWorst': 'ytw'})    # these are the column names for the BigQuery table
            upload_df_to_bigquery(latest_yields_df, bigquery_yields_table_name, 'yields')    # pass the DataFrame, not the original data
    except Exception as e:
        send_error_email(f'{type(e)} in cloud function: update_sp_all_indices_and_maturities: {e}', traceback.format_exc())
        raise e    # this ensures that even after we catch an error and send an email alert, we raise the error to make the function fail. This allows the cloud function instance to fail and retry
    return 'Success'


if __name__ == '__main__':
    main(None)