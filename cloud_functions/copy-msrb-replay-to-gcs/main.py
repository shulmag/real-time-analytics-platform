'''
Description: Retrieves the replay XML from the MSRB FTP. This cloud function loads the daily replay file to Google Cloud 
             Services. The daily replay file we believe is identical to the real-time MSRB trade messages feed, and so we 
             keep the replay file simply for completeness sake, and don't use it. This cloud function runs before `load-ice-file-to-bq`.'''
import time
import requests
from datetime import datetime, date
import logging as python_logging    # to not confuse with google.cloud.logging
from pytz import timezone

from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday    # used to create a business day defined on the US federal holiday calendar that can be added or subtracted to a datetime

from google.cloud import storage, bigquery, secretmanager, logging


# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/user/ficc/ficc/mitas_creds.json'


logging_client = logging.Client()
logging_client.setup_logging()


YEAR_MONTH_DAY = '%Y-%m-%d'

EASTERN = timezone('US/Eastern')

class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]
BUSINESS_DAY = CustomBusinessDay(calendar=USHolidayCalendarWithGoodFriday())    # used to skip over holidays when adding or subtracting business days

BQ_CLIENT = bigquery.Client()
STORAGE_CLIENT = storage.Client()

SECONDS_TO_WAIT_BEFORE_NEXT_CALL = 30    # to not hit MSRB API too quickly
MAXIMUM_NUM_DAYS_TO_UPDATE = 10    # Google cloud functions version 1 allows for a maximum of a 9 minute timeout, and each call is made `SECONDS_TO_WAIT_BEFORE_NEXT_CALL` seconds after one another, so this value is set to prevent hitting the timeout while also updating as many dates as possible


def access_secret_version(secret_id: str, project_id: str = 'eng-reactor-287421', version_id='latest'):
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
    response = secretmanager.SecretManagerServiceClient().access_secret_version(request={'name': name})
    payload = response.payload.data.decode('UTF-8')
    return payload


def get_msrb_replay_file(date_string: str, print_instead_of_warn: bool = False):
    '''Fetch a file with daily data for a given date from MSRB. MSRB can report corrections for up to 20 days after the trade. 
    `print_instead_of_warn` is a boolean that determines whether we use `print` or `python_logging.warning` to display messages, 
    and is set to `True` when running the file in a local script.
    URL for the production site: rtrsprodsubscription.msrb.org
    URL for beta testing: https://rtrsbetasubscription.msrb.org'''
    if print_instead_of_warn: python_logging.warning = print
    try:
        # construct the URL
        formatted_date = datetime.strptime(date_string, YEAR_MONTH_DAY).strftime(YEAR_MONTH_DAY)
        url = f'https://rtrsprodsubscription.msrb.org/rtrssubscriptionfilewebservice/api/Subscription.GetFile?filetype=RTRSReplay&dt={formatted_date}'
        
        headers = {'credentials': access_secret_version('msrb_username') + ',' + access_secret_version('msrb_password'),
                   'accept': 'application/json'}
        print(f'Making request with URL: {url} and headers: {headers}')
        response = requests.get(url, headers=headers)    # make the GET request
        
        # check if the response is None
        if response is None:
            python_logging.warning('No response received from the server')
            return None
        
        # check for HTTP errors
        if response.status_code != 200:
            python_logging.warning(f'HTTP Error {response.status_code}: {response.reason}. Ignore if {date_string} is a date with no trades, e.g., a weekend or holiday')
            return None
        
        return response.text
    
    except Exception as e:
        python_logging.warning(f'{type(e)}: {e}')


def get_last_data_entry() -> date:
    '''get the last date for which we have data'''
    query = 'SELECT MAX(publish_date) AS current_date FROM MSRB.msrb_daily_replay_files'
    print(f'Making BigQuery call with query: {query}')
    result = BQ_CLIENT.query(query).result()
    first_row = next(result.__iter__())    # for background, see https://github.com/googleapis/google-cloud-python/issues/9259
    first_row_current_date = first_row['current_date']
    print(f'Current date of the first row: {first_row_current_date}')
    return first_row_current_date


def get_storage_filepath(file_date: date) -> str:
    return f'{file_date.year}/daily_replay_file_{file_date.strftime(YEAR_MONTH_DAY)}.txt'    # use `.strftime(YEAR_MONTH_DAY)` to get rid of the timestamp of 00:00:00


def upload_to_storage(file_name, file_text):
    '''Upload MSRB data to Google Cloud storage bucket: `msrb_trade_history`.'''
    bucket_name = 'msrb_trade_history'
    bucket = STORAGE_CLIENT.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(file_text)
    print(f'{file_name} uploaded to {bucket_name}')


def main(args):
    wanted_date = get_last_data_entry() + (BUSINESS_DAY * 1)
    num_days_updated = 0
    num_consecutive_days_not_found_until_terminating = 2    # terminate procedure if more than `num_consecutive_days_not_found_until_terminating` consecutive days are not found in the data
    while wanted_date < datetime.now(EASTERN).date() and num_days_updated < MAXIMUM_NUM_DAYS_TO_UPDATE:
        data = get_msrb_replay_file(wanted_date.strftime(YEAR_MONTH_DAY))
        
        if data is None:
            if num_consecutive_days_not_found_until_terminating == 0: break
            num_consecutive_days_not_found_until_terminating -= 1
        else:
            upload_to_storage(get_storage_filepath(wanted_date), data)
            num_days_updated += 1
        
        wanted_date += (BUSINESS_DAY * 1)
        time.sleep(SECONDS_TO_WAIT_BEFORE_NEXT_CALL)    # to not hit MSRB API too quickly
    return 'SUCCESS'
