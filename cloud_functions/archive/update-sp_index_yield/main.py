# @Date: 2021-04-19
# @Last Modified date: 2024-02-01
import pandas as pd
import urllib.request
import urllib.error

from google.cloud import bigquery

import smtplib
from email.mime.text import MIMEText


opener = urllib.request.build_opener()
opener.addheaders = [
    (
        'User-Agent',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36',
    )
]
urllib.request.install_opener(opener)

table_headers = ['date', 'ytw']
link = 'https://www.spglobal.com/spdji/en/idsexport/file.xls?hostIdentifier=48190c8c-42c4-46af-8d1a-0cd5db894797&redesignExport=true&languageId=1&selectedModule=YieldToWorstGraphView&selectedSubModule=Graph&yearFlag=threeYearFlag&indexId=30000005'

PROJECT_ID = 'eng-reactor-287421'
TABLE_ID = 'eng-reactor-287421.spBondIndex.sp_muni_index_yield'


def get_data(link, filename_to_store_retrieved_url):
    def send_email_and_raise_error(error_message, error):
        print('Error message:', error_message)    # puts `error_message` in the logs of the cloud function
        send_email('Error in update-sp_index_yield cloud function', 'Error in get_data(...)\n' + error_message)
        raise error    # forces the cloud function to fail triggering a retry attempt on the cloud scheduler
    
    try:
        urllib.request.urlretrieve(link, filename_to_store_retrieved_url)
        return filename_to_store_retrieved_url
    except urllib.error.HTTPError as e:
        send_email_and_raise_error(f'HTTP Error {e.code}: {e.reason}', e)
    except urllib.error.URLError as e:
        send_email_and_raise_error(f'URL Error: {e.reason}', e)
    except Exception as e:
        send_email_and_raise_error(f'Exception occurred: {e}', e)


def send_email(subject, error_message):
    receiver_email = 'ficcteam@ficc.ai'
    sender_email = 'notifications@ficc.ai'
    sender_password = 'ztwbwrzdqsucetbg'
    msg = MIMEText(error_message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    smtp_server = 'smtp.gmail.com'
    port = 587
    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        except Exception as e:
            print(f'Exception occurred when sending email: {e}')
        finally:
            server.quit()


def convert_types(df):
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
    df.ytw = df.ytw * 100
    return df


def getSchema():
    schema = [bigquery.SchemaField('date', 'DATE'),
              bigquery.SchemaField('ytw', 'FLOAT')]
    return schema


def upload_data(df):
    client = bigquery.Client(project=PROJECT_ID, location='US')
    job_config = bigquery.LoadJobConfig(schema=getSchema(), write_disposition='WRITE_APPEND')
    job = client.load_table_from_dataframe(df, TABLE_ID, job_config=job_config)
    try:
        job.result()
        print(f'Upload to {TABLE_ID} Successful')
    except Exception as e:
        print(f'Failed to upload dataframe to {TABLE_ID}')
        raise e


def main(args):
    filename_to_store_retrieved_url = '/tmp/temp.xls'
    get_data(link, filename_to_store_retrieved_url)
    df = pd.read_excel(filename_to_store_retrieved_url)

    # remove the headers and tail disciption
    df.rename(columns={'Unnamed: 0': 'date', 'Unnamed: 1': 'ytw'}, inplace=True)
    df = df[8:-4]
    df = convert_types(df)

    df = df[-1:]  # get the last row as dataframe
    upload_data(df)
    return 'SUCCESS'
