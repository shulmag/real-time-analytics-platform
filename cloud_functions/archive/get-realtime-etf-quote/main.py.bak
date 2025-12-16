'''Last edited by Developer on 2024-05-23.'''
import pandas as pd
import datetime
import pytz
import requests
from google.cloud import bigquery
import time

finnhub_apikey = 'c499cpiad3ieskgqq5lg'
PROJECT_ID = "eng-reactor-287421"

best_funds = {
    'sp_12_22_year_national_amt_free_index': ['FMHI', 'MUB'],
    'sp_15plus_year_national_amt_free_index': [
        'FMHI',
        'MLN',
        'MUB',
        'TFI',
        'SUB',
        'SHYD',
        'HYMB',
        'HYD',
    ],
    'sp_7_12_year_national_amt_free_index': ['TFI', 'PZA', 'ITM', 'MLN'],
    'sp_high_quality_index': ['PZA', 'TFI', 'ITM'],
    'sp_high_quality_intermediate_managed_amt_free_index': ['TFI', 'PZA', 'ITM', 'MLN'],
    'sp_high_quality_short_intermediate_index': ['PZA', 'TFI', 'ITM'],
    'sp_high_quality_short_index': [
        'PZA',
        'HYMB',
        'HYD',
        'IBMM',
        'MLN',
        'ITM',
        'TFI',
        'SHYD',
        'SHM',
    ],
}

unique_funds = set([item for sublist in best_funds.values() for item in sublist])

def get_last_minute(timestamp: datetime.datetime):
    date = str(timestamp.date())
    hour = str(timestamp.hour).zfill(2)
    minute = str(timestamp.minute).zfill(2)

    datestring = '{} {}:{}'.format(date, hour, minute)

    return datetime.datetime.fromisoformat(datestring)


def getSchema():
    schema = [
        bigquery.SchemaField("datetime", "DATETIME"),
        bigquery.SchemaField("close", "FLOAT"),
    ]
    return schema


def get_quote_finnhub(etf: str):
    '''
    This function gets the current price for the given ETF using the finnhub.io API. There is a maximum of 60 calls per minute. The request returns a json file with a number of variables. 'c' refers to the Current Price.

    Parameters:
    etf:str

    '''
    response = requests.get(
        'https://finnhub.io/api/v1/quote?symbol={}&token={}'.format(etf, finnhub_apikey)
    )
    return response.json()['c']

def uploadData(df, TABLE_ID, schema):
    client = bigquery.Client(project=PROJECT_ID, location="US")
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND")

    job = client.load_table_from_dataframe(df, TABLE_ID, job_config=job_config)

    try:
        print(f'Uploading to {TABLE_ID}')
        job.result()
        print(f"Successfully uploaded to {TABLE_ID}")
        return True
    except Exception as e:
        print(f"Failed to upload to {TABLE_ID}; {e}")
        return False

        
def main(args):

    tz = pytz.timezone('US/Eastern')
    timestamp = datetime.datetime.now(tz)
    timestamp = get_last_minute(timestamp)

    target_date = str(timestamp.date())
    data = {}
    results = []
    
    # Get quotes first 
    for etf in unique_funds:
        close = get_quote_finnhub(etf)
        quote_df = pd.DataFrame([[timestamp, close]], columns=['datetime', 'close'])
        data[etf] = quote_df
    
    # Upload to BQ
    start = time.time()
    for etf, quote_df in data.items():
        results.append(uploadData(quote_df,
                        f"eng-reactor-287421.etf_prices.{etf}",
                        getSchema(),
                        )
                    )
    print(f'Took {time.time()-start:.2f}')

    # If not all passed, fail function               
    if not all(results): 
        raise Exception("One or more jobs failed.")

    return "SUCCESS"
