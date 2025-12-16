'''
Description: Updates the daily ETF prices that are used to calculate the intraday delta between the current ETF price and the 
             last close price in the `train-minute-yield-curve` cloud function.
             See [this Notion page](https://www.notion.so/Yield-Curve-0e9d3fb1a49a4789826083361257a962?pvs=4#189eb87466c280d9ad01dc717ba0c6ae) for more details on related cloud functions and procedures.
'''
import time
import requests
import logging as python_logging    # to not confuse with google.cloud.logging
from functools import wraps

import pandas as pd
from auxiliary_functions import access_secret_version

from google.cloud import bigquery, logging

FAILURE_EMAIL_RECIPIENTS = ['ficc-eng@ficc.ai']
SUCCESS_EMAIL_RECIPIENTS = FAILURE_EMAIL_RECIPIENTS

TESTING = False

BQ_CLIENT = bigquery.Client()

API_KEY = 'EZR0IHAAL6MFWX4B'

# Define the ETFs that we are using for our models
ETFs = ['HYD',
        'HYMB',
        'IBMJ',
        'IBMK',
        'IBML',
        'ITM',
        'MLN',
        'MUB',
        'PZA',
        'SHM',
        'SHYD',
        'SMB',
        'SUB',
        'TFI',
        'VTEB',
        'FMHI',
        'MMIN']

# Define our BigQuery projects and tables
PROJECT_ID = 'eng-reactor-287421'
ETF_DAILY_DATASET = 'ETF_daily_alphavantage'
BQ_PROJECT_DATASET = f'{PROJECT_ID}.{ETF_DAILY_DATASET}'

if TESTING:
    python_logging.info = print
    python_logging.warning = print
else:
    # set up logging client; https://cloud.google.com/logging/docs/setup/python
    logging_client = logging.Client()
    logging_client.setup_logging()


def run_multiple_times_before_failing(error_types: tuple, max_runs: int, exponential_backoff: bool = False):
    '''This function returns a decorator. It will run `function` over and over again until it does not 
    raise an Exception for a maximum of `max_runs` times. If `exponential_backoff` is set to `True`, then 
    the wait time is increased exponentially, otherwise it is a constant value.
    NOTE: max_runs = 1 is the same functionality as not having this decorator.
    NOTE: identical to `app_engine/demo/server/modules/ficc/utils/auxiliary_functions.py::run_multiple_times_before_failing(...)`.'''
    def decorator(function):
        @wraps(function)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
        def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
            runs_so_far = 0
            while runs_so_far < max_runs:
                try:
                    return function(*args, **kwargs)
                except error_types as e:
                    runs_so_far += 1
                    if runs_so_far >= max_runs:
                        python_logging.warning(f'Already caught {type(e)}: {e}, {max_runs} times in {function.__name__}, so will now raise the error')
                        raise e
                    python_logging.warning(f'Caught {type(e)}: {e}, and will retry {function.__name__} {max_runs - runs_so_far} more times')
                    delay = min(2 ** (runs_so_far - 1), 10) if exponential_backoff else 1
                    time.sleep(delay)    # have a delay to prevent overloading the server
        return wrapper
    return decorator


def load_daily_etf_prices_bq() -> dict:
    '''Loads the maturity data from the specified bigquery tables in `ETFs` and returns a dictionary with keys correspond to the ETF names.'''
    etf_data = {}

    for table in ETFs:
        query = f'SELECT * FROM {ETF_DAILY_DATASET}.{table} ORDER BY Date DESC LIMIT 1'
        df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')

        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True, drop=True)
        etf_data[table] = df

    assert list(etf_data.keys()) == ETFs, f'etf_data keys do not match ETFs\netf_data.keys(): {etf_data.keys()}\nETFs: {ETFs}'
    return etf_data


@run_multiple_times_before_failing((requests.exceptions.RequestException, ValueError), 5, False)
def get_alpha_vantage_data(symbol: str) -> dict:
    '''Downloads the daily ETF price data from the Alpha Vantage API.'''
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol={symbol}&apikey={API_KEY}&adjusted=False'
    print(f'Getting data for {symbol} with URL: {url}')
    r = requests.get(url)
    r.raise_for_status()    # Raise an HTTPError for bad responses (4xx and 5xx)
    data = r.json()
    if 'Time Series (Daily)' not in data: raise ValueError(f"Unexpected response format for {symbol}: {data}")
    return data


def download_daily_prices() -> dict:
    '''Downloads the daily ETF price data from the Alpha Vantage API, and saves it to a dictionary of dataframes.'''
    dataframes = {}

    for symbol in ETFs:
        data = get_alpha_vantage_data(symbol)
        df = pd.DataFrame(data['Time Series (Daily)']).T

        for col in df:
            df[col] = df[col].astype(float)

        df.index.rename('Date', inplace=True)
        df = df.rename({'1. open': 'Open',
                        '2. high': 'High',
                        '3. low': 'Low',
                        '4. close': 'Close',
                        '5. volume': 'Volume'}, axis=1)
        df.columns = df.columns + '_' + symbol
        df.index = pd.to_datetime(df.index)

        dataframes[symbol] = df
        time.sleep(15)    # adding delay so that we don't hit upper limit of API calls
    return dataframes


def get_col_names(df: pd.DataFrame) -> tuple:
    '''The columns of each ETF's dataframe have a naming prefix (ie Open_MUB, Close_MUB, etc). For convenience, this function retrieves each of those columns.'''
    return (df.filter(regex='Open').columns[0],
            df.filter(regex='Close').columns[0],
            df.filter(regex='Volume').columns[0],
            df.filter(regex='High').columns[0],
            df.filter(regex='Low').columns[0])


def get_schema(Open: str, Close: str, Volume: str, High: str, Low: str):
    '''Returns the schema of the bigquery table for each ETF. The names of the Open, Close, Volume, High and Low columns are taken as input because they are prefixed with the ETF name.'''
    job_config = bigquery.LoadJobConfig(schema=[bigquery.SchemaField('Date', bigquery.enums.SqlTypeNames.DATE),
                                                bigquery.SchemaField(Open, bigquery.enums.SqlTypeNames.FLOAT),
                                                bigquery.SchemaField(Close, bigquery.enums.SqlTypeNames.FLOAT),
                                                bigquery.SchemaField(Volume, bigquery.enums.SqlTypeNames.FLOAT),
                                                bigquery.SchemaField(High, bigquery.enums.SqlTypeNames.FLOAT),
                                                bigquery.SchemaField(Low, bigquery.enums.SqlTypeNames.FLOAT)], write_disposition='WRITE_APPEND')
    return job_config

def send_email(subject, message, recipients=FAILURE_EMAIL_RECIPIENTS):
    import smtplib    # lazy loading for lower latency
    from email.mime.text import MIMEText    # lazy loading for lower latency
    sender_email = access_secret_version('notifications_username')
    password = access_secret_version('notifications_password')
    smtp_server = 'smtp.gmail.com'
    port = 587
    
    server = smtplib.SMTP(smtp_server, port)
    server.starttls()
    server.login(sender_email, password)
    
    message = MIMEText(message)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = ', '.join(recipients)
    
    try:
        server.sendmail(sender_email, recipients, message.as_string())
    except Exception as e:
        python_logging.error(f"Failed to send email: {e}")
        print(e)
    server.quit()

def main(args):
    '''First, download the bigquery tables of existing ETF price data, and then download the daily data from Alpha Vantage. Check if the observations in the downloaded data are in bigquery. 
    If they are, make a note to print those with data already available. If not, we upload that data to bigquery.'''
    
    bq_data = load_daily_etf_prices_bq()
    daily_data = download_daily_prices()

    data_old = []    # keep track of (ETF, last updated date) pairs for ETfs that have the same data as what is already in the BigQuery table, which will help avoid duplicates
    errors = []

    # for each ETF in the data downloaded from Alpha Vantage, check if it is already available, and then upload if needed
    for name, df in daily_data.items():
        try:
            df = df.sort_index(ascending=True)
            df.index = pd.to_datetime(df.index)
            data_last_date = str(df.index[-1].date())    # date of the most recent entry in the downloaded data

            dates = pd.to_datetime(bq_data[name].index)
            last_date = str(dates[0].date())    # date of the most recent entry in the bigquery table

            if data_last_date == last_date:    # if the data is already available, then move on to the next ETF
                data_old.append((name, last_date))
                continue

            df = df.loc[data_last_date:]  # if the data is not available, get all the data from the last date as a dataframe

            # retrieve the column names, schema and then upload to Bigquery
            Open, Close, Volume, High, Low = get_col_names(df)
            job_config = get_schema(Open, Close, Volume, High, Low)
            df = df.reset_index(drop=False).sort_values(by='Date')

            job = BQ_CLIENT.load_table_from_dataframe(df, f'{BQ_PROJECT_DATASET}.{name}', job_config=job_config)
            job.result()
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            errors.append(f"Failed to process {name}:\n{error_trace}")
    if errors:
        subject = "[ALERT] ETF Daily Upload Failures"
        message = "One or more ETFs failed to upload:\n\n" + "\n\n".join(errors)
        send_email(subject, message)
        python_logging.error("Failures encountered:\n" + "\n".join(errors))

    if data_old:
        old_etfs, old_dates = zip(*data_old)
        warning = f'Data for {old_etfs} ETFs already available. '
        if len(set(old_dates)) == 1:    # all dates are the same
            warning += f'All share the same most recent updated date: {old_dates[0]}. '
        else:
            warning += f'Respective most recent updated dates: {old_dates}. '
        warning += f'This is expected behavior if the previous business day was a holiday.'
        python_logging.warning(warning)
    else:
        print(f'Upload Successful for all ETFs: {ETFs}')
    return 'SUCCESS'