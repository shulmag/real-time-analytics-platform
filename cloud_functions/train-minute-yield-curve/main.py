'''
Description: Fits the realtime (minute) yield curve that is is used in production. S&P index tables in the `spBondIndex` and `spBondIndexMaturities` 
             datasets contain the raw yield and maturity scraped from the S&P website. S&P index tables in the `DATASET_NAME` dataset contains the 
             coefficients for the models being used to predict real-time S&P index values using the ETFs.

             See [this Notion page](https://www.notion.so/Yield-Curve-0e9d3fb1a49a4789826083361257a962?pvs=4#189eb87466c280d9ad01dc717ba0c6ae) for more details on related cloud functions and procedures.
'''
import time
import requests
import logging as python_logging    # to not confuse with google.cloud.logging
from datetime import datetime
import finnhub

import numpy as np
import pandas as pd
import pytz
import redis
import pickle

from google.cloud import bigquery, logging

from auxiliary_variables import TESTING, PROJECT_ID, DATASET_NAME, FINNHUB_API_KEY, ALPHA, BEST_FUNDS, DAILY_ETF_WEIGHTS_TABLES, SP_INDEX_DATASET, ETFs
from auxiliary_functions import timestamp_exists_in_redis, set_timestamp_in_redis, previous_business_day
from bigquery_utils import load_daily_etf_prices_bq, load_maturity_bq, load_scaler_daily_bq, load_index_yields_bq, load_etf_models_bq, get_scalar_df, load_shape_parameter, upload_etf_prices_to_bq
from yieldcurve import get_maturity_dict, get_NL_inputs, scale_X, run_NL_ridge, get_coefficient_df, get_values_for_date_from_df, get_scaler_params


if TESTING:
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/user/ficc/ficc/mitas_creds.json'
    python_logging.info = print
    python_logging.warning = print
else:
    # set up logging client; https://cloud.google.com/logging/docs/setup/python
    logging_client = logging.Client()
    logging_client.setup_logging()


YIELD_CURVE_REDIS_HOST = '10.227.69.60'
YIELD_CURVE_REDIS_CLIENT = redis.Redis(host=YIELD_CURVE_REDIS_HOST, port=6379, db=0)
FINNHUB_CLIENT = finnhub.Client(api_key=FINNHUB_API_KEY)

TIMEZONE = pytz.timezone('US/Eastern')


# Lazy initialization helps to avoid unecessary computations and cold starts, but defining these variables in global scope could help reduce latency
# According to GCP, cloud function instances are somtimes recycled and if they are, there is no need to load data from BigQuery each time the function is invoked if we do it at the start
if 'etf_data' not in globals(): etf_data = None
if 'maturity_df' not in globals(): maturity_df = pd.DataFrame()
if 'scaler_daily_parameters' not in globals(): scaler_daily_parameters = pd.DataFrame()
if 'index_data' not in globals(): index_data = None
if 'etf_model_data' not in globals(): etf_model_data = None
if 'last_updated' not in globals(): last_updated = None


def get_last_minute(timestamp: datetime) -> datetime:
    date = str(timestamp.date())
    hour = str(timestamp.hour).zfill(2)
    minute = str(timestamp.minute).zfill(2)

    datestring = f'{date} {hour}:{minute}'
    return datetime.fromisoformat(datestring)


def get_current_date():    # return type is `date` from the `datetime` module, but decided not to import it because it will get overwritten by other references to `date`
    return datetime.now(TIMEZONE).date()


def get_current_minute() -> datetime:
    return get_last_minute(datetime.now(TIMEZONE))


def get_schema_minute_yield() -> list:
    schema = [bigquery.SchemaField('date', 'DATETIME'),
              bigquery.SchemaField('const', 'FLOAT'),
              bigquery.SchemaField('exponential', 'FLOAT'),
              bigquery.SchemaField('laguerre', 'FLOAT')]
    return schema


def upload_data_to_bigquery(nelson_siegel_coefficients: pd.DataFrame, table_id: str, schema: list) -> None:
    client = bigquery.Client(project=PROJECT_ID, location='US')
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition='WRITE_APPEND')
    job = client.load_table_from_dataframe(nelson_siegel_coefficients, table_id, job_config=job_config)

    # perform exponential backoff a maximum of `max_runs` times if the procdure of uploading to the BigQuery table fails
    runs_so_far = 0
    max_runs = 5
    while runs_so_far < max_runs:
        try:
            job.result()
            python_logging.info(f'Upload to {table_id} successful')
            return None
        except Exception as e:
            sleep_time = min(2 ** runs_so_far, 10)
            runs_so_far += 1
            if runs_so_far >= max_runs:
                python_logging.warning(f'Already caught {type(e)}: {e}, {max_runs} times in `upload_data_to_bigquery`, so will now raise the error')
                raise e
            python_logging.warning(f'Failed to upload to {table_id} on attempt {runs_so_far} due to error {type(e)}: {e}. Will retry `upload_data_to_bigquery` {max_runs - runs_so_far} more times (next run will be {sleep_time} seconds later)')
            time.sleep(sleep_time)    # have a delay to prevent overloading the server


def upload_data_to_redis(timestamp_to_the_minute: datetime, nelson_siegel_coefficients: pd.DataFrame, scalar_values: pd.DataFrame, shape_parameter: float) -> None:
    timestamp_to_the_minute = timestamp_to_the_minute.strftime('%Y-%m-%d:%H:%M')

    if timestamp_exists_in_redis(timestamp_to_the_minute, YIELD_CURVE_REDIS_CLIENT):    # will only enter this `if` statement if using this function outside of the cloud function, e.g. to fill in missing values
        python_logging.warning(f'Since {timestamp_to_the_minute} already exists in redis on host {YIELD_CURVE_REDIS_HOST}, no value was uploaded.')
    else:
        nelson_siegel_coefficients = nelson_siegel_coefficients.set_index('date', drop=True)
        yield_curve_values = {'nelson_values': nelson_siegel_coefficients, 
                              'scalar_values': scalar_values, 
                              'shape_parameter': shape_parameter}

        set_timestamp_in_redis(timestamp_to_the_minute, YIELD_CURVE_REDIS_CLIENT, yield_curve_values)
        python_logging.info(f'Upload to redis on host {YIELD_CURVE_REDIS_HOST} successful. Key: {timestamp_to_the_minute}. Value: {yield_curve_values}')


def update_yc_all_data(timestamp_to_the_minute: datetime,
                       coefficient_df: pd.DataFrame,
                       exponential_mean: float,
                       exponential_std: float,
                       laguerre_mean: float,
                       laguerre_std: float,
                       tau: float) -> None:
    """Update the aggregate yc:all-data key in Redis by appending the new row."""
    try:
        existing_bytes = YIELD_CURVE_REDIS_CLIENT.get("yc:all-data")
        if existing_bytes:
            df_all = pickle.loads(existing_bytes)
        else:
            df_all = pd.DataFrame()


        new_row = pd.DataFrame([{
            "minute": pd.to_datetime(timestamp_to_the_minute),
            "const": float(coefficient_df["const"].iloc[0]),
            "exponential": float(coefficient_df["exponential"].iloc[0]),
            "laguerre": float(coefficient_df["laguerre"].iloc[0]),
            "exponential_mean": float(exponential_mean),
            "exponential_std": float(exponential_std),
            "laguerre_mean": float(laguerre_mean),
            "laguerre_std": float(laguerre_std),
            "shape_parameter": float(tau),
        }])


        df_all = pd.concat([df_all, new_row]).sort_values("minute").reset_index(drop=True)

        YIELD_CURVE_REDIS_CLIENT.set("yc:all-data", pickle.dumps(df_all))
        python_logging.info(f"Updated yc:all-data with {len(df_all)} rows")

    except Exception as e:
        python_logging.warning(f"Failed to update yc:all-data: {e}")

def get_quote_from_finnhub(etf: str, datetime_to_the_minute: datetime = None):
    '''This function gets the current price for the given ETF using the finnhub python library function .quote() There is a maximum of 60 calls per minute. 
    The request returns a json file with a number of variables, where 'c' refers to the Current Price. `datetime_to_the_minute` is 
    an optional argument only used for printing a descriptive error message.'''
    try:
        quote = FINNHUB_CLIENT.quote(etf)
        print(f'''This is the current quote {quote} for the EFT ticker {etf}.''')
        return quote['c']
    except Exception as e:
        datetime_suffix = '' if datetime_to_the_minute is None else f' for timestamp: {datetime_to_the_minute}'
        print(f'Unable to get quote data{datetime_suffix} for ETF: {etf} due to {type(e)}: {e}')
        return None
    
def get_quotes_with_retry(etfs, datetime_to_the_minute, max_attempts=5, base_sleep=0.5):
    """
    Re-requests only the ETFs that failed on the previous attempt.
    Returns a dict {ticker: price}. Raises if any still missing after retries.
    """
    quotes = {}
    missing = set(etfs)

    for attempt in range(1, max_attempts + 1):
        for etf in list(missing):
            price = get_quote_from_finnhub(etf, datetime_to_the_minute)
            # Treat None or 0 as missing
            if price not in (None, 0):
                quotes[etf] = float(price)
                missing.discard(etf)

        if not missing:
            return quotes  # success

        if attempt < max_attempts:
            # simple exponential backoff (capped)
            sleep_s = min(base_sleep * (2 ** (attempt - 1)), 5.0)
            python_logging.info(
                f'Missing quotes after attempt {attempt}: {sorted(missing)}. Retrying in {sleep_s:.2f}s...'
            )
            time.sleep(sleep_s)

    raise RuntimeError(f'Could not fetch quotes for: {sorted(missing)} after {max_attempts} attempts')


def series_as_one_line_string(series: pd.Series) -> None:
    '''Return `series` as one line string to be printed and not clutter the output.'''
    if not isinstance(series, pd.Series): return f'{series} is not a pd.Series as expected, but is instead a {type(series)}'
    return f'Name: {series.name}, Data: ' + ', '.join(f'{index}: {value}' for index, value in series.items())


def get_prediction_for_sp_maturity_table(table_name: str, 
                                         timestamp_to_the_minute: datetime, 
                                         previous_day: str, 
                                         intraday_change: pd.DataFrame, 
                                         model_parameters_from_bigquery_table: pd.DataFrame, 
                                         index_data_from_bigquery_table: pd.DataFrame, 
                                         verbose: bool = True):
    '''Returns a prediction based on an ETF model created in the `train_daily_etf_model` cloud function. `verbose` is 
    an optional boolean flag that determines if there is print output when running the function.'''
    _features_from_bigquery_table = list(model_parameters_from_bigquery_table.columns.drop('constant'))    # using '_' in the prefix of the variable name to denote that this variable is not used further downstream beyond printing a message
    _funds_from_bigquery_table = [fund_with_close_prefix.split('_')[1] for fund_with_close_prefix in _features_from_bigquery_table]    # using '_' in the prefix of the variable name to denote that this variable is not used further downstream beyond printing a message that there are extra ETFs (i.e., funds) available in the BigQuery table that are not being used; most likely because one of the ETFs is no longer active

    funds = BEST_FUNDS[table_name]    # these are the ETFs (i.e., funds) that will actually be used throughout this function
    if sorted(funds) != sorted(_funds_from_bigquery_table):
        if verbose: print(f'Best funds for {table_name}: {funds}. Funds available in BigQuery table: {_funds_from_bigquery_table}')
    else:
        if verbose: print(f'funds: {funds} for {table_name} at timestamp: {timestamp_to_the_minute}')

    # get model parameters for the `previous_day` if it exists, otherwise the most recent model parameters
    funds_with_close_prefix = [f'Close_{fund}' for fund in funds]
    model_parameters = model_parameters_from_bigquery_table[['constant'] + funds_with_close_prefix]
    model_parameters = get_values_for_date_from_df(previous_day, model_parameters, f'{PROJECT_ID}.{DATASET_NAME}.{table_name}')
    model_parameters = model_parameters.to_numpy()
    
    current_intraday_subset = intraday_change[funds_with_close_prefix]
    if current_intraday_subset.empty: raise RuntimeError(f'current_intraday_subset for {table_name} is empty. This may be a downstream error resulting from a failure in the `update_daily_etf_prices` cloud function.')
    if verbose: print(f'current_intraday_subset for {table_name}:\n{current_intraday_subset}')
    ones = np.ones((len(current_intraday_subset), 1))
    model_data = np.hstack((ones, current_intraday_subset.to_numpy()))

    predicted_ytw_change = (model_data * model_parameters).sum(axis=1)
    prev_ytw = get_values_for_date_from_df(previous_day, index_data_from_bigquery_table, f'{PROJECT_ID}.{SP_INDEX_DATASET}.{table_name}').ytw
    prediction = prev_ytw + predicted_ytw_change
    return prediction


def main(args):
    global etf_data, maturity_df, scaler_daily_parameters, index_data, etf_model_data, last_updated
    current_timestamp_to_the_minute = get_current_minute()

    current_date = get_current_date()
    last_updated == None
    if last_updated is None or last_updated < current_date:
        if last_updated is None:
            print('BQ DATA DOES NOT EXIST, LOADING NOW')
        else:    # last_updated < current_date; ensure that BigQuery tables are refreshed every day
            print(f'BQ DATA OUTDATED, LAST UPDATED: {last_updated}')

        last_updated = current_date    # take note of when tables were last updated
        etf_data = load_daily_etf_prices_bq()
        maturity_df = load_maturity_bq()
        scaler_daily_parameters = load_scaler_daily_bq()
        index_data = load_index_yields_bq()
        etf_model_data = load_etf_models_bq()

        print(f'BQ DATA REFRESHED, UPDATED: {last_updated}')
        print(f'Maturity data timestamp: {maturity_df.iloc[-1, :].name}')
        print(f'Scaler data timestamp: {scaler_daily_parameters.iloc[-1, :].name}')
    else:
        print(f'BQ DATA UP TO DATE, LAST UPDATED: {last_updated}')

    # Fetch all quotes, retrying only the ones that fail
    quotes = get_quotes_with_retry(ETFs, current_timestamp_to_the_minute)
    quote_data = pd.DataFrame([quotes], columns=ETFs)
    print(quote_data.to_markdown(index=False))
    if not TESTING:
        upload_etf_prices_to_bq(quote_data, f'{PROJECT_ID}.finnhub_io.finnhub_etf_data')

    # Get the most recent scaler and maturity data
    target_date = str(current_timestamp_to_the_minute.date())
    day_before_target_date = previous_business_day(target_date)
    exponential_mean, exponential_std, laguerre_mean, laguerre_std = get_scaler_params(day_before_target_date, scaler_daily_parameters)
    maturity_dict = get_maturity_dict(maturity_df, day_before_target_date)

    prev_close_data = []
    for fund in ETFs:
        close_data = etf_data[fund][f'Close_{fund}']
        most_recent_date = close_data.index.max()
        prev_close_data.append(close_data.loc[most_recent_date:])
        if most_recent_date != day_before_target_date: python_logging.warning(f'For {fund}, using close data from {most_recent_date} instead of the desired {day_before_target_date}. This is expected if {day_before_target_date} is a holiday.')
    prev_close_data = pd.concat(prev_close_data, axis=1)
    intraday_change = ((quote_data.values - prev_close_data) / prev_close_data) * 100 * 100    # first 100 is for percent, and second 100 is for basis points

    predicted_ytw = pd.DataFrame()
    for daily_etf_weights_table_name in DAILY_ETF_WEIGHTS_TABLES:
        predicted_ytw[daily_etf_weights_table_name] = get_prediction_for_sp_maturity_table(daily_etf_weights_table_name, 
                                                                                           current_timestamp_to_the_minute, 
                                                                                           day_before_target_date, 
                                                                                           intraday_change, 
                                                                                           etf_model_data[daily_etf_weights_table_name], 
                                                                                           index_data[daily_etf_weights_table_name])

    yield_curve_df = predicted_ytw.T.rename({0: 'ytw'}, axis=1)
    yield_curve_df['Weighted_Maturity'] = yield_curve_df.index.map(maturity_dict).astype(float)
    tau = load_shape_parameter(day_before_target_date)
    X, y = get_NL_inputs(yield_curve_df, tau)
    X = scale_X(X, exponential_mean, exponential_std, laguerre_mean, laguerre_std)
    ridge_model = run_NL_ridge(X, y, scale=False, alpha=ALPHA)
    coefficient_df = get_coefficient_df(ridge_model, current_timestamp_to_the_minute)

    if not TESTING: upload_data_to_bigquery(coefficient_df, f'{PROJECT_ID}.{DATASET_NAME}.nelson_siegel_coef_minute', get_schema_minute_yield())
    if not TESTING: upload_data_to_redis(current_timestamp_to_the_minute, coefficient_df, get_scalar_df(day_before_target_date), tau)
    if not TESTING: update_yc_all_data(current_timestamp_to_the_minute, coefficient_df, exponential_mean, exponential_std,laguerre_mean, laguerre_std, tau)

    return 'SUCCESS'


if __name__ == '__main__':
    main(None)
