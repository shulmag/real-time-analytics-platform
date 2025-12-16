'''
Description: This cloud function trains daily LASSO regression models to compute coefficients (weights) that relate ETF price 
             changes to yield-to-worst (YTW) changes for S&P municipal bond indices. It uses the last 45 (`TRAIN_WINDOW_SIZE`) days 
             of daily close S&P index values and ETF prices, and selects the optimal subset of ETFs for each index. The function 
             calculates coefficients and intercepts for each model, storing them in tables of the form `yield_curves_v2.{sp_index_name}`. 
             These tables include a constant value and weights for each ETF, labeled `{ETF}_close`. If a weight is 0, it indicates that 
             the LASSO model assigns no predictive value to that ETF for the specific index. In production, these coefficients are used 
             with real-time ETF price changes to predict real-time YTW changes for the indices. 
             NOTE: The LASSO model is computing the weights of each ETF in predicting the *changes* in the SP index YTW, not directly 
             predicting the YTW itself.

             See [this Notion page](https://www.notion.so/Yield-Curve-0e9d3fb1a49a4789826083361257a962?pvs=4#189eb87466c280d9ad01dc717ba0c6ae) for more details on related cloud functions and procedures.
             NOTE: `set_target_date(...)` is used when running this code for recovery when one or other of the upstream cloud functions has not run, see above. 
'''
import numpy as np 
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
from pytz import timezone
from datetime import datetime, timedelta

from sklearn.linear_model import Lasso

from google.cloud import bigquery


TESTING = False
if TESTING:
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/user/ficc/creds.json'


YEAR_MONTH_DAY = '%Y-%m-%d'
HOUR_MIN_SEC = '%H:%M:%S'

EASTERN = timezone('US/Eastern')

def _get_default_target_datetime():
    now = datetime.now(EASTERN)
    current_weekday = now.weekday()
    
    # If it's Saturday (5) or Sunday (6), adjust to the most recent Friday
    if current_weekday >= 5:  # Weekend
        days_to_subtract = current_weekday - 4  # 4 is Friday
        now = now - timedelta(days=days_to_subtract)
        print(f"It was {'Saturday' if current_weekday == 5 else 'Sunday'}, so adjusted to Friday:", now)
    
    return now, now.date()


TARGET_DATETIME, TARGET_DATE = _get_default_target_datetime()


def set_target_date(date_string=None) -> None:
    '''This function is not used in production. It is only used when running this code for recovery, e.g., ...'''
    global TARGET_DATETIME, TARGET_DATE 
    
    if date_string:
        try:
            TARGET_DATETIME = EASTERN.localize(datetime.strptime(date_string, YEAR_MONTH_DAY))
            TARGET_DATE = TARGET_DATETIME.date()
        except ValueError:
            raise ValueError(f"Invalid date format: {date_string}. Expected format: YYYY-MM-DD")
    else:
        TARGET_DATETIME, TARGET_DATE = _get_default_target_datetime()

    print(f"TARGET_DATETIME updated to: {TARGET_DATETIME}")
    print(f"TARGET_DATE updated to: {TARGET_DATE}")



PROJECT_ID = 'eng-reactor-287421' 
ETF_DAILY_DATASET = 'ETF_daily_alphavantage'
SP_INDEX_DATASET = 'spBondIndex'
DATASET_NAME = 'yield_curves_v2'

TRAIN_WINDOW_SIZE = 45    # number of days of prior data to train the ETF model on; previously tuned hyperparameter

SP_INDEX_TABLES = ['sp_12_22_year_national_amt_free_index',
                   'sp_15plus_year_national_amt_free_index',
                   'sp_7_12_year_national_amt_free_municipal_bond_index_yield',
                   'sp_muni_high_quality_index_yield',
                   'sp_high_quality_intermediate_managed_amt_free_municipal_bond_index_yield',
                   'sp_high_quality_short_intermediate_municipal_bond_index_yield',
                   'sp_high_quality_short_municipal_bond_index_yield',
                   'sp_long_term_national_amt_free_municipal_bond_index_yield']

SP_MATURITY_TABLES = ['sp_12_22_year_national_amt_free_index',
                      'sp_15plus_year_national_amt_free_index',
                      'sp_7_12_year_national_amt_free_index',
                      'sp_high_quality_index',
                      'sp_high_quality_intermediate_managed_amt_free_index',
                      'sp_high_quality_short_intermediate_index',
                      'sp_high_quality_short_index',
                      'sp_long_term_national_amt_free_municipal_bond_index_yield']

BEST_FUNDS = {'sp_12_22_year_national_amt_free_index': ['FMHI', 'MUB'],
              'sp_15plus_year_national_amt_free_index': ['FMHI', 'MLN', 'MUB', 'TFI', 'SUB', 'SHYD', 'HYMB', 'HYD'],
              'sp_7_12_year_national_amt_free_index': ['TFI', 'PZA', 'ITM', 'MLN'],
              'sp_high_quality_index': ['PZA', 'TFI', 'ITM'],
              'sp_high_quality_intermediate_managed_amt_free_index': ['TFI', 'PZA', 'ITM', 'MLN'],
              'sp_high_quality_short_intermediate_index': ['PZA', 'TFI', 'ITM'],
              'sp_high_quality_short_index': ['PZA', 'HYMB', 'HYD', 'MLN', 'ITM', 'TFI', 'SHYD', 'SHM'],
              'sp_long_term_national_amt_free_municipal_bond_index_yield': ['FMHI', 'MLN', 'MUB', 'SUB']}

BEST_LAMBDAS = {'sp_12_22_year_national_amt_free_index': 5.0,
                'sp_15plus_year_national_amt_free_index': 5.0,
                'sp_7_12_year_national_amt_free_index': 1.0,
                'sp_high_quality_index': 1.0,
                'sp_high_quality_intermediate_managed_amt_free_index': 1.0,
                'sp_high_quality_short_intermediate_index': 1.0,
                'sp_high_quality_short_index': 1.0,
                'sp_long_term_national_amt_free_municipal_bond_index_yield': 5.0}

SP_INDEX_TABLES_TO_SP_MATURITY_TABLES = dict(zip(SP_INDEX_TABLES, SP_MATURITY_TABLES))
ETFs = list(set([fund for funds in BEST_FUNDS.values() for fund in funds]))


class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]


def target_date_is_a_holiday() -> bool:
    '''Determine whether the target date is a US national holiday.'''
    target_date_as_timestamp = pd.Timestamp(TARGET_DATETIME).tz_localize(None).normalize()    # `.tz_localize(None)` is to remove the time zone; `.normalize()` is used to remove the time component from the timestamp
    current_year = TARGET_DATETIME.year
    holidays_in_last_year_and_next_year = set(USHolidayCalendarWithGoodFriday().holidays(start=f'{current_year - 1}-01-01',end=f'{current_year + 1}-12-31'))
    if target_date_as_timestamp in holidays_in_last_year_and_next_year:
        print(f'Target date, {target_date_as_timestamp}, is a national holiday')
        return True
    return False


def load_etf_data():
    '''Load the daily ETF prices from BigQuery. The data for each etf is loaded as a dataframe and then combined  
    into a dictionary.'''
    etf_data  = {}
    for table in ETFs:
        query = f'''SELECT DISTINCT * FROM {ETF_DAILY_DATASET}.{table}'''
        print(f'Making a BigQuery call with query: {query}')
        
        df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True, drop=True)
        etf_data[table] = df.drop_duplicates()
        
    return etf_data


def load_index_yields():
    '''Load the S&P index yields from BigQuery. Each individual index is read a dataframe which are then combined 
    into a dictionary.'''
    index_data  = {}
    for table in SP_INDEX_TABLES:
        query = f'''SELECT DISTINCT * FROM {SP_INDEX_DATASET}.{table}'''
        print(f'Making a BigQuery call with query: {query}')

        df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')
        df = df.drop_duplicates('date')
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df.sort_values('date', inplace=True, ascending=True)
        df.set_index('date', inplace=True, drop=True)
        assert df.index.max().date() == TARGET_DATE, f'Most recent date in {SP_INDEX_DATASET}.{table} is {df.index.max().date()}, which is not TARGET_DATE: {TARGET_DATE}. Most likely, the upstream cloud function `update_sp_all_indices_and_maturities` has failed.'    # check that the date is correct for each dataset

        df['ytw'] = df['ytw'] * 100    # convert to basis points
        
        table_name = SP_INDEX_TABLES_TO_SP_MATURITY_TABLES[table]    # standardize names between maturity and yield data
        index_data[table_name] = df 
        
    return index_data


def preprocess_data(index_data: dict, etf_data: dict, index_name: str, etf_names: list, date_start='2020-05', var='Close') -> pd.DataFrame:
    '''Takes as input (1) the loaded S&P index data and ETF data from BigQuery, which is stored as a dictionary of dataframes, 
    and (2) the name of a single S&P index and a list of ETFs that are relevant to predicting that index. Then merges this data 
    into a single dataframe, calculating the `pct_change` in ETF prices in basis points and the change in index YTW in basis points. 
    This is done, by default, for observations after May 2020 and for the Close prices of the ETFs. The merged result is returned.'''
    data = []
    
    # preprocess etf data by retrieving ETFs of interest and calculating pct_change in basis points
    for etf_name in etf_names:
        etf = etf_data[etf_name].copy()
        etf = etf[~etf.index.duplicated(keep='last')]    # remove any rows where the date is duplicated because this causes `pd.concat(...)` to fail with `ValueError: cannot reindex on an axis with duplicate labels`
        etf_change = etf[f'{var}_{etf_name}'].pct_change() * 100 * 100    # first 100 is for percent, and second 100 is for basis points
        data.append(etf_change)
    etf = pd.concat(data, axis=1)
    
    # preprocess index data by first-differencing YTW
    index = index_data[index_name].copy()
    index['ytw_diff'] = index['ytw'].diff()
    
    # merge ETF and index data
    merged_df = pd.merge(etf, index, left_index=True, right_index=True).loc[date_start:]
    return merged_df.dropna()


def get_schema_etf(coefficient_df: pd.DataFrame):
    '''Gets the BigQuery schema to upload the data to the bq table containing the coefficients 
    for the linear model using ETF prices to predict index yield, for each index.'''
    schema = [bigquery.SchemaField('Date', 'DATE')] + [bigquery.SchemaField(column, 'FLOAT') for column in coefficient_df.columns if column != 'Date']
    return schema


def upload_df_to_bigquery(df: pd.DataFrame, table_id: str, schema):
    client = bigquery.Client(project=PROJECT_ID, location='US')
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition='WRITE_APPEND')
    try:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        print(f'Successfully uploaded the following dataframe to {table_id}:\n{df.to_markdown()}')
    except Exception as e:
        print(f'Failed to upload the following dataframe to {table_id} with schema:\n{schema}\n{df.to_markdown()}')
        raise e
  

def main(args):
    '''First load the index and etf data, then for each S&P index, train a model using the previously identified 
    optimal subset of ETFs to predict yields. Training data size is equal to the window size, also previously identified.'''
    if target_date_is_a_holiday(): return 'SUCCESS'    # since there is no S&P index data on national holidays, we do not need to run this function

    index_data = load_index_yields()
    etf_data = load_etf_data()

    for current_index, current_best_funds in BEST_FUNDS.items():
        current_data = preprocess_data(index_data, etf_data, current_index, current_best_funds)
        current_data = current_data.tail(TRAIN_WINDOW_SIZE)    # training data size is the window size

        # get data and labels
        X = current_data.drop(['ytw', 'ytw_diff'], axis=1)
        y = current_data['ytw_diff']
        X_cols = list(X.columns)
        assert len(X) == len(y), f'Number of training data points: {len(X)}, does not match the number of training labels: {len(y)}'
        
        # train the model 
        current_best_lambda = BEST_LAMBDAS[current_index]
        lasso = Lasso(alpha=current_best_lambda, random_state=1, max_iter=5000).fit(X, y)

        # save the coefficients to one row dataframe and append it to bigquery
        columns = ['constant'] + X_cols
        coefficients = np.hstack([lasso.intercept_, lasso.coef_])
        date = X.index.max().date().isoformat()
        results_dict = {date: dict(zip(columns, coefficients))}    # used to save the coefficients
        coefficient_df = pd.DataFrame(results_dict).T
        coefficient_df.index = pd.to_datetime(coefficient_df.index)
        coefficient_df = coefficient_df.reset_index(drop=False).rename({'index': 'Date'}, axis=1)
        coefficient_df['Date'] = coefficient_df['Date'].dt.date    # convert `Date` to a date type to avoid type conversion error: `pyarrow.lib.ArrowNotImplementedError: Unsupported cast from timestamp[ns] to double using function cast_double`
        
        if not TESTING:
            schema = get_schema_etf(coefficient_df)
            table_id = f'{PROJECT_ID}.{DATASET_NAME}.{current_index}'
            upload_df_to_bigquery(coefficient_df, table_id, schema)
    
    return 'SUCCESS'

if __name__ == "__main__":
    main({})
