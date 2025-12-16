'''
Description: See [this Notion page](https://www.notion.so/Yield-Curve-0e9d3fb1a49a4789826083361257a962?pvs=4#189eb87466c280d9ad01dc717ba0c6ae) for more details on related cloud functions and procedures.
             NOTE: `set_target_date(...)` is used when running this code for recovery when one or other of the upstream cloud functions has not run, see above. 
'''
from datetime import datetime, timedelta
from pytz import timezone

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

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


PROJECT_ID = 'eng-reactor-287421'
DATASET_NAME = 'yield_curves_v2'
TABLE_ID = f'{PROJECT_ID}.{DATASET_NAME}.shape_parameters'
SCHEMA = [bigquery.SchemaField('Date', 'DATE'),
          bigquery.SchemaField('L', 'FLOAT')]

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


class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]


def target_date_is_a_holiday() -> bool:
    '''Determine whether the target date is a US national holiday.'''
    target_date_as_timestamp = pd.Timestamp(TARGET_DATETIME).tz_localize(None).normalize()    # `.tz_localize(None)` is to remove the time zone; `.normalize()` is used to remove the time component from the timestamp
    current_year = TARGET_DATETIME.year
    holidays_in_last_year_and_next_year = set(USHolidayCalendarWithGoodFriday().holidays(start=f'{current_year - 1}-01-01',end=f'{current_year + 1}-12-31'))
    if target_date_as_timestamp in holidays_in_last_year_and_next_year:
        print(f'Target date, {target_date_as_timestamp}, is a national holiday, and so we will not perform large batch pricing, and so there will not be any files in the SFTP')
        return True
    return False


def upload_df_to_bigquery(df, table_id, schema):
    client = bigquery.Client(project=PROJECT_ID, location='US')
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition='WRITE_APPEND')
    job = client.load_table_from_dataframe(df, table_id,job_config=job_config)
    try:
        job.result()
        print(f'Successfully uploaded the following dataframe to {table_id}:\n{df.to_markdown()}')
    except Exception as e:
        print(f'Failed to upload the following dataframe to {table_id}:\n{df.to_markdown()}')
        raise e
    

def concatenate_dataframes_and_fill_in_values(df_list: list):
    df = pd.concat(df_list, axis=1)
    df.columns = SP_MATURITY_TABLES
    df.ffill(axis=0, inplace=True)
    df.dropna(inplace=True)
    return df


def load_index_data():
    index_data  = [] 
    for table in SP_INDEX_TABLES:
        query = f'SELECT * FROM `eng-reactor-287421.spBondIndex.{table}` ORDER BY date DESC'
        df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['ytw'] = df['ytw'] * 100
        df = df.drop_duplicates('date')
        df.set_index('date', inplace=True, drop=True)
        index_data.append(df)
    return concatenate_dataframes_and_fill_in_values(index_data)


def load_maturity_data():
    maturity_data  = []
    for table in SP_MATURITY_TABLES:
        query = f'SELECT * FROM `eng-reactor-287421.spBondIndexMaturities.{table}` ORDER BY effectivedate DESC'
        print(f'Making BigQuery call with query: {query}')
        df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')        
        df['effectivedate'] = pd.to_datetime(df['effectivedate'], format='%Y-%m-%d')
        df = df.drop_duplicates('effectivedate')
        df.set_index('effectivedate', inplace=True, drop=True)
        df = df[['weightedAverageMaturity']]
        maturity_data.append(df) 
    return concatenate_dataframes_and_fill_in_values(maturity_data)


def get_maturity_dict(maturity_df, date):
    print(f'Calling `get_maturity_dict()` with `date`: {date} and `maturity_df`:\nFirst 10 rows of `maturity_df`:\n{maturity_df.head(10).to_markdown()}\nLast 10 rows of `maturity_df`:\n{maturity_df.tail(10).to_markdown()}')
    temp_df = maturity_df.loc[date].T
    temp_dict = dict(zip(temp_df.index, temp_df.values))
    return temp_dict


def decay_transformation(t, L):
    return L * (1 - np.exp(-t/L)) / t


def laguerre_transformation(t, L):
    return (L * (1 - np.exp(-t/L)) / t) - np.exp(-t/L)


def run_NL_model(summary_df: pd.DataFrame, L):
    summary_df['X1'] = decay_transformation(summary_df['Weighted_Maturity'], L)
    summary_df['X2'] = laguerre_transformation(summary_df['Weighted_Maturity'], L)

    X = sm.add_constant(summary_df[['X1','X2']])
    y = summary_df.ytw
    lm = Ridge(alpha=0.001, random_state=1).fit(X , y)

    predictions = lm.predict(X)
    mae = mean_absolute_error(y,predictions)
    return lm, mae

def check_for_existing_value():
    """Checks for existing values in BigQuery for TARGET_DATE and returns the value if found."""
    client = bigquery.Client(project=PROJECT_ID, location='US')
    query = f"""
        SELECT L
        FROM `{TABLE_ID}`
        WHERE Date = @date
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("date", "DATE", TARGET_DATE)]
    )

    try:
        rows = list(client.query(query, job_config=job_config).result())
        return rows[0].L if rows else None
    except Exception as e:
        print(f"Error checking for existing data in BigQuery: {e}")
    return None


def main(args):
    if target_date_is_a_holiday(): return 'SUCCESS'    # since there is no S&P index data on national holidays, we do not need to run this function

    if (existing := check_for_existing_value()) is not None:
        print(f"WARNING: Value {existing} for {TARGET_DATE} already in {TABLE_ID}")
        return "SUCCESS"

    index_data = load_index_data()
    maturity_data = load_maturity_data()
    target_date_as_timestamp = pd.Timestamp(TARGET_DATE)     # conversion to `pd.Timestamp` needed for downstream operations due to legacy code
    maturity_dict = get_maturity_dict(maturity_data, target_date_as_timestamp)
    summary_df = pd.DataFrame(index_data.loc[target_date_as_timestamp])
    summary_df.columns = ['ytw']
    summary_df['Weighted_Maturity'] = summary_df.index.map(maturity_dict).astype(float)

    tau_dict = {}
    result_df = []

    for i in np.arange(0.001, 20.000, 0.001):
        model, mae = run_NL_model(summary_df, i)
        result_df.append({'L': i, 'MAE': mae, 'model': model})

    result_df = pd.DataFrame(result_df)
    result_df.set_index('L', inplace=True, drop=True)
    result_df = result_df.sort_values('MAE', ascending=True)
    tau_dict[target_date_as_timestamp] = result_df.index[0]

    tau_table = pd.DataFrame(tau_dict.items(), columns=['Date', 'L'])
    tau_table['Date'] = pd.to_datetime(tau_table['Date'])
    tau_table['L'] = tau_table['L'].astype(float)  
    if not TESTING: upload_df_to_bigquery(tau_table, TABLE_ID, SCHEMA)
    return 'SUCCESS'


if __name__ == '__main__':
    main(None)
