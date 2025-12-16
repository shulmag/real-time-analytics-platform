'''
Description: See [this Notion page](https://www.notion.so/Yield-Curve-0e9d3fb1a49a4789826083361257a962?pvs=4#189eb87466c280d9ad01dc717ba0c6ae) for more details on related cloud functions and procedures.
             NOTE: `set_target_date(...)` is used when running this code for recovery when one or other of the upstream cloud functions has not run, see above. 
'''
from datetime import datetime, timedelta
from pytz import timezone
import redis
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday    # used to create a business day defined on the US federal holiday calendar that can be added or subtracted to a datetime

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
    
    return now, now.date(), now.strftime(YEAR_MONTH_DAY), now.replace(hour=0, minute=0, second=0, microsecond=0).strftime(f'{YEAR_MONTH_DAY} {HOUR_MIN_SEC}')


TARGET_DATETIME, TARGET_DATE, TARGET_DATE_STRING, TARGET_DATETIME_MIDNIGHT_STRING = _get_default_target_datetime()


def set_target_date(date_string=None) -> None:
    '''This function is not used in production. It is only used when running this code for recovery, e.g., cloud_functions/scripts/rerun_yield_curve_functions.py.'''
    global TARGET_DATE, TARGET_DATETIME, TARGET_DATE_STRING, TARGET_DATETIME_MIDNIGHT_STRING
    
    if date_string:
        try:
            TARGET_DATETIME = EASTERN.localize(datetime.strptime(date_string, YEAR_MONTH_DAY))
            TARGET_DATE = TARGET_DATETIME.date()
            TARGET_DATE_STRING = TARGET_DATETIME.strftime(YEAR_MONTH_DAY)
            TARGET_DATETIME_MIDNIGHT_STRING = TARGET_DATETIME.replace(hour=0, minute=0, second=0, microsecond=0).strftime(f'{YEAR_MONTH_DAY} {HOUR_MIN_SEC}')
        except ValueError:
            raise ValueError(f"Invalid date format: {date_string}. Expected format: YYYY-MM-DD")
    print(f"TARGET_DATETIME updated to: {TARGET_DATETIME}")
    print(f"TARGET_DATE updated to: {TARGET_DATE}")
    print(f"TARGET_DATE_STRING updated to: {TARGET_DATE_STRING}")
    print(f"TARGET_DATETIME_MIDNIGHT_STRING updated to: {TARGET_DATETIME_MIDNIGHT_STRING}")


class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]
BUSINESS_DAY = CustomBusinessDay(calendar=USHolidayCalendarWithGoodFriday())    # used to skip over holidays when adding or subtracting business days


if not TESTING: BQ_CLIENT = bigquery.Client()

PROJECT_ID = 'eng-reactor-287421'
DATASET_NAME = 'yield_curves_v2'
TABLE_ID_MODEL = f'{PROJECT_ID}.{DATASET_NAME}.nelson_siegel_coef_daily'
TABLE_ID_SCALER = f'{PROJECT_ID}.{DATASET_NAME}.standardscaler_parameters_daily'

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


def target_date_is_a_holiday() -> bool:
    '''Determine whether the target date is a US national holiday.'''
    target_date_as_timestamp = pd.Timestamp(TARGET_DATETIME).tz_localize(None).normalize()    # `.tz_localize(None)` is to remove the time zone; `.normalize()` is used to remove the time component from the timestamp
    current_year = TARGET_DATETIME.year
    holidays_in_last_year_and_next_year = set(USHolidayCalendarWithGoodFriday().holidays(start=f'{current_year - 1}-01-01',end=f'{current_year + 1}-12-31'))
    if target_date_as_timestamp in holidays_in_last_year_and_next_year:
        print(f'Target date, {target_date_as_timestamp}, is a national holiday.')
        return True
    return False


def load_index_data() -> pd.DataFrame:
    '''Load the S&P index data into a single dataframe. Returns a dataframe containing the 
    yield to worst of all the indices.'''
    index_data = [] 
    for table in SP_INDEX_TABLES:
        query = f'SELECT * FROM `{PROJECT_ID}.spBondIndex.{table}` ORDER BY date DESC LIMIT 10'    # TODO: why do we have a limit of 10 instead of 1?
        df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['ytw'] = df['ytw'] * 100
        df = df.drop_duplicates('date')
        df.set_index('date', inplace=True, drop=True)
        index_data.append(df)
    
    df = pd.concat(index_data, axis=1)
    df.columns = SP_MATURITY_TABLES
    df.ffill(inplace=True, axis=0)
    print(f'Using the following index data:\n{df.to_markdown()}')
    return df


def load_maturity_data() -> pd.DataFrame:
    '''Load the S&P maturity data into a single dataframe. Returns a dataframe containing the 
    weighted average maturities of all the indices.'''
    maturity_data  = []
    for table in SP_MATURITY_TABLES:
        query = f'SELECT * FROM `{PROJECT_ID}.spBondIndexMaturities.{table}` ORDER BY effectivedate DESC LIMIT 10'    # the limit of 10 ensures that there is enough of a data range so that we can forward fill the data using `.ffill(...)` in case certain indices are missing
        df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')        
        df['effectivedate'] = pd.to_datetime(df['effectivedate'], format='%Y-%m-%d')
        df = df.drop_duplicates('effectivedate')
        df.set_index('effectivedate', inplace=True, drop=True)
        
        df = df[['weightedAverageMaturity']]
        maturity_data.append(df) 
        
    df = pd.concat(maturity_data, axis=1)
    df.columns = SP_MATURITY_TABLES
    df.sort_index(ascending=True, inplace=True)
    df.ffill(inplace=True, axis=0)
    df = df.iloc[-1:]    # select the data corresponding to the most recent date
    print(f'Using the following maturity data:\n{df.to_markdown()}')
    effective_date_in_df = df.index[0].strftime(YEAR_MONTH_DAY)
    assert effective_date_in_df == TARGET_DATE_STRING, f'The effective date in the maturity data: {effective_date_in_df}, does not match the target date: {TARGET_DATE_STRING}'
    return df


def get_maturity_dict(maturity_df: pd.DataFrame, date: str) -> dict:
    '''Creates a dictonary with the index namebeing the key and the weighted average maturities as the values.'''
    df = maturity_df.loc[date].T
    return dict(zip(df.index, df.values))


def get_yield_curve_maturity_df(index_data: pd.DataFrame, date: str, maturity_dict: dict) -> pd.DataFrame:
    '''Creates a dataframe that contains the yield to worst and weighted average maturity for a specific date.'''
    df = index_data.loc[[date]].T
    df.columns = ['ytw']
    df['Weighted_Maturity'] = df.index.map(maturity_dict)
    return df


def decay_transformation(t: np.array, L: float):
    '''Returns the exponential function calculated from the inputted numpy array of maturities and shape parameter.'''
    return L * (1 - np.exp(-t/L)) / t


def laguerre_transformation(t: np.array, L: float):
    '''Returns the laguerre function calculated from inputted numpy array of maturities and shape parameter.'''
    return (L * (1 - np.exp(-t/L)) / t) - np.exp(-t/L)    # TODO: use `decay_transformation(...)` as a subprocedure, i.e., replace this line with `decay_transformation(t, L) - np.exp(-t/L)`


def get_model_inputs(yield_curve_maturity_df: pd.DataFrame, L: int):
    '''Creates the inputs for the regression model. Inputs are created using the exponential and laguerre transform.'''
    yield_curve_maturity_df = yield_curve_maturity_df.copy()
    yield_curve_maturity_df['X1'] = decay_transformation(yield_curve_maturity_df['Weighted_Maturity'], L)
    yield_curve_maturity_df['X2'] = laguerre_transformation(yield_curve_maturity_df['Weighted_Maturity'], L)
    
    X = yield_curve_maturity_df[['X1', 'X2']]
    y = yield_curve_maturity_df['ytw']
    return X, y


def train_model(X: np.array, Y: float):
    '''Train a regression model to estimate the Nelson-Siegel coefficients.'''
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model = Ridge(alpha=0.001, random_state=1).fit(X , Y)
    return scaler, model


def get_schema_model():
    '''Returns the schema required for the bigquery table storing the Nelson-Siegel coefficients.'''
    schema = [bigquery.SchemaField('date', 'DATE', 'REQUIRED'),
              bigquery.SchemaField('const','FLOAT', 'REQUIRED'),
              bigquery.SchemaField('exponential','FLOAT', 'REQUIRED'),
              bigquery.SchemaField('laguerre','FLOAT', 'REQUIRED')]
    return schema


def get_schema_scaler():
    '''Returns the schema required for the BigQuery table storing the sklearn StandardScaler's parameters 
    Nelson-Siegel coefficients.'''
    schema = [bigquery.SchemaField('date', 'DATE', 'REQUIRED'),
              bigquery.SchemaField('exponential_mean', 'FLOAT', 'REQUIRED'),
              bigquery.SchemaField('exponential_std', 'FLOAT', 'REQUIRED'),
              bigquery.SchemaField('laguerre_mean', 'FLOAT', 'REQUIRED'),
              bigquery.SchemaField('laguerre_std', 'FLOAT', 'REQUIRED')]
    return schema


def upload_df_to_bigquery(df: pd.DataFrame, table_id: str):
    if table_id == TABLE_ID_MODEL:
        job_config = bigquery.LoadJobConfig(schema=get_schema_model(), write_disposition='WRITE_APPEND')
    elif table_id == TABLE_ID_SCALER:
        job_config = bigquery.LoadJobConfig(schema=get_schema_scaler(), write_disposition='WRITE_APPEND')
    else:
        raise ValueError(f'Table ID: {table_id} is not supported')
    
    job = BQ_CLIENT.load_table_from_dataframe(df, table_id, job_config=job_config)
    try:
        job.result()
        print(f'Successfully uploaded the following dataframe to {table_id}:\n{df.to_markdown()}')
    except Exception as e:
        print(f'Failed to upload the following dataframe to {table_id}:\n{df.to_markdown()}')
        raise e


def load_shape_parameter() -> float:
    '''Grabs the latest shape parameter for the Nelson-Siegel model.'''
    query = f'SELECT Date, L FROM `{PROJECT_ID}.{DATASET_NAME}.shape_parameters` ORDER BY Date DESC LIMIT 1'    # `LIMIT 1` in the query gets the most recent shape parameter
    df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')
    print(f'Query: {query}, results in the following dataframe:\n{df.to_markdown()}')
    date, L = df.iloc[0].values
    previous_business_date = (TARGET_DATE - (BUSINESS_DAY * 1)).date() #to ensure previous_business_date does not become a timestamp. 
    assert date == previous_business_date, f'The date in the shape parameter: {date}, does not match the previous business date: {previous_business_date}'    # shape parameter for `TARGET_DATE` is being updated in the `compute_shape_parameter` cloud function which runs after this function
    return L


def main(args):
    if target_date_is_a_holiday(): return 'SUCCESS'
    maturity_data = load_maturity_data()
    index_data = load_index_data()
    L = load_shape_parameter()
    coefficient_df = pd.DataFrame()
    scaler_df = pd.DataFrame()

    # creating a dataframe to send inputs to the model
    print(f'Calculating the coefficients for {TARGET_DATE_STRING}')
    maturity_dict = get_maturity_dict(maturity_data, TARGET_DATETIME_MIDNIGHT_STRING)    # need to use `TARGET_DATETIME_MIDNIGHT_STRING` since the table has `00:00:00` at the end of the date
    yield_curve_maturity_df = get_yield_curve_maturity_df(index_data, TARGET_DATETIME_MIDNIGHT_STRING, maturity_dict)    # need to use `TARGET_DATETIME_MIDNIGHT_STRING` since the table has `00:00:00` at the end of the date

    # creating the inputs for the model
    X, Y = get_model_inputs(yield_curve_maturity_df, L)
    scaler, model = train_model(X, Y)

    # retrieve model parameters
    const = model.intercept_
    exponential = model.coef_[0]
    laguerre = model.coef_[1]

    # retrieve scaler parameters, used to standardize the data
    exponential_mean = scaler.mean_[0]
    exponential_std = np.sqrt(scaler.var_[0])
    laguerre_mean = scaler.mean_[1]
    laguerre_std = np.sqrt(scaler.var_[1])
    
    temp_coefficient_df = pd.DataFrame({'date': TARGET_DATE,
                                        'const': const,
                                        'exponential': exponential,
                                        'laguerre': laguerre}, index=[0])

    temp_scaler_df = pd.DataFrame({'date': TARGET_DATE,
                                   'exponential_mean': exponential_mean,
                                   'exponential_std': exponential_std,
                                   'laguerre_mean': laguerre_mean,
                                   'laguerre_std': laguerre_std}, index=[0])

    coefficient_df = coefficient_df.append(temp_coefficient_df)
    scaler_df = scaler_df.append(temp_scaler_df)    

    if not TESTING:
        upload_df_to_bigquery(coefficient_df, TABLE_ID_MODEL) 
        upload_df_to_bigquery(scaler_df, TABLE_ID_SCALER)

    next_business_day = TARGET_DATE + (BUSINESS_DAY * 1)
    next_business_day = next_business_day.strftime('%Y-%m-%d')
    
    coefficient_df.reset_index(inplace=True, drop=True)
    scaler_df.reset_index(inplace=True, drop=True)
    nelson_values = coefficient_df.set_index('date')
    scalar_values = scaler_df.set_index('date')

    values_dict = {'nelson_values': nelson_values, 'scalar_values': scalar_values, 'shape_parameter': L}
    if not TESTING:
        redis_client = redis.Redis(host='10.227.69.60', port=6379, db=0)
        redis_client.set(next_business_day, pickle.dumps(values_dict, protocol=pickle.HIGHEST_PROTOCOL))    # uploading daily yield curve values to redis so that if we want to switch to using the daily yield curve in production, we can do so very easily by changing the server code to get the yield curve values corresponding to a date instead of a datetime
    uploaded_prefix = 'Would have uploaded' if TESTING else 'Uploaded'
    print(f'{uploaded_prefix} the following values to redis for {next_business_day}:')
    for key, value in values_dict.items():
        value_string = value.to_markdown() if isinstance(value, pd.DataFrame) else value
        print(f'{key}:\n{value_string}')
    return 'SUCCESS'


if __name__ == "__main__":
    main({})
