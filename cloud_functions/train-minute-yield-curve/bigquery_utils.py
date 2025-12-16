'''
'''
import pandas as pd
from datetime import datetime

from google.cloud import bigquery

from auxiliary_variables import EASTERN, \
                                PROJECT_ID, \
                                DATASET_NAME, \
                                SP_ETF_DAILY_DATASET, \
                                SP_ETF_HOURLY_DATASET, \
                                SP_INDEX_DATASET, \
                                SP_MATURITY_DATASET, \
                                DAILY_ETF_WEIGHTS_TABLES, \
                                SP_INDEX_TABLES, \
                                SP_INDEX_TABLE_TO_DAILY_ETF_WEIGHTS_TABLE, \
                                ETFs
from auxiliary_functions import sqltodf, set_date_as_index, get_values_for_date_from_df


LIMIT = 15    # used to limit the number of rows returned from BigQuery; chosen to be greater than `max_attempts` defined in `auxiliary_functions.py::get_values_for_date_from_df(...)`


def drop_duplicates_considering_index_and_values(df: pd.DataFrame) -> pd.DataFrame:
    '''Drop duplicate rows from `df` where duplicate takes into account both the index value and the row values.
    
    >>> data = {'A': [1, 2, 2, 2, 3], 'B': [4, 5, 5, 5, 6]}
    >>> df = pd.DataFrame(data, index=[0, 1, 1, 2, 2])
    >>> df
       A  B
    0  1  4
    1  2  5
    1  2  5
    2  2  5
    2  3  6
    >>> drop_duplicates_considering_index_and_values(df)
       A  B
    0  1  4
    1  2  5
    2  2  5
    2  3  6
    '''
    index_and_values = df.reset_index()    # Step 1: Convert the index to a column in the DataFrame; `.reset_index()` moves the original index into a column called 'index' and resets the original index to be 0...n
    deduplicated = index_and_values.drop_duplicates()    # Step 2: Drop duplicates considering both index and row values
    return df.iloc[deduplicated.index]    # Step 3: Use `.iloc` to select rows based on the deduplicated positions; using .index always refers to the index, even if there is a column called 'index' which in this case must be selected with df['index']


def _load_etf_prices_bq(daily_or_hourly: str) -> dict:
    '''This function loads the maturity data from the specified bigquery tables in the global ETFs list and concatenates them
    into a single dataframe.'''
    assert daily_or_hourly in ('daily', 'hourly')
    dataset_name = SP_ETF_DAILY_DATASET if daily_or_hourly == 'daily' else SP_ETF_HOURLY_DATASET

    etf_data = {}

    for table in ETFs:
        query = f'SELECT DISTINCT * FROM {dataset_name}.{table} ORDER BY Date DESC LIMIT {LIMIT}'
        df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')

        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True, drop=True)
        if daily_or_hourly == 'daily':
            df = drop_duplicates_considering_index_and_values(df)    # TODO: why is this only done when `daily_or_hourly` is 'daily'?
            etf_data[table] = df

    assert list(etf_data.keys()) == ETFs, f'Keys in ETF data: {list(etf_data.keys())}\nExpected ETFs: {ETFs}'
    return etf_data


def load_daily_etf_prices_bq():
    return _load_etf_prices_bq('daily')


def load_hourly_etf_prices_bq():
    return _load_etf_prices_bq('hourly')


def load_index_yields_bq():
    '''This function loads the index yield data from the specified bigquery tables in the global SP_INDEX_TABLES list and concatenates them
    into a single dataframe.'''
    index_data = {}

    for table in SP_INDEX_TABLES:
        query = f'SELECT DISTINCT * FROM {SP_INDEX_DATASET}.{table} ORDER BY date DESC LIMIT {LIMIT}'
        df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')

        assert list(df.columns) == ['date', 'ytw']

        df = df.drop_duplicates('date')
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df.sort_values('date', inplace=True, ascending=True)
        df.set_index('date', inplace=True, drop=True)

        df['ytw'] = df['ytw'] * 100    # convert to basis points

        daily_etf_weights_table_name = SP_INDEX_TABLE_TO_DAILY_ETF_WEIGHTS_TABLE[table]    # standardize names between maturity and yield data
        index_data[daily_etf_weights_table_name] = df

    assert list(index_data.keys()) == DAILY_ETF_WEIGHTS_TABLES, f'Keys in index data: {list(index_data.keys())}\nExpected table names: {DAILY_ETF_WEIGHTS_TABLES}'
    return index_data


def load_maturity_bq():
    '''This function loads the maturity data from the specified bigquery tables in the global `DAILY_ETF_WEIGHTS_TABLES` list and concatenates them
    into a single dataframe.'''
    maturity_data = {}

    for daily_etf_weights_table_name in DAILY_ETF_WEIGHTS_TABLES:
        query = f'SELECT DISTINCT * FROM {SP_MATURITY_DATASET}.{daily_etf_weights_table_name} ORDER BY effectivedate DESC LIMIT {LIMIT}'
        df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')

        assert list(df.columns) == ['effectivedate', 'weightedAverageMaturity', 'weightedAverageDuration']

        df['effectivedate'] = pd.to_datetime(df['effectivedate'], format='%Y-%m-%d')
        df = df.drop_duplicates('effectivedate')
        df.sort_values('effectivedate', inplace=True)
        df.set_index('effectivedate', inplace=True, drop=True)
        df = df[['weightedAverageMaturity']]
        maturity_data[daily_etf_weights_table_name] = df

    assert list(maturity_data.keys()) == DAILY_ETF_WEIGHTS_TABLES, f'Keys in maturity data: {list(maturity_data.keys())}\nExpected table names: {DAILY_ETF_WEIGHTS_TABLES}'

    maturity_data = pd.concat(maturity_data, axis=1)
    maturity_data.columns = maturity_data.columns.droplevel(-1)
    maturity_data = maturity_data.ffill()
    return maturity_data


def load_scaler_daily_bq():
    '''Loads the scaler parameters used in the sklearn StandardScaler to scale the input data for the daily Nelson-Siegel model
    during training.'''
    bq_client = bigquery.Client()
    query = f'SELECT DISTINCT * FROM {DATASET_NAME}.standardscaler_parameters_daily ORDER BY date DESC LIMIT {LIMIT}'
    df = bq_client.query(query).result().to_dataframe()
    return set_date_as_index(df, True)


def get_scalar_df(target_date: str) -> pd.DataFrame:
    '''This function retrieves the latest standard scalar coefficient from a BigQuery table
    (populated by the Cloud Function train_daily_yield_curve) which will be the standard
    scalar coefficient for the most recent business date.'''
    table_name = f'{PROJECT_ID}.{DATASET_NAME}.standardscaler_parameters_daily'
    scalar_df = sqltodf(f'SELECT * FROM `{table_name}`')
    scalar_df = set_date_as_index(scalar_df, False)
    return get_values_for_date_from_df(target_date, scalar_df, table_name)


def load_etf_models_bq():
    '''This function loads the maturity data from the specified bigquery tables in the global ETFs list and concatenates them
    into a single dataframe.'''
    model_data = {}

    for daily_etf_weights_table_name in DAILY_ETF_WEIGHTS_TABLES:
        query = f'SELECT * FROM {PROJECT_ID}.{DATASET_NAME}.{daily_etf_weights_table_name} ORDER BY Date DESC LIMIT {LIMIT}'
        df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True, drop=True)
        model_data[daily_etf_weights_table_name] = df

    assert list(model_data.keys()) == DAILY_ETF_WEIGHTS_TABLES
    return model_data


def load_shape_parameter(target_date: str) -> float:
    '''This function grabs the latest shape parameters for the Nelson-Siegel model.'''
    table_name = f'{PROJECT_ID}.{DATASET_NAME}.shape_parameters'
    shape_parameter_df = pd.read_gbq(f'SELECT date, L FROM `{table_name}`', project_id=PROJECT_ID, dialect='standard')
    shape_parameter_df = set_date_as_index(shape_parameter_df, False)
    return get_values_for_date_from_df(target_date, shape_parameter_df, table_name).iloc[0]    # `.iloc[0]` isolates the value


def current_datetime_as_string():
    '''Returns the current datetime as a formatted string.'''
    return datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S')


def upload_etf_prices_to_bq(quote_data: pd.DataFrame, table_id: str):
    '''Upload ETF prices to the specified BigQuery table. `quote_data` is a DataFrame containing ETF prices and
    `table_id` is the BigQuery table ID.'''
    client = bigquery.Client()

    upload_data = quote_data.copy()    # create a copy of the data so that downstream modifications to the dataframe do not modify the original dataframe
    upload_data['upload_time'] = current_datetime_as_string()

    # ensure all numeric columns are float
    for col in upload_data.columns:
        if col != 'upload_time': upload_data[col] = upload_data[col].astype('float')

    # define the schema for BigQuery
    schema = [bigquery.SchemaField('upload_time', 'DATETIME'),
              *[bigquery.SchemaField(col, 'FLOAT') for col in upload_data.columns if col != 'upload_time']]

    upload_data_dict = upload_data.to_dict(orient='records')    # convert the DataFrame to a list of dictionaries for BigQuery upload
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition='WRITE_APPEND')
    try:
        job = client.load_table_from_json(upload_data_dict, table_id, job_config=job_config)    # upload data to BigQuery
        job.result()    # wait for the job to complete
        print(f'Uploaded ETF prices to {table_id}')
    except Exception as e:
        print(f'Error uploading to BigQuery: {type(e)}: {e}')
        print(f'Data types: {upload_data.dtypes}')
        print(f'Data head: {upload_data.head()}')
        print(f'Columns: {upload_data.columns.tolist()}')
