import numpy as np 
import pandas as pd
import requests
import datetime
from bigquery_utils import *
from yieldcurve import *
from tqdm import tqdm
import time

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/shayaan/ficc/ahmad_creds.json'


# finnhub_apikey = 'c499cpiad3ieskgqq5lg'
PROJECT_ID = 'eng-reactor-287421'
DATASET_NAME = 'yield_curves_v2'

START_DATE = datetime.datetime(2023, 10, 23).date()
END_DATE = datetime.datetime(2023, 10, 23).date()

# model hyperparameters
TAU = 5
ALPHA = 0.001

# best subset of funds for each index selected from experiments in TODO: add notebook containing experiments
best_funds = {'sp_12_22_year_national_amt_free_index' : ['FMHI', 'MUB'], 
              'sp_15plus_year_national_amt_free_index': ['FMHI', 'MLN', 'MUB', 'TFI', 'SUB', 'SHYD', 'HYMB', 'HYD'], 
              'sp_7_12_year_national_amt_free_index': ['TFI', 'PZA', 'ITM', 'MLN'], 
              'sp_high_quality_index': ['PZA', 'TFI', 'ITM'], 
              'sp_high_quality_intermediate_managed_amt_free_index': ['TFI', 'PZA', 'ITM', 'MLN'], 
              'sp_high_quality_short_intermediate_index': ['PZA', 'TFI', 'ITM'], 
              'sp_high_quality_short_index': ['PZA', 'HYMB', 'HYD', 'MLN', 'ITM', 'TFI', 'SHYD', 'SHM'], 
              'sp_long_term_national_amt_free_municipal_bond_index_yield': ['FMHI', 'MLN', 'MUB', 'SUB']}

unique_funds = set([fund for funds_list in best_funds.values() for fund in funds_list])

indices = list(best_funds.keys())


def get_last_minute(timestamp: datetime.datetime):
    date = str(timestamp.date())
    hour = str(timestamp.hour).zfill(2)
    minute = str(timestamp.minute).zfill(2)
    datestring = f'{date} {hour}:{minute}'
    return datetime.fromisoformat(datestring)


def load_etf_models_bq():
    '''This function loads the maturity data from the specified bigquery tables in the global ETFs list and concatenates them
    into a single dataframe.'''
    model_data  = {}
    for table in indices:
        query = f'''SELECT * FROM eng-reactor-287421.{DATASET_NAME}.{table}'''
        df = pd.read_gbq(query, project_id=project_id, dialect='standard')
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True, drop=True)
        model_data[table] = df 
    assert list(model_data.keys()) == indices
    return model_data


def get_schema_minute_yield():
    schema = [bigquery.SchemaField('date', 'DATETIME'),
              bigquery.SchemaField('const', 'FLOAT'),
              bigquery.SchemaField('exponential', 'FLOAT'),
              bigquery.SchemaField('laguerre', 'FLOAT')]
    return schema


def upload_data(df, table_id, schema):
    client = bigquery.Client(project=PROJECT_ID, location='US')
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition='WRITE_APPEND')
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    try:
        job.result()
        print('Upload Successful')
    except Exception as e:
        print('Failed to Upload')
        raise e


def get_min_etf_data(etf, start_date):
    print(etf)
    api_key  = '_x7jC7IqbGwJiuGLQpO45H6tfL2bFgKC'
    limit = 50000
    # query = f'https://api.polygon.io/v2/aggs/ticker/{etf}/range/1/minute/{start_date}/{start_date}?adjusted=true&sort=asc&limit=50000&apiKey=_x7jC7IqbGwJiuGLQpO45H6tfL2bFgKC'
    query = f'https://api.polygon.io/v2/aggs/ticker/{etf}/range/1/minute/{start_date}/{start_date}?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}'
    print(query)
    response = requests.get(query)
    response = response.json()
    print(response)
    response = response['results']
    response = pd.DataFrame(response)
    response['t'] = pd.to_datetime(response['t'], unit='ms')
    response['t'] = response['t'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    return response['c'].values


def load_shape_parameter():
    query = f''' SELECT * FROM `eng-reactor-287421.{DATASET_NAME}.shape_parameters` order by date'''
    df = pd.read_gbq(query, project_id=project_id, dialect='standard')
    df.Date = pd.to_datetime(df.Date)
    df.set_index('Date', drop=True, inplace=True)
    return df


def get_date_list(start_dt, end_dt):
    def daterange(date1, date2):
        for day_count in range(int((date2 - date1).days) + 1):
            yield date1 + datetime.timedelta(day_count)

    weekdays = [5, 6]
    date_list = []
    for dt in daterange(start_dt, end_dt):
        if dt.weekday() not in weekdays:
            date_list.append(dt)
    return date_list


def create_index(current_date):
    current_date = current_date.strftime('%Y-%m-%d')
    current_date = current_date + ' 09:30:00'
    current_date = datetime.datetime.strptime(current_date,'%Y-%m-%d %H:%M:%S')
    index_list = []
    for i in range(393):
        index_list.append(current_date)
        current_date += datetime.timedelta(minutes=1)
    return index_list


def main(args):
    etf_data = load_daily_etf_prices_bq()
    maturity_df = load_maturity_bq()
    maturity_df.ffill(inplace=True, axis=0)
    maturity_df.dropna(inplace=True)
    scaler_daily_parameters = load_scaler_daily_bq()
    index_data = load_index_yields_bq()
    etf_model_data = load_etf_models_bq()
    
    # get quote data plus the time for right now
    date_list = get_date_list(START_DATE, END_DATE)
    for current_date in tqdm(date_list):
        # try:
        coefficient_df = pd.DataFrame()
        quote_data = {}
        df_index = create_index(current_date)
        
        for etf in unique_funds:
            quote_data[etf] = get_min_etf_data(etf,current_date)
            time.sleep(20)        # have `time.sleep(15)` so that we do not want to keep hitting the API, otherwise they will block our access
        quote_data = pd.DataFrame.from_dict(quote_data, orient='index').T
        quote_data.ffill(inplace=True)
        
        df_index = df_index[:len(quote_data)]
        quote_data.index = df_index
    
        # get the most recent scaler and maturity data
        day_before_target_date = get_day_before(current_date, maturity_df)
        exponential_mean, exponential_std, laguerre_mean, laguerre_std = get_scaler_params(day_before_target_date, scaler_daily_parameters)
        maturity_dict = maturity_df.loc[day_before_target_date].to_dict()

        prev_close_data = []
        for fund in unique_funds:
            prev_close_data.append(etf_data[fund][f'Close_{fund}'].loc[day_before_target_date])

        intraday_change = ((quote_data - prev_close_data) / prev_close_data) / 0.0001
        
        ### ETF model
        for index, row in intraday_change.iterrows():
            predicted_ytw = pd.DataFrame()
            for table in indices:
                current_model_parameters = etf_model_data[table]
                current_features = list(current_model_parameters.columns.drop('constant'))
                current_features = [i.split('_')[-1] for i in current_features]

                # get model parameters for the day
                try:
                    parameters = current_model_parameters.loc[day_before_target_date]
                except:
                    parameters = current_model_parameters.iloc[-1]
                parameters = parameters.to_numpy()
                
                current_intraday_subset = row[current_features]
                current_intraday_subset = current_intraday_subset.values
                current_intraday_subset = current_intraday_subset.reshape(1,len(current_intraday_subset))
                ones = np.ones((len(current_intraday_subset), 1))
                model_data = np.hstack((ones, current_intraday_subset ))

                predicted_ytw_change = (model_data*parameters).sum(axis=1)
                prev_ytw = index_data[table].loc[day_before_target_date].ytw
                prediction = prev_ytw + predicted_ytw_change
                predicted_ytw[table] = prediction

            yield_curve_df = predicted_ytw.T.rename({0: 'ytw'}, axis=1)
            yield_curve_df['Weighted_Maturity'] = yield_curve_df.index.map(maturity_dict).astype(float)
            # tau = shape_parameter.loc[current_date.strftime('%Y-%m-%d'), 'L']
            X, y = get_NL_inputs(yield_curve_df, TAU)
            X = scale_X(X, exponential_mean, exponential_std, laguerre_mean, laguerre_std)
            ridge = run_NL_ridge(X,y, scale=False, alpha=ALPHA)

            # Retrieve model parameters
            const = ridge.intercept_
            exponential = ridge.coef_[0]
            laguerre = ridge.coef_[1]

            coef_df = pd.DataFrame({'date': index,
                                    'const': const,
                                    'exponential': exponential,
                                    'laguerre': laguerre}, index=[0])
            coefficient_df = pd.concat([coefficient_df, coef_df])
        print(coefficient_df)
        coefficient_df.to_pickle('ns_20230=_06_23.pkl')

        # upload_data(coefficient_df, f'eng-reactor-287421.{DATASET_NAME}.nelson_siegel_coef_minute_temp', get_schema_minute_yield()) 
        
        # except Exception as e:
        #     print(f'Failed for {current_date}')
        #     with open('date_errors.txt', 'w') as f:
        #         f.write(f'{current_date}:{e}\n') 


if __name__ == '__main__':
    main('Test')