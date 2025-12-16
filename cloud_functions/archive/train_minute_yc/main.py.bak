import numpy as np 
import pandas as pd
import datetime
from sklearn.linear_model import Ridge
import pytz
import requests
import redis
import pickle5 as pickle

from pandas.tseries.holiday import USFederalHolidayCalendar

from bigquery_utils import *

from yieldcurve import *

bday = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())

finnhub_apikey = 'c499cpiad3ieskgqq5lg'
PROJECT_ID = "eng-reactor-287421"

#Define model hyperparameters
window_size = 45
tau = 17
alpha = 0.001

#Define the best subset of funds for each index. These were selected using a separate notebook {to add reference}
best_funds = {'sp_7_12_year_national_amt_free_index':['TFI', 'PZA', 'ITM', 'MLN'],
'sp_high_quality_index':['PZA', 'TFI', 'ITM'],
'sp_high_quality_intermediate_managed_amt_free_index':['TFI', 'PZA', 'ITM', 'MLN'],
'sp_high_quality_short_intermediate_index':['PZA', 'TFI', 'ITM'],
'sp_high_quality_short_index':['PZA', 'HYMB', 'HYD', 'IBMM', 'MLN', 'ITM', 'TFI', 'SHYD', 'SHM']}

unique_funds = set([item for sublist in best_funds.values() for item in sublist])

indices = ['sp_7_12_year_national_amt_free_index',
 'sp_high_quality_intermediate_managed_amt_free_index',
 'sp_high_quality_index',
 'sp_high_quality_short_index',
 'sp_high_quality_short_intermediate_index']

def get_last_minute(timestamp: datetime.datetime):
    date = str(timestamp.date())
    hour = str(timestamp.hour).zfill(2)
    minute = str(timestamp.minute).zfill(2)
    
    
    datestring = '{} {}:{}'.format(date, hour, minute)
        
    return datetime.datetime.fromisoformat(datestring)

def load_etf_models_bq():
    '''
    This function loads the maturity data from the specified bigquery tables in the global etfs list and concatenates them
    into a single dataframe.
    '''
        
    client = bigquery.Client()
    model_data  = {}
    
    for table in indices:
        query = '''
                SELECT * FROM eng-reactor-287421.etf_model.{}
                '''.format(table)
        
        df = pd.read_gbq(query, project_id=project_id, dialect='standard')
        
        df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')

        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True, drop=True)
        model_data[table] = df 
        
    assert list(model_data.keys()) == indices
   
    
    return model_data

def getSchema_hourly_yield():
    schema = [bigquery.SchemaField("date", "DATETIME"),
              bigquery.SchemaField("const", "FLOAT"),
              bigquery.SchemaField("exponential", "FLOAT"),
              bigquery.SchemaField("laguerre", "FLOAT")]
    return schema

def uploadData(df, TABLE_ID, schema):
    client = bigquery.Client(project=PROJECT_ID, location="US")
    job_config = bigquery.LoadJobConfig(schema = schema, write_disposition="WRITE_APPEND")

    job = client.load_table_from_dataframe(df, TABLE_ID,job_config=job_config)

    try:
        job.result()
        print("Upload Successful")
    except Exception as e:
        print("Failed to Upload")
        raise e
        
def get_quote_finnhub(etf:str):
    '''
    This function gets the current price for the given ETF using the finnhub.io API. There is a maximum of 60 calls per minute. The request returns a json file with a number of variables. 'c' refers to the Current Price.
    
    Parameters: 
    etf:str
    
    '''
    response = requests.get('https://finnhub.io/api/v1/quote?symbol={}&token={}'.format(etf, finnhub_apikey))
    return response.json()['c']

def main(args):
    etf_data = load_daily_etf_prices_bq()

    maturity_df = load_maturity_bq()
    scaler_daily_parameters = load_scaler_daily_bq()
    index_data = load_index_yields_bq()
    
    etf_model_data = load_etf_models_bq()

    #Get quote data plus the time for right now
    tz = pytz.timezone('US/Eastern')
    timestamp = datetime.datetime.now(tz)
    timestamp = get_last_minute(timestamp)

    target_date = str(timestamp.date())

    quote_data = pd.DataFrame()
    for etf in unique_funds:
        quote_data[etf] = [get_quote_finnhub(etf)]

    #Get the most recent scaler and maturity data
    day_before_target_date = get_day_before(target_date, maturity_df)
    exponential_mean, exponential_std, laguerre_mean, laguerre_std = get_scaler_params(day_before_target_date, scaler_daily_parameters)
    maturity_dict = maturity_df.loc[day_before_target_date].to_dict()

    prev_close_data = []
    for fund in unique_funds:
        prev_close_data.append(etf_data[fund]['Close_{}'.format(fund)].loc[day_before_target_date:])

    prev_close_data = pd.concat(prev_close_data, axis=1)

    intraday_change = ((quote_data.values - prev_close_data) / prev_close_data) / 0.0001

    ### ETF MODEL
    predicted_ytw = pd.DataFrame()
        
    for table in indices:
        current_model_parameters = etf_model_data[table]
        current_features = list(current_model_parameters.columns.drop('constant'))
        current_funds = [x.split('_')[1] for x in current_features]

        #Get model parameters for the day
        try:
            parameters = current_model_parameters.loc[day_before_target_date]
        except:
            parameters = current_model_parameters.iloc[-1]
            
        parameters = parameters.to_numpy()
        
        current_intraday_subset = intraday_change[current_features]
        ones = np.ones((len(current_intraday_subset), 1))
        model_data = np.hstack((ones, current_intraday_subset.to_numpy()))

        predicted_ytw_change = (model_data*parameters).sum(axis=1)
        prev_ytw = index_data[table].loc[day_before_target_date].ytw
        prediction = prev_ytw + predicted_ytw_change

        predicted_ytw[table] = prediction

    yield_curve_df = predicted_ytw.T.rename({0:'ytw'}, axis = 1)
    yield_curve_df['Weighted_Maturity'] = yield_curve_df.index.map(maturity_dict).astype(float)

    X, y = get_NL_inputs(yield_curve_df, tau)
    X = scale_X(X, exponential_mean, exponential_std, laguerre_mean, laguerre_std)
    ridge = run_NL_ridge(X,y, scale=False, alpha=alpha)

    #Retrieve model parameters
    const = ridge.intercept_
    exponential = ridge.coef_[0]
    laguerre = ridge.coef_[1]

    coef_df = pd.DataFrame({'date':pd.to_datetime(timestamp),
        'const': const,
        'exponential':exponential,
        'laguerre':laguerre}, index=[0])
    uploadData(coef_df,
            "eng-reactor-287421.yield_curves.nelson_siegel_coef_minute",
            getSchema_hourly_yield())
    
    nelson=coef_df
    scalar=get_scalar_lbd() #Retrieves the scalar coefficient for the last business day from BigQuery. 
    scalar.set_index("date",drop=True,inplace=True)
    scalar.index = pd.to_datetime(scalar.index)
    nelson.set_index("date",drop=True,inplace=True)
    nelson.index = pd.to_datetime(nelson.index)
    for index, _ in nelson.iterrows():
        nelson_values=nelson.loc[[index]]
        scalar_values=scalar
        temp_dict={"nelson_values":nelson_values, "scalar_values":scalar_values}
        string_date = index.strftime('%Y-%m-%d:%H:%M')
        redis_client = redis.Redis(host='10.146.62.92', port=6379, db=0)
        value = pickle.dumps(temp_dict,protocol=pickle.HIGHEST_PROTOCOL)
        redis_client.set(string_date, value)
        return 'Done'        



