
import numpy as np 
import pandas as pd
from sklearn.linear_model import Lasso
from warnings import filterwarnings
from bigquery_utils import *

filterwarnings('ignore')
pd.set_option('mode.chained_assignment',None)

best_funds = {'sp_7_12_year_national_amt_free_index':['TFI', 'PZA', 'ITM', 'MLN'],
'sp_high_quality_index':['PZA', 'TFI', 'ITM'],
'sp_high_quality_intermediate_managed_amt_free_index':['TFI', 'PZA', 'ITM', 'MLN'],
'sp_high_quality_short_intermediate_index':['PZA', 'TFI', 'ITM'],
'sp_high_quality_short_index':['PZA', 'HYMB', 'HYD', 'IBMM', 'MLN', 'ITM', 'TFI', 'SHYD', 'SHM']}

best_lambdas = {'sp_7_12_year_national_amt_free_index': 1.0,
 'sp_high_quality_index': 1.0,
 'sp_high_quality_intermediate_managed_amt_free_index': 1.0,
 'sp_high_quality_short_intermediate_index': 1.0,
 'sp_high_quality_short_index': 1.0}    
    
PROJECT_ID = "eng-reactor-287421"

dataset_name = 'etf_model'

tableNames = ['sp_7_12_year_national_amt_free_index',
 'sp_high_quality_intermediate_managed_amt_free_index',
 'sp_high_quality_index',
 'sp_high_quality_short_index',
 'sp_high_quality_short_intermediate_index']

#The number of days of prior data to train the etf model on; a previously tuned hyperparameter, equals roughly 2 months (working days)
train_window_size = 45

def preprocess_data(index_data:dict, etf_data:dict, index_name:str, etf_names:list, date_start='2020-05', var='Close'):
    '''
    This function takes as input the loaded S&P index data and ETF data from bigquery, which is stored as a dictionary of dataframes. It also takes the name of a single S&P index and a list of ETFs that are relevant to predicting that index. It then merges this data into a single dataframe, calculating the pct_change in ETF prices in basis points and the change in index ytw in basis points. This is done, by default, for observations after 2020-May and for the Close prices of the ETFs. The merged result is returned 
    
    Parameters: 
    index_data:dict
    etf_data:dict
    index_name:str
    etf_names:list
    date_start:str
    var:str
    '''
    
    data = []
    
    #preprocess etf data by retrieving etfs of interest and calculating pct_change in basis points
    for etf_name in etf_names:
        etf = etf_data[etf_name].copy()
        etf = etf.drop_duplicates()
        data.append(etf['{}_{}'.format(var,etf_name)].pct_change()/0.0001)
    etf = pd.concat(data, axis = 1)
    
    #preprocess index data by first-differencing ytw
    index = index_data[index_name].copy()
    index['ytw_diff'] =index['ytw'].diff()
    
    #merge etf and index date
    temp_df = pd.merge(etf, index, left_index=True, right_index=True).loc[date_start:]
    return temp_df.dropna()

def getSchema_etf(coefficient_df:pd.DataFrame):
    '''
    Gets the bq schema to upload the data to the bq table containing the coefficients for the linear model using ETF prices to predict index yield, for each index. To recycle the function, we take the coefficient dataframe for the models as input to retrieve the feature names, with an additional column indexing the date the model is for. 
    
    Parameters:
    coefficient_df: pd.DataFrame 
    
    '''
    schema = [bigquery.SchemaField("Date", "DATE")] + [bigquery.SchemaField(x,"FLOAT") for x in coefficient_df.columns]
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

def main(args):
    '''
    Main function training the daily etf model. We first load the index and etf data, then for each S&P index, we train a model using the previously identified optimal subset of ETFs to predict yields. Training data size is equal to the window size, also previously identified. Since it is a linear model, we can afford to save the coefficients to a bigquery table. This function is scheduled to run everyday.
    
    '''
    index_data = load_index_yields_bq()
    etf_data = load_daily_etf_prices_bq()



    for current_index in list(best_funds.keys()): 
        #To save the coefficients 
        results_dict = {}

        #Load data and hyperparameters 
        current_best_lambda = best_lambdas[current_index]
        current_best_funds = best_funds[current_index]
        current_data = preprocess_data(index_data, etf_data, current_index, current_best_funds, date_start='2020-05')

        #Get X and Y data
        X = current_data.drop(['ytw','ytw_diff'],axis=1)
        y = current_data['ytw_diff']
        X_cols = list(X.columns)

        #Training data size is the window size 
        X_train = X.iloc[-train_window_size:, :]
        y_train = y.iloc[-train_window_size:]

        assert len(X_train) == len(y_train)
        
        #Get the date to index the model and train the model 
        date = X_train.index.min().date().isoformat()
        lasso = Lasso(alpha = current_best_lambda, random_state=1, max_iter=5000).fit(X_train, y_train)

        #Save the coefficients to one row dataframe and append it to bigquery
        columns = ['constant'] + X_cols
        coefficients = np.hstack([lasso.intercept_,lasso.coef_])
        results_dict[date] = dict(zip(columns ,coefficients))
        coefficient_df = pd.DataFrame(results_dict).T
        coefficient_df.index = pd.to_datetime(coefficient_df.index)
        
        schema = getSchema_etf(coefficient_df)
        coefficient_df = coefficient_df.reset_index(drop=False).rename({'index':'Date'}, axis=1)

        TABLE_ID = "eng-reactor-287421.etf_model" + '.' + current_index

        uploadData(coefficient_df, TABLE_ID, schema)
        
    return 'Done'