import numpy as np
import pandas as pd
import os
from google.cloud import bigquery
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
   
project_id = "eng-reactor-287421"
sp_maturity_dataset = 'spBondIndexMaturities'
sp_index_dataset = 'spBondIndex'

#Specifying table names in bigquery
sp_index_tables = ['sp_12_22_year_national_amt_free_index',
                   'sp_15plus_year_national_amt_free_index',
                   'sp_7_12_year_national_amt_free_municipal_bond_index_yield',
                   'sp_muni_high_quality_index_yield',
                   'sp_high_quality_intermediate_managed_amt_free_municipal_bond_index_yield',
                   'sp_high_quality_short_intermediate_municipal_bond_index_yield',
                   'sp_high_quality_short_municipal_bond_index_yield']

sp_maturity_tables = ['sp_12_22_year_national_amt_free_index',
                      'sp_15plus_year_national_amt_free_index',
                      'sp_7_12_year_national_amt_free_index',
                      'sp_high_quality_index',
                      'sp_high_quality_intermediate_managed_amt_free_index',
                      'sp_high_quality_short_intermediate_index',
                      'sp_high_quality_short_index']

name_dict = dict(zip(sp_index_tables,sp_maturity_tables))

###Functions to transform maturities into the components in the nelson-siegel model
def decay_transformation(t:np.array, L:float):
    '''
    This function takes a numpy array of maturities (or a single float) and a shape parameter. It returns the exponential function
    calculated from those values. This is the first feature of the nelson-siegel model.
    
    Parameters:
    t:np.array
    L:float
    '''
    return L*(1-np.exp(-t/L))/t

###Functions to preprocess data and fit model
def get_maturity_dict(maturity_data:pd.DataFrame, date:str):
    '''
    This function takes as input the dataframe returned by the load_maturity_bq function, retrieves the row corresponding to the date of
    interesting and converts that row into a dictionary, with key-value pairs corresponding to S&P index names and their maturities.
    
    Parameters:
    maturity_data:pd.DataFrame
    date:str
    '''
    
    temp_df = maturity_data.loc[date]
    maturity_dict = dict(zip(temp_df.index, temp_df.values))
    assert len(maturity_dict) != 0, "No data for given date found!"
    return maturity_dict

def get_yield_curve_df(index_data:pd.DataFrame, date:str, maturity_dict:dict):
    '''
    This function takes as input the dataframe returned by the load_index_yields_bq and retrieves the yields corresponding to the specified
    date. It then uses a dictionary of maturities for each index to map the right maturity to each index. The output is a dataframe with a
    column for maturity and a column for ytw, indexed by index names. 
    
    Parameters:
    index_data:pd.DataFrame
    date:str
    maturity_dict:dict
    '''
    
    yield_curve_df = pd.DataFrame(index_data.loc[date])
    yield_curve_df.columns = ['ytw']
    yield_curve_df['Weighted_Maturity'] = yield_curve_df.index.map(maturity_dict)
    return yield_curve_df 

def get_NL_inputs(yield_curve_df:pd.DataFrame, tau:float):
    '''
    This function takes the output of the get_yield_curve_df function alongside a shape parameter to generate the features of the nelson
    siegel yield curve. It then creates the exponential feature and the laguerre feature using the decay_transformation and
    laguerre_transformation corresponding to the given shape parameter, and returns the features (X1,X2) and the labels (ytw).
    
    Parameters:
    yield_curve_df:pd.DataFrame
    L:float
    '''
    temp_df = yield_curve_df.copy()
    
    temp_df['X1'] = tau*(1-np.exp(-temp_df['Weighted_Maturity']/tau))/temp_df['Weighted_Maturity']
    
    temp_df['X2'] = (tau*(1-np.exp(-temp_df['Weighted_Maturity']/tau))/temp_df['Weighted_Maturity']) -np.exp(-temp_df['Weighted_Maturity']/tau)
    
    X = temp_df[['X1', 'X2']]
    y = temp_df['ytw']
    
    return X, y 

def run_NL_ridge(X:pd.DataFrame, y:pd.Series, alpha = 0.001, scale=True):
    '''
    This function takes the X and Y values and runs a ridge regression to estimate the nelson-siegel yield curve model. If the sklearn
    StandardScaler is used, then the scaler object is returned alongside the model object so that the scaler parameters can be saved as
    well.
    
    Parameters:
    X: pd.DataFrame/np.array
    y: pd.Series/np.array
    alpha:float
    scale:bool
    '''
    
    if scale: 
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        ridge = Ridge(alpha=alpha, random_state = 1).fit(X , y)
        return scaler, ridge
    else: 
        ridge = Ridge(alpha=alpha, random_state = 1).fit(X , y)
        return ridge   
    
def load_scaler_daily_bq():
    '''
    This function loads the scaler parameters used in the sklearn StandardScaler to scale the input data for the daily nelson-siegel model
    during training.
    '''
    bq_client = bigquery.Client()
    query = '''
            SELECT DISTINCT * FROM yield_curves_v2.standardscaler_parameters_daily
            '''
    query_job = bq_client.query(query)
    df = query_job.result().to_dataframe()
    df['date'] = pd.to_datetime(df['date'])    
    df = df.sort_values(by='date',ascending=True).set_index('date', drop=True)
    return df

def scale_X(X, exponential_mean, exponential_std, laguerre_mean, laguerre_std):
    X['X1'] = (X['X1'] - exponential_mean)/exponential_std
    X['X2'] = (X['X2'] - laguerre_mean)/laguerre_std
    return X

def get_day_before(target_date, maturity_df):
    try:
        if len(maturity_df.loc[:target_date]) == len(maturity_df):
            #if target_date is somehow beyond the available sample, then use the most recent entry
            day_before_target_date = str(maturity_df.iloc[-1].name.date())
        elif len(maturity_df.loc[:target_date]) <= 1:
            #if target date is before the available sample, then use the oldest entry
            day_before_target_date = str(maturity_df.iloc[0].name.date())
        else:
            #else, the day before the target_date is the entry before the target date
            day_before_target_date = str(maturity_df.loc[:target_date].iloc[-2].name.date())
    except:
        day_before_target_date = str(maturity_df.iloc[-1].name.date())

    return day_before_target_date

def get_scaler_params(date, scaler_daily_parameters):
    exponential_mean, exponential_std, laguerre_mean, laguerre_std = scaler_daily_parameters.loc[date].values
    return exponential_mean, exponential_std, laguerre_mean, laguerre_std

#load the actual index data for use later
def combine_index_data(index_data):
    index_df = []
    for item in index_data.keys():
        temp_df = index_data.get(item)['ytw']
        temp_df.name = item
        index_df.append(temp_df)
    index_df = pd.concat(index_df, axis=1).dropna()
    index_df.index = pd.to_datetime(index_df.index)
    return index_df.sort_index(ascending=True)

def preprocess_data(index_data, etf_data, index_name, etf_names, date_start='2020-05', var='Close'):
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