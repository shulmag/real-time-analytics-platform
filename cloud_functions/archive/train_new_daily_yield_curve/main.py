'''
 '''

import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import BDay
import datetime
import redis
import pickle5 as pickle


project_id = "eng-reactor-287421"
TABLE_ID_MODEL = "eng-reactor-287421.yield_curves_v2.nelson_siegel_coef_daily"
TABLE_ID_SCALER ="eng-reactor-287421.yield_curves_v2.standardscaler_parameters_daily" 

sp_index_tables = ['sp_12_22_year_national_amt_free_index',
                   'sp_15plus_year_national_amt_free_index',
                   'sp_7_12_year_national_amt_free_municipal_bond_index_yield',
                   'sp_muni_high_quality_index_yield',
                   'sp_high_quality_intermediate_managed_amt_free_municipal_bond_index_yield',
                   'sp_high_quality_short_intermediate_municipal_bond_index_yield',
                   'sp_high_quality_short_municipal_bond_index_yield',
                  'sp_long_term_national_amt_free_municipal_bond_index_yield']

sp_maturity_tables = ['sp_12_22_year_national_amt_free_index',
                      'sp_15plus_year_national_amt_free_index',
                      'sp_7_12_year_national_amt_free_index',
                      'sp_high_quality_index',
                      'sp_high_quality_intermediate_managed_amt_free_index',
                      'sp_high_quality_short_intermediate_index',
                      'sp_high_quality_short_index',
                      'sp_long_term_national_amt_free_municipal_bond_index_yield']


def load_index_data():
    '''
    This function load the S&P index data into a single dataframe.
    The output of the function is a datafraem containing the 
    yield to worst of all the indices in a single dataframe 
    
    Parameters:
    None 
    
    Return:
    Pandas dataframe
    '''
    client = bigquery.Client()
    index_data  = [] 
    for table in sp_index_tables:
        query = f'''SELECT * FROM `eng-reactor-287421.spBondIndex.{table}` order by date desc limit 1'''
        df = pd.read_gbq(query, project_id=project_id, dialect='standard')
        df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')
        df['ytw'] = df['ytw'] * 100
        df = df.drop_duplicates('date')
        df.set_index('date', inplace=True, drop=True)
        index_data.append(df)
    
    df = pd.concat(index_data, axis=1)
    df.columns = sp_maturity_tables
    df.ffill(inplace=True, axis=0)
    return df

def load_maturity_data():
    '''
    This function load the S&P maturity data into a single dataframe.
    The output of the function is a datafraem containing the 
    weighted average maturities of all the indices in a single dataframe 
    
    Parameters:
    None 
    
    Return:
    Pandas dataframe
    '''
    client = bigquery.Client()
    maturity_data  = []

    for table in sp_maturity_tables:
        query = f'SELECT * FROM `eng-reactor-287421.spBondIndexMaturities.{table}` order by effectivedate desc limit 1;'
        df = pd.read_gbq(query, project_id=project_id, dialect='standard')        
        df['effectivedate'] = pd.to_datetime(df['effectivedate'], format = '%Y-%m-%d')
        df = df.drop_duplicates('effectivedate')
        df.set_index('effectivedate', inplace=True, drop=True)
        
        df = df[['weightedAverageMaturity']]
        maturity_data.append(df) 
        
    df = pd.concat(maturity_data, axis=1)
    df.columns = sp_maturity_tables
    df.ffill(inplace=True, axis=0)
    return df

def get_maturity_dict(maturity_df, date):
    '''
    This function creates a dictonary with the index name
    being the key and the weighted average maturities as the
    values
    
    Parameters:
    maturity_df: pandas dataframe
    date: string
    
    Return
    Dictionary
    '''
    temp_df = maturity_df.loc[date].T
    temp_dict = dict(zip(temp_df.index, temp_df.values))
    return temp_dict

def get_yield_curve_maturity_df(index_data, date, maturity_dict):
    '''
    This function creates a dataframe that contains the yield to worst
    and weighted average maturity for a specific date.
    
    Parameters
    index_data: pandas dataframe
    date: string
    maturity_dict: dictionary
    
    Return:
    Dataframe
    '''
    df = pd.DataFrame(index_data.loc[date])
    df.columns = ['ytw']
    df['Weighted_Maturity'] = df.index.map(maturity_dict)
    return df

def decay_transformation(t:np.array, L:float):
    '''
    This function takes a numpy array of maturities and a shape parameter. 
    It returns the exponential function calculated from those values.
    
    Parameters:
    t:np.array
    L:float
    
    '''
    return L*(1-np.exp(-t/L))/t

def laguerre_transformation(t:np.array, L:float):
    '''
    This function takes a numpy array of maturities and a shape parameter. 
    It returns the laguerre function calculated from those values.
    
    Parameters:
    t:np.array
    L:float
    '''
    return (L*(1-np.exp(-t/L))/t) -np.exp(-t/L)

def get_model_inputs(yield_curve_maturity_df, L):
    '''
    This function creates the inputs for the regression model.
    The inputs are created using the exponential and laguerre transform
    
    Parameter:
    yield_curve_maturity_df: pandas dataframe
    L: int
    '''
    temp_df = yield_curve_maturity_df.copy()
    temp_df['X1'] = decay_transformation(temp_df['Weighted_Maturity'], L)
    temp_df['X2'] = laguerre_transformation(temp_df['Weighted_Maturity'], L)
    
    X = temp_df[['X1', 'X2']]
    y = temp_df['ytw']
    
    return X, y

def train_model(X, Y):
    '''
    This function train a regression model to estimate the Nelson
    Seigle coefficients. The inputs to the model is
    
    Parameters:
    X: np.array()
    Y: float
    '''
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model = Ridge(alpha=0.001, random_state = 1).fit(X , Y)

    return scaler, model

def getSchema_model():
    '''
    This function returns the schema required for the bigquery table storing the nelson siegel coefficients. 
    '''

    schema = [
        bigquery.SchemaField("date", "DATE", "REQUIRED"),
        bigquery.SchemaField("const","FLOAT", "REQUIRED"),
        bigquery.SchemaField("exponential","FLOAT", "REQUIRED"),
        bigquery.SchemaField("laguerre","FLOAT", "REQUIRED")]
    return schema

def getSchema_scaler():
    '''
    This function returns the schema required for the bigquery table storing 
    the sklearn StandardScaler's parameters siegel coefficients.
    '''

    schema = [
        bigquery.SchemaField("date", "DATE", "REQUIRED"),
        bigquery.SchemaField("exponential_mean","FLOAT", "REQUIRED"),
        bigquery.SchemaField("exponential_std","FLOAT", "REQUIRED"),
        bigquery.SchemaField("laguerre_mean","FLOAT", "REQUIRED"),
        bigquery.SchemaField("laguerre_std","FLOAT", "REQUIRED")]
    return schema

def uploadData(df: pd.DataFrame, TABLE_ID:str):
    '''
    This function upload the coefficient and scalar dataframe
    to BigQuery.
    
    Parameters
    df:pd.DataFrame
    TABLE_ID:str path of the bigquery table to upload to
    '''
    
    client = bigquery.Client()
    
    if TABLE_ID == TABLE_ID_MODEL:
        job_config = bigquery.LoadJobConfig(schema = getSchema_model(), write_disposition="WRITE_APPEND")
    elif TABLE_ID == TABLE_ID_SCALER:
        job_config = bigquery.LoadJobConfig(schema = getSchema_scaler(), write_disposition="WRITE_APPEND")
    else:
        raise ValueError

    job = client.load_table_from_dataframe(df, TABLE_ID,job_config=job_config)

    try:
        job.result()
        print("Upload Successful")
    except Exception as e:
        print("Failed to Upload")
        raise e

def load_shape_parameter():
    '''
    This function grabs the latest shape parameters for the Nelson Siegel

    parameters: None

    return: Float
    '''
    query = ''' SELECT L FROM `eng-reactor-287421.yield_curves_v2.shape_parameters` order by date desc limit 1 '''
    df = pd.read_gbq(query, project_id=project_id, dialect='standard')
    return df.loc[0].values[0]

def main(args):
    maturity_data = load_maturity_data()
    index_data = load_index_data()
    L = load_shape_parameter()
    coefficient_df = pd.DataFrame()
    scaler_df = pd.DataFrame()
    
    #Creating a datframe to send inputs to the model
    for target_date in list(maturity_data.index.astype(str)):
        print(f"Calculating the coefficients for {target_date}")
        maturity_dict = get_maturity_dict(maturity_data, target_date)
        yield_curve_maturity_df = get_yield_curve_maturity_df(index_data, target_date, maturity_dict)

        #Creating the inputs for the model
        X,Y = get_model_inputs(yield_curve_maturity_df, L)
        scaler, model = train_model(X,Y)

        #Retrieve model parameters
        const = model.intercept_
        exponential = model.coef_[0]
        laguerre = model.coef_[1]

        #Retrieve scaler parameters, used to standardize the data
        exponential_mean = scaler.mean_[0]
        exponential_std = np.sqrt(scaler.var_[0])
        laguerre_mean = scaler.mean_[1]
        laguerre_std = np.sqrt(scaler.var_[1])

        #Convert date to datetime object
        date = pd.to_datetime(target_date).date()

        temp_coefficient_df = pd.DataFrame({'date':date,
                                       'const': const,
                                       'exponential':exponential,
                                       'laguerre':laguerre}, index=[0])

        temp_scaler_df = pd.DataFrame({'date':date,
                               'exponential_mean': exponential_mean,
                               'exponential_std':exponential_std,
                               'laguerre_mean':laguerre_mean,
                               'laguerre_std':laguerre_std}, index=[0])

        coefficient_df = coefficient_df.append(temp_coefficient_df)
        scaler_df = scaler_df.append(temp_scaler_df)    
    
    print('Uploading Data')
    uploadData(coefficient_df, TABLE_ID_MODEL) 
    uploadData(scaler_df, TABLE_ID_SCALER)

    print("Uploading data to redis")
    string_date = date
    string_date = string_date +  BDay(1)
    string_date = string_date.strftime('%Y-%m-%d')
    
    coefficient_df.reset_index(inplace=True, drop=True)
    scaler_df.reset_index(inplace=True, drop=True)
    nelson_values = coefficient_df.set_index("date")
    scalar_values = scaler_df.set_index("date")

    temp_dict = {"nelson_values":nelson_values, "scalar_values":scalar_values, "shape_parameter":L}
    redis_client = redis.Redis(host='10.227.69.60', port=6379, db=0)
    value = pickle.dumps(temp_dict,protocol=pickle.HIGHEST_PROTOCOL)
    redis_client.set(string_date, value)

    return "SUCCESS"