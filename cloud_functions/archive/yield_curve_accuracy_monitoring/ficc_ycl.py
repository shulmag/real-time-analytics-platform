'''
 '''

import numpy as np 
import pandas as pd
import pickle5 as pickle
from google.cloud import bigquery

PROJECT_ID = "eng-reactor-287421"


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

def laguerre_transformation(t, L):
    '''
    This function takes a numpy array of maturities (or a single float) and a shape parameter. It returns the laguerre function
    calculated from those values. This is the second feature of the nelson-siegel model.
    
    Parameters:
    t:np.array
    L:float
    '''
    return (L*(1-np.exp(-t/L))/t) -np.exp(-t/L)


###Functions to load the trained daily yield curve model parameters and daily scaler parameters from bigquery
def load_nelson_siegel_daily_bq(target_date):
    '''
    This function loads the model coefficients for the daily nelson-siegel model from bigquery.
    '''
    bq_client = bigquery.Client()
    query = '''
            SELECT * FROM yield_curves_v2.nelson_siegel_coef_daily WHERE date = '{}' ORDER BY date ASC
            '''.format(pd.to_datetime(target_date).date()) #date is stored as datetime, so we must filter in bigquery using a datetime target_date

    query_job = bq_client.query(query)
    df = query_job.result().to_dataframe()
    return df

def load_scaler_daily_bq(target_date):
    '''
    This function loads the scaler parameters used in the sklearn StandardScaler to scale the input data for the daily nelson-siegel model
    during training.
    '''
    bq_client = bigquery.Client()
    query = '''
            SELECT * FROM yield_curves_v2.standardscaler_parameters_daily WHERE date = '{}' ORDER BY date ASC
            '''.format(pd.to_datetime(target_date).date()) #date is stored as datetime, so we must filter in bigquery using a datetime target_date

    query_job = bq_client.query(query)
    df = query_job.result().to_dataframe()   
    return df


def load_shape_parameter():
    query = ''' SELECT L FROM `eng-reactor-287421.yield_curves_v2.shape_parameters` order by Date desc limit 1'''
    df = pd.read_gbq(query, project_id='eng-reactor-287421', dialect='standard')
    return df.loc[0].values[0]

###Functions used for prediction  Function to 
def get_scaled_features(t:np.array, exponential_mean:float, exponential_std:float, laguerre_mean:float, laguerre_std:float):
    
    '''
    This function takes as input the parameters loaded from the scaler parameter table in bigquery on a given day, alongside an array (or a
    single float) value to be scaled as input to make predictions. It then manually recreate the transformations from the sklearn
    StandardScaler used to scale data in training by first creating the exponential and laguerre functions then scaling them.
    
    Parameters:
    t:np.array
    exponential_mean:float
    exponential_std:float
    laguerre_mean:float
    laguerre_std:float
    '''
    L = load_shape_parameter()
    X1 = (decay_transformation(t, L) - exponential_mean)/exponential_std 
    X2 = (laguerre_transformation(t, L) - laguerre_mean)/laguerre_std 
    return X1, X2

def predict_ytw(t:np.array, const:float , exponential:float , laguerre:float , exponential_mean:float , exponential_std:float , laguerre_mean:float , laguerre_std:float ):
    '''
    This is a wrapper function that takes the prediction inputs, the scaler parameters and the model parameters from a given day. It then
    scales the input using the get_scaled_features function to obtain the model inputs, and predicts the yield-to-worst implied by the
    nelson-siegel model on that day. Because the nelson-siegel model is linear, we can do a simple calculation. 
    
    Parameters:
    t:np.array
    const:float 
    exponential:float 
    laguerre:float 
    exponential_mean:float
    exponential_std:float
    laguerre_mean:float
    laguerre_std:float
    '''
    
    X1, X2 = get_scaled_features(t, exponential_mean, exponential_std, laguerre_mean, laguerre_std)
    return const + exponential*X1 + laguerre*X2


def yield_main(request):
    '''
    This is the main function takes as input a json containing two arguments: the maturity we want the yield-to-worst for and the target 
    ate from which we want the yield curve used in the ytw calculations to be from. There are several conditional statements to deal with
    different types of exceptions. 
    
    The cloud function returns a json containing the status (Failed or Success), the error message (if any)
    and the result (nan if calculation was unsuccessful).
    '''
    
    error = ''

    #Experienced some inconsistency in passing arguments to the HTTP request, it seems that using a combination of args and get_json is
    #recommende by GCP in their stock code. If neither works in reading the arguments, the json returns an error message.
    t = float(request['maturity'])
    target_date = request['target_date']

    
    #The maturity value cannot be <= 0, hence an error is returned
    if t <= 0:
        return {'status':'Failed', 'error':'Enter a valid maturity greater than 0', 'result':np.nan}

    #If no target_date is provided, an error is also returned 
    if not target_date:
        return {'status':'Failed', 'error':'Enter a valid target_date from available dates in the format YYYY-MM-DD', 'result':np.nan}
    
    #If a target_date is provided but it is in an invalid format, then the correct values from the model and scaler parameters cannot be
    #retrieved, and an error is also returned.
    try:
        nelson_siegel_daily_coef = load_nelson_siegel_daily_bq(target_date).drop('date',axis=1)
        scaler_daily_parameters = load_scaler_daily_bq(target_date).drop('date',axis=1)
    except Exception as e:
        raise e
        return {'status':'Failed', 'error':'Failed to load data from bigquery', 'result':np.nan} 
    
    #If the retrieved parameters are a pd.Series, then there are no duplicates. If they are a dataframe, then there are duplicates and we
    #take the first row. If they are anything else, something wrong has occured, likely that no data was retrieved 
    if len(nelson_siegel_daily_coef)==1:
        const, exponential, laguerre = nelson_siegel_daily_coef.values[0]
    elif len(nelson_siegel_daily_coef)>1:
        error = 'Multiple rows for target date in nelson_siegel_coef_daily, taking first one. Check bigquery table.'
        const, exponential, laguerre = nelson_siegel_daily_coef.iloc[0, :]
    else:
        return {'status':'Failed', 'error':'Target date not in nelson_siegel_coef_daily', 'result':np.nan}
   
    if len(scaler_daily_parameters)==1:
        exponential_mean, exponential_std, laguerre_mean, laguerre_std = scaler_daily_parameters.values[0]
    elif len(scaler_daily_parameters)>1:
        error = 'Multiple rows for target date in standardscaler_parameters_daily, taking first one. Check bigquery table.'
        exponential_mean, exponential_std, laguerre_mean, laguerre_std = scaler_daily_parameters.iloc[0, :]
    else:
        return {'status':'Failed', 'error':'Target date not in standardscaler_parameters_daily', 'result':np.nan}
    
    #If the function gets this far, the values are correct. A prediction is made and returned appropriately.
    prediction = predict_ytw(t, const, exponential, laguerre, exponential_mean, exponential_std, laguerre_mean, laguerre_std)
    
    return {'status':'Success', 'error':error, 'result':prediction}