'''
 # model to predic the yield curve. 
 # @ Modification: Nelson-Siegel coefficients are used from a dataframe
 # instead of grabbing them from memory store
 '''

from shutil import ExecError
import numpy as np 
import pandas as pd
import sys
from datetime import datetime, timedelta
import redis
import pickle5 as pickle
from google.cloud import bigquery
from modules.ficc.utils.yc_data import get_yc_data

# Please comment before deploying
# from utils.yc_data import get_yc_data

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

def load_model_parameters(target_date):
    '''
    This function grabs the nelson siegel and standard scalar coefficient from the dataframes 
    '''

    target_date = datetime.strptime(target_date, "%Y-%m-%d:%H:%M")
    yield_curve_parameters = get_yc_data(target_date)

    nelson_coeff = yield_curve_parameters['nelson_values']
    scalar_coeff = yield_curve_parameters['scalar_values']
    shape_parameter = yield_curve_parameters['shape_parameter']

    return nelson_coeff, scalar_coeff, shape_parameter


###Functions used for prediction  Function to 
def get_scaled_features(t:np.array, 
                        exponential_mean:float, 
                        exponential_std:float, 
                        laguerre_mean:float, 
                        laguerre_std:float, 
                        shape_parameter:float):
    
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
    
    X1 = (decay_transformation(t, shape_parameter) - exponential_mean)/exponential_std 
    X2 = (laguerre_transformation(t, shape_parameter) - laguerre_mean)/laguerre_std 
    return X1, X2

def predict_ytw(maturity:np.array, 
                const:float , 
                exponential:float , 
                laguerre:float , 
                exponential_mean:float, 
                exponential_std:float, 
                laguerre_mean:float, 
                laguerre_std:float,
                shape_parameter:float):
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
    
    X1, X2 = get_scaled_features(maturity, 
                                 exponential_mean, 
                                 exponential_std, 
                                 laguerre_mean, 
                                 laguerre_std, 
                                 shape_parameter)

    return const + (exponential * X1) + (laguerre * X2)


def yield_curve_level(maturity, target_date):
    '''
    This is the main function takes as input a json containing two arguments: the maturity we want the yield-to-worst for and the target 
    ate from which we want the yield curve used in the ytw calculations to be from. There are several conditional statements to deal with
    different types of exceptions. 
    
    The cloud function returns a json containing the status (Failed or Success), the error message (if any)
    and the result (nan if calculation was unsuccessful).
    '''
    
    #If a target_date is provided but it is in an invalid format, then the correct values from the model and scaler parameters cannot be
    #retrieved, and an error is also returned.
    try:
        nelson_siegel_daily_coef, scaler_daily_parameters, shape_parameter = load_model_parameters(target_date)
    except Exception as e:
        raise e 
    
    if len(nelson_siegel_daily_coef)==1:
        try:
            const, exponential, laguerre = nelson_siegel_daily_coef.values[0]
        except Exception as e:
            _, const, exponential, laguerre = nelson_siegel_daily_coef.values[0]

    elif len(nelson_siegel_daily_coef.shape)>1 and len(nelson_siegel_daily_coef)>1:
        error = 'Multiple rows for target date in nelson_siegel_coef_daily, taking first one. Check bigquery table.'
        const, exponential, laguerre = nelson_siegel_daily_coef.iloc[0, :]
    elif len(nelson_siegel_daily_coef)>1:
        #print(nelson_siegel_daily_coef)
        const, exponential, laguerre = nelson_siegel_daily_coef
    else:
        raise Exception("Nelson-Siegel coefficients for the selected dates do not exist")
        sys.exit()
   
    if len(scaler_daily_parameters)==1:
        try:
            exponential_mean, exponential_std, laguerre_mean, laguerre_std = scaler_daily_parameters.values[0]
        except Exception as e:
            _, exponential_mean, exponential_std, laguerre_mean, laguerre_std = scaler_daily_parameters.values[0]
    elif len(scaler_daily_parameters.shape)>1 and len(scaler_daily_parameters)>1:
        error = 'Multiple rows for target date in standardscaler_parameters_daily, taking first one. Check bigquery table.'
        exponential_mean, exponential_std, laguerre_mean, laguerre_std = scaler_daily_parameters.iloc[0, :]
    elif len(scaler_daily_parameters)>1:
        exponential_mean, exponential_std, laguerre_mean, laguerre_std = scaler_daily_parameters
    else:
        raise Exception("Failed to grab scalar coefficient, they do not exist")
        sys.exit()
    
    #If the function gets this far, the values are correct. A prediction is made and returned appropriately.
    prediction = predict_ytw(maturity, 
                             const, 
                             exponential, 
                             laguerre, 
                             exponential_mean, 
                             exponential_std, 
                             laguerre_mean, 
                             laguerre_std,
                             shape_parameter)
    return prediction
