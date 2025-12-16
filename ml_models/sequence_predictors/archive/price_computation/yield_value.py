# Modificatoin: Nelson Siegel and standard scalar coefficients fetched from memory store
import numpy as np 
import pandas as pd
import sys
from google.cloud import bigquery

PROJECT_ID = "eng-reactor-287421"

#Default shape parameter based on initial hyperparameter tuning. This affects the curvature and slope of the nelson-siegel curve
#higher values generally imply a straighter, more monotonic yield curve (particularly at maturities < 1)
L = 17 

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


def load_model_parameters(target_date, nelson_params, scalar_params):
    '''
    This function grabs the nelson siegel and standard scalar coefficient from the dataframes 
    '''

    nelson_coeff = nelson_params.loc[target_date]
    scalar_coeff = scalar_params.loc[target_date]

    return nelson_coeff, scalar_coeff


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


def yield_curve_level(maturity:float, target_date:str, nelson_params, scalar_params):
    '''
    This is the main function takes as input a json containing two arguments: the maturity we want the yield-to-worst for and the target 
    ate from which we want the yield curve used in the ytw calculations to be from. There are several conditional statements to deal with
    different types of exceptions. 
    
    The cloud function returns a json containing the status (Failed or Success), the error message (if any)
    and the result (nan if calculation was unsuccessful).
    '''
    
    t = maturity
    
    #If a target_date is provided but it is in an invalid format, then the correct values from the model and scaler parameters cannot be
    #retrieved, and an error is also returned.
    try:
        nelson_siegel_daily_coef, scaler_daily_parameters = load_model_parameters(target_date, nelson_params, scalar_params)
    except Exception as e:
        raise e 
    
    if len(nelson_siegel_daily_coef)==1:
        const, exponential, laguerre = nelson_siegel_daily_coef.values[0]
    elif len(nelson_siegel_daily_coef)>1:
        error = 'Multiple rows for target date in nelson_siegel_coef_daily, taking first one. Check bigquery table.'
        const, exponential, laguerre = nelson_siegel_daily_coef.iloc[0, :]
    else:
        print("Failed to grab coefficients")
        sys.exit()
   
    if len(scaler_daily_parameters)==1:
        exponential_mean, exponential_std, laguerre_mean, laguerre_std = scaler_daily_parameters.values[0]
    elif len(scaler_daily_parameters)>1:
        error = 'Multiple rows for target date in standardscaler_parameters_daily, taking first one. Check bigquery table.'
        exponential_mean, exponential_std, laguerre_mean, laguerre_std = scaler_daily_parameters.iloc[0, :]
    else:
        print("Failed to grab scalar coefficient")
        sys.exit()
    
    #If the function gets this far, the values are correct. A prediction is made and returned appropriately.
    prediction = predict_ytw(t, const, exponential, laguerre, exponential_mean, exponential_std, laguerre_mean, laguerre_std)
    
    return prediction
