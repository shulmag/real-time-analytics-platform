# last edited by Gil Shulman 11-11-2021

# This module implements the function to get the daily yield curve. We use the value of the yield to calculate the ficc price for the trade. 

import numpy as np
import pandas as pd
import datetime
import holidays
from dateutil import parser
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

#Default shape parameter based on initial hyperparameter tuning. This affects the curvature and slope of the nelson-siegel curve
#higher values generally imply a straighter, more monotonic yield curve (particularly at maturities < 1)
L = 17 

#Creates a dataframe containing US Federal holidays.
dr = pd.date_range(start='2010-01-01', end='2100-01-01')
df = pd.DataFrame()
df['Date'] = dr
cal = calendar()
holidays = cal.holidays(start=dr.min(), end=dr.max())

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "eng-reactor-287421-112eb767e1b3.json"

from google.cloud import bigquery

bq_client = bigquery.Client()

project = "eng-reactor-287421"

def sqltodf(sql,limit = ""):
    if limit != "": 
        limit = f" ORDER BY RAND() LIMIT {limit}"
    bqr = bq_client.query(sql + limit).result()
    return bqr.to_dataframe()

# This module gets the most recent Nelson-Siegel coefficient and
# the standard scalar coefficient from Redis Memorystore given a particular datetime is
# Scaling the laguerre and exponential functions: the laguerre is of a smaller scale than the exponential, suggesting that the ridge penalty imposed on the laguerre is disproportionately smaller than the exponential. 

# Between the hours of 10 and 16 EST on business days, the Nelson-Siegel coefficient will be 
# the coefficient for that minute and the standard scalar will be that of the last business day.

import pandas as pd
nelson = sqltodf("SELECT * FROM `eng-reactor-287421.yield_curves.nelson_siegel_coef_minute`")
scalar = sqltodf("select * from`eng-reactor-287421.yield_curves.standardscaler_parameters_daily`")

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
scalar['date'] = scalar['date'] + 1* US_BUSINESS_DAY

scalar['date'] = pd.to_datetime(scalar['date'])
nelson['date'] = pd.to_datetime(nelson['date'])

nelson['join_date'] = nelson['date'].dt.date

nelson.set_index("join_date",drop=True,inplace=True)
scalar.set_index("date",drop=True,inplace=True)

## WHAT NEEDS TO BE DONE: Join Nelson and scalar on the index such that for each we have the correct scalar values for every nelson key
# Then: drop "join_date" and make the key the date of Nelson

#result = nelson.join(scalar, on=index)
nelson_scalar_df = pd.merge(nelson,scalar,left_index=True, right_index=True)
nelson_scalar_df.set_index("date",inplace=True)

def get_business_day(date):
    '''
    Checks whether the object is a datetime object.
    Then checks whether the date is before we began collecting yield curve coefficients,
    then checks if the date is a weekend or a US Federal Holiday. 
    If the last condition is true, the function loops back to the most recent business day. 
    '''

    if isinstance(date,datetime.date):
        date=date
    else:
        date=parser.parse(date)
    data_start_date= datetime.datetime.strptime("2021-7-27:0:00:00","%Y-%m-%d:%H:%M:%S")
    if date < data_start_date:
        return data_start_date
    else: date=date
    while date.strftime("%Y%m%d") in holidays or date.weekday() in set((0,5,6)):
        date = date-pd.DateOffset(1)
    else:
        return date

def get_last_business_time(date):

    '''
    Checks whether the time of the datetime object is before or after business hours. 
    If so, it sends us back to the last business datetime. 
    '''
    market_open = datetime.time(10, 0)
    market_close = datetime.time(16, 0)
    
    if date.time() < market_open:
        date = get_business_day(date-pd.DateOffset(1))
        date = date.replace(hour=15, minute=59)
        return date
    elif date.time() > market_close:
        date = date.replace(hour=15, minute=59)
        return date
    else: 
        return date

def find_last_minute(date):

    '''
    Checks whether the datetime exists as a key in redis. 
    If not, we loop back to the previous datetime key.
    '''
    #print(f"find_last_minute: {date}")
    while date not in nelson_scalar_df.index:
        date=date-pd.Timedelta(minutes=1)
        #print(f"date=date-pd.Timedelta(minutes=1): {date}")
    #print(f"Out of loop returning:{date}")
    return date

def get_yc_data(date):

    '''
    Fetches the most recent data from redis given a particular datetime. 
    '''
    date=get_business_day(date)
    date=get_last_business_time(date)
    date=find_last_minute(date.replace(second=0))
    data = nelson_scalar_df[nelson_scalar_df.index == date]
    return {"nelson_values":data[["const","exponential","laguerre"]],"scalar_values":data[["exponential_mean","exponential_std","laguerre_mean","laguerre_std"]]}

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
    This function grabs the nelson siegel and standard scalar coefficient from 
    memory store 
    '''
    temp_dict = get_yc_data(target_date)
    
    # The keys for the dictionary in memory store are defined
    # at the time the data is uploaded on the server
    nelson_coeff = temp_dict['nelson_values']
    scalar_coeff = temp_dict['scalar_values']

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


def get_current_yield_curve(maturity,target_date):
    '''
    This is the main function takes as input a json containing two arguments: the maturity we want the yield-to-worst for and the target 
    ate from which we want the yield curve used in the ytw calculations to be from. There are several conditional statements to deal with
    different types of exceptions. 
    
    The cloud function returns a json containing the status (Failed or Success), the error message (if any)
    and the result (nan if calculation was unsuccessful).
    '''
    
    error = ''
    try:
        nelson_siegel_daily_coef, scaler_daily_parameters = load_model_parameters(target_date)
    except Exception as e:
        raise e
    
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
    prediction = predict_ytw(maturity, const, exponential, laguerre, exponential_mean, exponential_std, laguerre_mean, laguerre_std)
    return prediction