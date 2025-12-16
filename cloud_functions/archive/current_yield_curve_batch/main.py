import numpy as np 
import pandas as pd
from sklearn.linear_model import Lasso
from datetime import datetime, timedelta
import requests

from bigquery_utils import *
from yieldcurve import *  

#Finnhub.io API key from a team member's account
finnhub_apikey = 'c499cpiad3ieskgqq5lg'

#Define model hyperparameters
window_size = 45 #window-size to train the ETF model 
L = 17 #shape parameter for yield curve 
alpha = 0.001 #regularization penalty for the ridge regression used to estimate the yield curve
best_lambdas = {'sp_7_12_year_national_amt_free_index':5.0,
'sp_high_quality_index':5.0,
'sp_high_quality_intermediate_managed_amt_free_index':5.0,
'sp_high_quality_short_intermediate_index':5.0,
'sp_high_quality_short_index':5.0} #optimal regularization penalty for lasso using ETF prices to predict index ytw

#Define the best subset of funds for each index. These were selected using a separate notebook {to add reference}
best_funds = {'sp_7_12_year_national_amt_free_index':
['PZA', 'ITM', 'MLN', 'TFI', 'HYMB', 'HYD', 'MUB', 'SHYD'],
'sp_high_quality_index':
['PZA', 'ITM', 'MLN', 'TFI', 'HYMB', 'HYD', 'MUB', 'SHYD'],
'sp_high_quality_intermediate_managed_amt_free_index':
['PZA', 'ITM', 'MLN', 'TFI', 'HYMB', 'HYD', 'MUB', 'SHYD'],
'sp_high_quality_short_intermediate_index':
['PZA', 'ITM', 'MLN', 'TFI', 'HYMB', 'HYD', 'MUB'],
'sp_high_quality_short_index':
['PZA', 'ITM', 'TFI', 'HYMB', 'MLN', 'HYD', 'MUB']}

#from the optimal funds we find the unique funds, to avoid having to do repeat calculations
unique_funds = set([item for sublist in best_funds.values() for item in sublist])

def get_quote_finnhub(etf:str):
    '''
    This function gets the current price for the given ETF using the finnhub.io API. There is a maximum of 60 calls per minute. The request returns a json file with a number of variables. 'c' refers to the Current Price.
    
    Parameters: 
    etf:str
    
    '''
    response = requests.get('https://finnhub.io/api/v1/quote?symbol={}&token={}'.format(etf, finnhub_apikey))
    return response.json()['c']

def main(request):
    '''
    This is the main function. It takes as input a  (serialized) list of maturities for which the realtime ytw is needed, and returns a (serialized) list of realtime ytw. 
    
    '''    
    if request.args:
        t = request.args.get('maturity')
    elif request.get_json(silent=True):
        t =  request.get_json(silent=True).get('maturity')
    else:
        return  {'status':'Failed', 'error':'Error passing arguments', 'result':np.nan, 'timestamp':np.nan}

    #List of maturities has to be passed as a string and loaded using json then converted to array
    t = json.loads(t)
    t = np.array(t).astype(float)

    #The maturity value cannot be <= 0, hence an error is returned if any are <= 0 
    if (t <=0).any():
        return  {'status':'Failed', 'error':'Enter a valid maturity greater than 0', 'result':np.nan, 'timestamp':np.nan}
        
    #Load SP index data 
    index_data = load_index_yields_bq()
    indices = list(index_data.keys())
    
    #Load daily ETF data 
    etf_data = load_daily_etf_prices_bq()

    #Load maturity data and scalers for those maturities 
    maturity_df = load_maturity_bq()
    scaler_daily_parameters = load_scaler_daily_bq()
    
    #Get quote data for today, now 
    response = requests.get('https://finnhub.io/api/v1/quote?symbol={}&token={}'.format('MUB', finnhub_apikey)).json()
    quote_time = (pd.to_datetime(response['t'], unit='s') - timedelta(hours=4)).replace(second=0) #finnhub data is not in Eastern Time

    #Get the timestamp for the quote, and today's date 
    target_time = quote_time
    target_date = str(target_time.date())

    quote_data = pd.DataFrame()

    for etf in unique_funds:
        quote_data[etf] = [get_quote_finnhub(etf)]

    #Get the most recent scaler and maturity data, from the day before today (the target_date)
    day_before_target_date = get_day_before(target_date, maturity_df)
    exponential_mean, exponential_std, laguerre_mean, laguerre_std = get_scaler_params(day_before_target_date, scaler_daily_parameters)
    maturity_dict = maturity_df.loc[day_before_target_date].to_dict()

    #Next get the closing ETF price data from the day before today and calculate the % change relative to the current quoted price, in basis points
    prev_close_data = []
    
     #ETF data stored in dictionary of dataframes, so we retrieve the right entry and concatenate them
    for fund in unique_funds:
        prev_close_data.append(etf_data[fund]['Close_{}'.format(fund)].loc[day_before_target_date:])

    prev_close_data = pd.concat(prev_close_data, axis=1)

    intraday_change = ((quote_data.values - prev_close_data) / prev_close_data) / 0.0001

    ### ETF MODEL
    models = {}
    X_cols = {}

    predicted_ytw = pd.DataFrame() #dataframes to store the predicted index yields
    
    #For each index, we train its own model using the corresponding best parameters and etf subset
    for current_index in list(best_funds.keys()): 
        ###TRAINING MODEL
        #we load the optimal regularization penalty and features for each fund
        current_best_lambda = best_lambdas[current_index]
        current_best_funds = best_funds[current_index]

        #we load the data for the current day
        current_data = preprocess_data(index_data, etf_data, current_index, current_best_funds).loc[:target_date]
        model_data = current_data.copy()
        model_data = model_data.iloc[-window_size:,:] #we load the window of [- window size : now]

        assert len(model_data) == window_size

        #Retrieve our X and Y columns
        X = model_data.drop(['ytw_diff','ytw'], axis=1)
        y = model_data['ytw_diff']

        #Train the lasso model
        lasso = Lasso(alpha = current_best_lambda, random_state=1, max_iter=5000).fit(X, y)
        
        #Save the model and the ordering of columns; currently the saved models are not exported anywhere yet, but they can be to GC and served of Vertex AI 
        models[current_index] = lasso
        X_cols[current_index] = list(X.columns) #we retain the ordering of columns for the test set

        #Make predictions on the relevant columns for the quoted intraday data 
        predictions = lasso.predict(intraday_change[list(X.columns)])[0]

        #Retrieve the closing ytw from yesterday 
        target_date_ytw = current_data.loc[day_before_target_date].ytw
        
        #We predict the change in ytw, so we add these values to the previous closing ytw and save the result in a dataframe 
        target_date_ytw = target_date_ytw + predictions
        predicted_ytw[current_index] = [target_date_ytw]

        assert len(predicted_ytw) == 1
    
    ###YIELD CURVE MODEL 
    #Now we take the predicted ytw for the indices and use it to estimate the nelson-siegel model 
    yield_curve_df = predicted_ytw.T.rename({0:'ytw'}, axis = 1)
    yield_curve_df['Weighted_Maturity'] = yield_curve_df.index.map(maturity_dict).astype(float)
    
    #Transform and scale our data ahead of fitting yield curve 
    X, y = get_NL_inputs(yield_curve_df, L)
    X = scale_X(X, exponential_mean, exponential_std, laguerre_mean, laguerre_std)
    #Fit the yield curve 
    ridge = run_NL_ridge(X,y, scale=False, alpha=alpha)
    
    #Now we take the user input maturity for which we want the ytw, perform the same transformations and make a prediction 
    X_curve = pd.DataFrame()
    X_curve['X1'] = decay_transformation(t, L)
    X_curve['X2'] = laguerre_transformation(t, L)
    X_curve = scale_X(X_curve, exponential_mean, exponential_std, laguerre_mean, laguerre_std)
    
    predictions = ridge.predict(X_curve)
    #Return the prediction as json; again, data has to be serialized for json
    return {'status':'Success', 'error':'', 'result':json.dumps(predictions.tolist()), 'timestamp':target_time}