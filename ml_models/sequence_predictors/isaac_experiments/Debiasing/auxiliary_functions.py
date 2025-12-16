import json
import pandas as pd
import numpy as np
import gcsfs
from sklearn.metrics import mean_absolute_error
import os 
import warnings
from datetime import datetime
from finance import *
import time #FOR DEBUGGING
from google.cloud import bigquery
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/jupyter/ficc/isaac_creds.json"

WARM = 1000
trades_df_cols = ['trade_date', 'trade_datetime', 'published_datetime', 'trade_type', 'transaction_type', 'cusip', 'par_traded', 'yield', 'prediction', 'error']

global trades_df 

def load_from_cloud_storage(path, fs):
    try:
        with fs.open(path, 'rb') as f:
            temp = pickle.load(f)
        return temp
    except FileNotFoundError as f:
        print(f'{path} was not found in gcloud storage. Exception {f}.')


def upload_to_cloud_storage(path, obj, fs):
    try:
        with fs.open(path, 'wb') as f:
            temp = pickle.dump(obj, f)
    except FileNotFoundError as f:
        print(f'{path} upload was not successful. Exception {f}.')


### FUNCTIONS TO PARSE MSRB TRADE MESSAGES:
def get_all_msrb_files(date):
    ls = storage_client.list_blobs('msrb_intraday_real_time_trade_files')
    files = []
    for file in ls:
        if date in file.name: 
            files.append(f'msrb_intraday_real_time_trade_files/{file.name}')
    return files

def get_trade_messages(path, fs):
    '''Loads a trade message json from a specified path in gcloud storage and parses relevant data, with each row in the json representing a single trade.
    '''
    
    print('Getting trade messages') #DEBUG)
    try:
        with fs.open(path, 'rb') as f:
            trade_messages = json.load(f)
    except FileNotFoundError as f:
        print(f'{path} was not found in gcloud storage. Exception {f}.')
    print('Trade messages loaded') #DEBUG
    
    result_dict = {}
    cancellations = []
    
    def process_trade_message(trade_message):
        '''Takes a single trade message from MSRB, formatted as a string, and extracts relevant fields'''
        
        parsed_messages = trade_message.split(',')
        temp_dict = dict([s.split('=',1) for s in parsed_messages])

        try: # TODO: if we do a try and except block then there is no need got dict.get(,None), we just return {} in the case of any error. can clean up
            rtrs_control_number = temp_dict.get('4', None)
            trade_type = temp_dict.get('5', None)
            transaction_type = temp_dict.get('6', None)
            cusip = temp_dict.get('7', None)

            par_traded, trade_yield = temp_dict.get('17', None), temp_dict.get('19', None)
            if par_traded:  
                if par_traded == 'MM+':
                    par_traded = 50000000.0
                    is_trade_with_a_par_amount_over_5MM = True
                else: 
                    par_traded = float(par_traded)
                    is_trade_with_a_par_amount_over_5MM = False
            if trade_yield:  trade_yield = float(trade_yield)

            trade_date = datetime.strptime(temp_dict.get('14', None),'%Y%m%d').strftime('%Y-%m-%d')

            trade_datetime = ' '.join((temp_dict.get('14', None), temp_dict.get('15', None)))
            trade_datetime = datetime.strptime(trade_datetime,'%Y%m%d %H%M%S').strftime('%Y-%m-%d %H:%M:%S')

            published_datetime = ' '.join((temp_dict.get('23', None), temp_dict.get('24', None)))
            published_datetime = datetime.strptime(published_datetime,'%Y%m%d %H%M%S').strftime('%Y-%m-%d %H:%M:%S')

            if not cusip or not trade_yield or not trade_date or not par_traded or not trade_type or not transaction_type or not trade_datetime or not published_datetime:
                return {}
            else: 
                return {rtrs_control_number:dict(zip(trades_df_cols[:-2],
                                                     [trade_date, trade_datetime, published_datetime, trade_type, transaction_type, cusip, par_traded, trade_yield]
                                                 ))}
        except Exception as e:
            print(e)
            return {}
                
    try:          
        print(f'Processing {len(trade_messages)} trade messages') 
        for trade_message in trade_messages:
            processed_message = process_trade_message(trade_message['Message'])
            if processed_message: 
                if next(iter(processed_message.values())) == 'C':
                    cancellations.append(processed_message['cusip'])
                else:
                    result_dict.update(processed_message)
    except:
        return {}, []

    return result_dict, cancellations

def _price_df(df):
    try: 
        result = price_cusips_list(df['cusip'].tolist(), df['par_traded'].tolist(), df['trade_type'][0])['ficc_ytw'].tolist()
    except Exception as e:
        print(f"Error in _price_df. Exception: {e}")
        result = None
    return result

def price_trades(processed_trade_messages):

    temp = pd.DataFrame(processed_trade_messages).T
    
    S = temp[temp.trade_type == 'S']
    D = temp[temp.trade_type == 'D']
    P = temp[temp.trade_type == 'P']
    
    start = time.time() #FOR DEBUGGING
    if len(S):
        print(f'Pricing dealer sell, {len(S)}/{len(processed_trade_messages)} trades')   #FOR DEBUGGING
        S['prediction'] = _price_df(S)
        print('Dealer sell trades priced')
    if len(D):
        print(f'Pricing dealer dealer, {len(D)}/{len(processed_trade_messages)} trades')  #FOR DEBUGGING
        D['prediction'] = _price_df(D)
        print('Dealer dealer trades priced')
    if len(P):
        print(f'Pricing dealer purchase, {len(P)}/{len(processed_trade_messages)} trades') #FOR DEBUGGING
        P['prediction'] = _price_df(P)
        print('Dealer purchase trades priced')
            
    # print(f'Pricing {len(temp)} trades took {time.time() - start} seconds.')

    temp = pd.concat([S, D, P])
    temp['error'] = temp['prediction'] - temp['yield'] 
    #TODO: should we sort here? 
    #Scenario: a modification is sent, and we process it 
    
    return temp

def update_intraday_cusips(processed_trade_messages, cancellations, trades_df):
    '''Updates trades_df inplace with newly parsed trade messages, dropping cancellations, repricing modifications and pricing new trades
    
    Importantly, this function is indempotent because it modifies the dataframe inplace using .loc. 
    This ensures that if we accidentally parse the exact same trade message, it will not occur as an additional trade, which will double-weight the trade in debiasing.
    '''
    print('Running update_intraday_cusips') #DEBUG
    #if trade messages are empty, exit function 
    if not processed_trade_messages: return 

    #drop cancellations from in-memory trade dataframe
    trades_df.drop(cancellations, inplace=True, errors='ignore')
    if cancellations: print(f'Trades with rtrs {cancellations} were droppped')
    print(f'Pricing trades {len(processed_trade_messages)} trades.') #DEBUG
    
    
    priced_trades = price_trades(processed_trade_messages) 
    print('Trades priced.') #DEBUG
    priced_trades.dropna(subset=['prediction'], inplace=True) #drop any unpriceable cusips
    priced_trades['trade_datetime'] = pd.to_datetime(priced_trades['trade_datetime'])
    priced_trades['trade_date'] = pd.to_datetime(priced_trades['trade_date'])
    priced_trades['published_datetime'] = pd.to_datetime(priced_trades['published_datetime'])
    priced_trades.sort_values(by='published_datetime', ascending=True, inplace=True)
    
    for row in priced_trades[trades_df_cols].itertuples():
        if row[5]=='I':
            trades_df.loc[row[0]] = row[1:]

        else: 
            if row[0] in trades_df.index: #TODO: test this with a dictionary of rtrs control numbers instead, see if it is faster. technically should be of O(logn) vs O(1)
                print(f'Trade {row[0]} is a replacement; transaction_type = {row[5]}, replacing original entry.')
                trades_df.loc[row[0]] = row[1:]
            else: 
                print(f'Trade {row[0]} is anomalous; not an initial trade message (transaction_type = {row[5]}) but not found in intraday messages stored in current session. Trade_datetime {row[2]}, publish_datetime {row[3]}')

### FUNCTIONS TO PERFORM DEBIASING
def calculate_weighted_average(data, weighting_col, error_col, method = 'default', mask_large = None):
    '''Calculates weighted average of error_col based on weighting_col AND masks trades less than X seconds ago. 
    
    If weighted average is to be calculated based on error magnitude, weighting_col should be set to error_col. 
    Different ways of calculating the weighted average are dictated by the method keyword argument.
    By default, trades less than 60 seconds ago are not used to estimate biases. 
    '''
    
    data = data.iloc[:-1] 
    if len(data) == 0: return 0 

    errors = data[error_col].to_numpy()
    weights = data[weighting_col].to_numpy()
    
    if method == 'simple_average':
        weights = np.ones(len(errors))

    if method == 'default':
        #if method is default, weights are left as is 
        pass

    if method == 'reciprocal':
        #this gives larger weight to small errors and for large errors, should disregard them almost entirely 
        weights = np.abs(1/weights)

    if method == 'log':
        #this moderates large errors
        weights = np.log(np.abs(weights) + 1)

    if method == 'log_reciprocal':
        #this gives larger weight to small errors but sets also moderates how much extremely small errors can be weighted 
        weights = np.log(np.abs(1/weights)) 

    if mask_large:
        #this gives larger weight to small errors but sets also moderates how much extremely small errors can be weighted 
        weights = np.where(np.abs(errors) <= mask_large, weights, 0) 
 
    try:
        #try to calculate the average, if not possible then it means that the masked errors are all zero.
        return np.average(errors, weights=weights)
    except: 
        return 0 

def simulate_weighted_average(df, weighting_col, error_col, groupby_cols = ['trade_date'], window_size = 2000, weighting_method = 'default', mask_large = 35):
    '''Simulates debiasing procedure in production by calculating rolling average bias AND masks trades less than X seconds ago. 
    
    Window_size dictates the N preceding trades to estimate bias correction for each row. 
    Setting window_size larger than the dataframe is equivalent to using pd.expanding(). 
    '''
    
    subset = [weighting_col, error_col, 'published_datetime']
    if weighting_col == error_col:  subset.remove(error_col) #if we are weigthing by the error column then don't slice the column twice
    
    if window_size > len(df): window_size = len(df)
    groupby_dfs = list(df[subset].rolling(window_size, min_periods = 1, method='table'))
    
    biases = []
    
    if mask_large:
        print(f'Ignoring trades with errors larger than {mask_large}bps in bias correction calculations.')
    for sub_df in groupby_dfs:
        biases.append(calculate_weighted_average(sub_df, weighting_col, error_col, method = weighting_method, mask_large = mask_large))
    
    return biases 

def debias_series(pred, truth, bias_correction):
    '''Subtract bias correction from prediction and calculates MAE relative to true values
    
    The bias_correction values must be of the same length as pred and truth. Each row should correspond to the bias correction for the corresponding prediction.
    '''
    
    pred = np.array(pred).flatten()
    truth = np.array(truth).flatten()
    bias_correction = np.array(bias_correction).flatten()
    if len(bias_correction) != len(pred) != len(truth): raise ValueError('Pred, truth, bias_correction must be same shape')
    
    corrected_pred = pred - bias_correction
    print(f'Original bias: {np.mean(pred-truth):.3f}, Original MAE: {mean_absolute_error(pred, truth):.3f}, Corrected bias: {np.mean(corrected_pred-truth):.3f}, Corrected MAE: {mean_absolute_error(corrected_pred, truth):.3f}')


def bias_warm_start(bias, N):
    '''Masks the first N trades of each day with 0.'''
    N = min(N, len(bias))
    # print(type(N), bias[:N])
    bias[:N] = np.zeros(N)
    return bias



####NEW FUNCTIONS 
def getSchema():
    '''Get schema for trades_df to upload to BQ'''
    schema = [
        bigquery.SchemaField("trade_date", "DATE", "NULLABLE"),
        bigquery.SchemaField("trade_datetime","DATETIME", "NULLABLE"),
        bigquery.SchemaField("published_datetime","DATETIME", "NULLABLE"),
        bigquery.SchemaField("trade_type","STRING", "NULLABLE"),
        bigquery.SchemaField("transaction_type","STRING", "NULLABLE"),
        bigquery.SchemaField("cusip","STRING", "NULLABLE"),
        bigquery.SchemaField("par_traded","FLOAT", "NULLABLE"),
        bigquery.SchemaField("yield","FLOAT", "NULLABLE"),
        bigquery.SchemaField("prediction","FLOAT", "NULLABLE"),
        bigquery.SchemaField("error","FLOAT", "NULLABLE")]
    
    return schema


def uploadData(trades_df, TABLE_ID):
    ''' Upload trades_df to BQ'''
    
    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(schema=getSchema(),
                                        write_disposition='WRITE_APPEND')
    job = client.load_table_from_dataframe(trades_df, TABLE_ID, job_config=job_config)
    
    try:
        job.result()
        print('Upload Successful')
    except Exception as e:
        print(f'Failed to Upload, Exception: {e}')
        raise 
        
        
        
def calculate_weighted_average_pos(data, weighting_col, error_col, method = 'default', mask_large = None):
    '''Calculates weighted average of error_col based on weighting_col AND masks trades less than X seconds ago. 
    
    If weighted average is to be calculated based on error magnitude, weighting_col should be set to error_col. 
    Different ways of calculating the weighted average are dictated by the method keyword argument.
    By default, trades less than 60 seconds ago are not used to estimate biases. 
    '''
    
    data = data.iloc[:-1] 
    if len(data) == 0: return 0 

    errors = data[error_col].to_numpy()
    weights = data[weighting_col].to_numpy()
    
    if method == 'simple_average':
        weights = np.ones(len(errors))

    if method == 'default':
        #if method is default, weights are left as is 
        pass

    if method == 'reciprocal':
        #this gives larger weight to small errors and for large errors, should disregard them almost entirely 
        weights = np.abs(1/weights)

    if method == 'log':
        #this moderates large errors
        weights = np.log(np.abs(weights) + 1)

    if method == 'log_reciprocal':
        #this gives larger weight to small errors but sets also moderates how much extremely small errors can be weighted 
        weights = np.log(np.abs(1/weights)) 

    if mask_large:
        #this gives larger weight to small errors but sets also moderates how much extremely small errors can be weighted 
        weights = np.where(np.abs(errors) <= mask_large, weights, 0) 
 
    weights = np.where(errors>=0, weights, 0) 
    
    try:
        #try to calculate the average, if not possible then it means that the masked errors are all zero.
        return np.average(errors, weights=weights)
    except: 
        return 0 
    
    
def simulate_weighted_average_pos(df, weighting_col, error_col, groupby_cols = ['trade_date'], window_size = 2000, weighting_method = 'default', mask_large = 35):
    '''Simulates debiasing procedure in production by calculating rolling average bias AND masks trades less than X seconds ago. 
    
    Window_size dictates the N preceding trades to estimate bias correction for each row. 
    Setting window_size larger than the dataframe is equivalent to using pd.expanding(). 
    '''
    
    subset = [weighting_col, error_col, 'published_datetime']
    if weighting_col == error_col:  subset.remove(error_col) #if we are weigthing by the error column then don't slice the column twice
    
    if window_size > len(df): window_size = len(df)
    groupby_dfs = list(df[subset].rolling(window_size, min_periods = 1, method='table'))
    
    biases = []
    
    if mask_large:
        print(f'Ignoring trades with errors larger than {mask_large}bps in bias correction calculations.')
    for sub_df in groupby_dfs:
        biases.append(calculate_weighted_average_pos(sub_df, weighting_col, error_col, method = weighting_method, mask_large = mask_large))
    
    return biases 