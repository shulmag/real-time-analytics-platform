'''
TO DO LIST: 
1. Add checks to ensure that values retrieved for yield curves are on the correct days; same date for daily and last available date for minute
2. Add duration calculations or DV01 for a standard bond impleid by the yield curve 
3.
4.
5.
'''

import os
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from google.cloud import bigquery
from google.cloud import storage
from copy import deepcopy
import redis
import pickle5 as pickle

from dateutil import parser
from datetime import datetime 
from pytz import timezone
from pandas.tseries.offsets import BDay
from sklearn.metrics import mean_absolute_error

warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "/home/jupyter/ficc/isaac_creds.json"
bq_client = bigquery.Client()
PROJECT_ID = "eng-reactor-287421"
TABLE_ID = "eng-reactor-287421.yield_curves_v2.shape_parameters"

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

redis_client = redis.Redis(host='10.227.69.60', port=6379, db=0)

t = np.round(np.arange(0.1,30.1,0.1),2) #rounding to avoid precision issues
key_maturities = np.array([1, 2, 5, 10, 15, 30])
eastern = timezone('US/Eastern')
date_hour_format = "%Y-%m-%d %H:%M" 
date_format = "%Y-%m-%d"

################## LOADING DATA ##################
def sqltodf(sql, bq_client):
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()

def get_mmd_data(date_start = None, date_end = None):
    if not date_start: date_start = '2021-07-27'
    if not date_end: date_end = datetime.now().strftime('%Y-%m-%d')
    query = f'''
        SELECT date, maturity, AAA 
        FROM `eng-reactor-287421.yield_curves.mmd_approximation` 
        WHERE date <= '{date_end}' and date >= '{date_start}'
        order by date desc
        '''
    mmd_data = sqltodf(query, bq_client)
    mmd_data.AAA = mmd_data.AAA.astype(float)
    return pd.pivot_table(mmd_data, index = 'date', columns='maturity', values='AAA')*100

def get_index_data(date_start = None, date_end = None):
    if not date_start: date_start = '2021-07-27'
    if not date_end: date_end = datetime.now().strftime('%Y-%m-%d')
    
    index_data  = [] 
    for table in sp_index_tables:
        query = f'''SELECT * FROM `eng-reactor-287421.spBondIndex.{table}` WHERE date <= '{date_end}' and date >= '{date_start}' order by date desc'''
        df = sqltodf(query, bq_client)
        df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')
        df['ytw'] = df['ytw'] * 100
        df = df.drop_duplicates('date')
        df.set_index('date', inplace=True, drop=True)
        index_data.append(df)

    df = pd.concat(index_data, axis=1)
    df.columns = sp_maturity_tables
    df.ffill(axis=0, inplace=True)
    df.dropna(inplace=True)
    return df

def get_maturity_dict(maturity_df, date):
    temp_df = maturity_df.loc[date].T
    temp_dict = dict(zip(temp_df.index, temp_df.values))
    return temp_dict

def get_maturity_data(date_start = None, date_end = None):
    if not date_start: date_start = '2021-07-27'
    if not date_end: date_end = datetime.now().strftime('%Y-%m-%d')
    
    maturity_data  = []

    for table in sp_maturity_tables:
        query = f"SELECT * FROM `eng-reactor-287421.spBondIndexMaturities.{table}` WHERE effectivedate <= '{date_end}' and effectivedate >= '{date_start}' order by effectivedate desc"
        df = sqltodf(query, bq_client)   
        df['effectivedate'] = pd.to_datetime(df['effectivedate'], format = '%Y-%m-%d')
        df = df.drop_duplicates('effectivedate')
        df.set_index('effectivedate', inplace=True, drop=True)
        df = df[['weightedAverageMaturity']]
        maturity_data.append(df) 
        
    df = pd.concat(maturity_data, axis=1)
    df.columns = sp_maturity_tables
    df.ffill(axis=0,inplace=True)
    df.dropna(inplace=True)
    return df


def get_scaler_data(date_start = None, date_end = None):
    if not date_start: date_start = '2021-07-27'
    if not date_end: date_end = datetime.now().strftime('%Y-%m-%d')

    scaler_query = f"select * from`eng-reactor-287421.yield_curves_v2.standardscaler_parameters_daily`  WHERE date <= '{date_end}' and date >= '{date_start}' order by date asc"
    
    daily_ycl_scaler = sqltodf(scaler_query, bq_client)
    daily_ycl_scaler = daily_ycl_scaler.loc[~daily_ycl_scaler.duplicated(subset='date')]

    return daily_ycl_scaler


def get_shape_data(date_start = None, date_end = None):
    if not date_start: date_start = '2021-07-27'
    if not date_end: date_end = datetime.now().strftime('%Y-%m-%d')

    shape_query = f"select * from`eng-reactor-287421.yield_curves_v2.shape_parameters`  WHERE date <= '{date_end}' and date >= '{date_start}' order by date asc"
    
    daily_ycl_shape = sqltodf(shape_query, bq_client).rename({'Date':'date'}, axis=1)
    daily_ycl_shape = daily_ycl_shape.loc[~daily_ycl_shape.duplicated(subset='date')]

    return daily_ycl_shape


def load_yield_curve_params(daily:bool, date_start = None, date_end = None, use_redis = False):
    #If either start or end date not available, just fetch earliest possible/most recent data 
    if not date_start: date_start = '2021-07-27'
    if not date_end: date_end = datetime.now().strftime('%Y-%m-%d')

    
    if daily: 
        yield_query = f"select * from `eng-reactor-287421.yield_curves_v2.nelson_siegel_coef_daily`  WHERE date <= '{date_end}' and date >= '{date_start}' order by date asc"
    else:
        #note that timestamp must be included for bigquery queries filtering by datetime columns, else bigquery will take 00:00 as default time
        yield_query = f"select * from `eng-reactor-287421.yield_curves_v2.nelson_siegel_coef_minute`  WHERE date <= '{date_end} 23:59:00' and date >= '{date_start} 00:00:00' order by date asc" 
    
    scaler_query = f"select * from`eng-reactor-287421.yield_curves_v2.standardscaler_parameters_daily`  WHERE date <= '{date_end}' and date >= '{date_start}' order by date asc"
    shape_query = f"select * from`eng-reactor-287421.yield_curves_v2.shape_parameters`  WHERE date <= '{date_end}' and date >= '{date_start}' order by date asc"
    
    ycl_coef = sqltodf(yield_query, bq_client)
    daily_ycl_scaler = sqltodf(scaler_query, bq_client)
    daily_ycl_shape = sqltodf(shape_query, bq_client).rename({'Date':'date'}, axis=1)
    
    if daily:
        result = pd.merge(ycl_coef, daily_ycl_scaler, on='date').drop_duplicates()
        result = pd.merge(result, daily_ycl_shape, on='date').drop_duplicates()
    
    else:
        #for daily yield curve, we get unique dates from the yield curve table, map those to the last available previous day and join them with the scaler and shape tables for efficiency
        date_reference = ycl_coef.date.dt.date.drop_duplicates()
        
        def prev_day(date):
            #gets the last available date in the scaler and shape tables
            refs = daily_ycl_scaler.date.to_list()
            date = date -  pd.Timedelta(days=1)
                
            #if the previous day is outside of the first or last date in the scaler table, then take the earliest or most recent date 
            if date <= min(refs):
                return min(refs)

            if date > max(refs):
                return max(refs)

            #if data not inside the scaler table, go back by one day and try again 
            while date not in refs:
                date = date -  pd.Timedelta(days=1)

            return date
        
        date_reference = dict(zip(date_reference, date_reference.apply(lambda x: prev_day(x))))
        ycl_coef['date_date'] = ycl_coef.date.dt.date.apply(lambda x: date_reference.get(x))
        
        
        result = pd.merge(daily_ycl_shape, daily_ycl_scaler, on='date').drop_duplicates()
        result = pd.merge(ycl_coef, result,  left_on='date_date', right_on='date', suffixes=['','_right']).drop_duplicates()
        result.drop(['date_date', 'date_right'], axis=1, inplace=True)
        
    result['date'] = pd.to_datetime(result['date'])
    result.set_index('date',drop=True,inplace=True)
    
    return result



def get_redis_keys_date(date:str):
    cursor, key = redis_client.scan(0, match=date+'*', count=100000)
    keys = key
    while cursor!= 0:
        cursor, key = redis_client.scan(cursor, match=date+'*', count=10000)
        if len(key)!=0:
            keys = keys + key
            
    return [x.decode() for x in keys]

def process_redis_data(l):
    df = pd.DataFrame(columns = ['date','const', 'exponential', 'laguerre', 'exponential_mean', 'exponential_std', 'laguerre_mean', 'laguerre_std', 'L'])
    for row in l: 
        timestamp = row['nelson_values'].index[0]
        coef = row['nelson_values']
        scaler = row['scalar_values']
        shape = row['shape_parameter']
        try:
            df = df.append({'date':timestamp, 
                           'const':coef['const'].values[0], 
                            'exponential':coef['exponential'].values[0], 
                            'laguerre':coef['laguerre'].values[0], 
                            'exponential_mean':scaler['exponential_mean'].values[0], 
                            'exponential_std':scaler['exponential_std'].values[0], 
                            'laguerre_mean':scaler['laguerre_mean'].values[0], 
                            'laguerre_std':scaler['laguerre_std'].values[0], 
                            'L': shape},
                          ignore_index=True)
        except:
            print(f'data for {timestamp} problematic')
        
    return df

def get_yc_redis(date:str):
    
    print('Loading keys from redis')
    keys = get_redis_keys_date(date)
    print('Loading data from redis')
    values = [pickle.loads(x) for x in redis_client.mget(keys)] 
    print('Processing data from redis')
    try:
        return process_redis_data(values).set_index('date').sort_index(ascending=True)
    except:
        print('Problem processing data, returning raw values instead')
        return values

################## PROCESSING DATA AND CURVE ESTIMATION ##################
def resolve_date(date):
    if isinstance(date, datetime):
        if date.tzinfo is None: 
            date = eastern.localize(date)
        else: 
            date.astimezone(eastern)

    elif isinstance(date, str):
        date = parser.parse(date)
        date = eastern.localize(date)

    return date 


def predict_ytw(t:np.array, const:float , exponential:float , laguerre:float , exponential_mean:float , exponential_std:float , laguerre_mean:float , laguerre_std:float, L:float ):
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
    
    X1, X2 = get_scaled_features(t, exponential_mean, exponential_std, laguerre_mean, laguerre_std, L)
    return const + exponential*X1 + laguerre*X2

def decay_transformation(t, L):
    return L*(1-np.exp(-t/L))/t

def laguerre_transformation(t, L):
    return (L*(1-np.exp(-t/L))/t) -np.exp(-t/L)

def get_scaled_features(t:np.array, exponential_mean:float, exponential_std:float, laguerre_mean:float, laguerre_std:float, L:float):
    
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

def run_NL_model(model, scaler, df, L, **kwargs):
    df['X1'] = decay_transformation(df['Weighted_Maturity'], L)
    df['X2'] = laguerre_transformation(df['Weighted_Maturity'], L)

    X = df[['X1','X2']]
    X = scaler.fit_transform(X)
    
    y = df.ytw
    
    model = model.fit(X , y)

    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)

    return model, mae, scaler




# def estimate_yields(df, discrete = False):
#     if discrete:
#         temp = df.apply(lambda x: predict_ytw(key_maturities, 
#                                         x.const, 
#                                         x.exponential, 
#                                         x.laguerre, 
#                                         x.exponential_mean, 
#                                         x.exponential_std,
#                                         x.laguerre_mean,
#                                         x.laguerre_std,
#                                          x.L),
#                    axis=1)        
#     else: 
#         temp = df.apply(lambda x: predict_ytw(t, 
#                                         x.const, 
#                                         x.exponential, 
#                                         x.laguerre, 
#                                         x.exponential_mean, 
#                                         x.exponential_std,
#                                         x.laguerre_mean,
#                                         x.laguerre_std,
#                                          x.L),
#                    axis=1)

#     temp = pd.DataFrame(zip(*temp)).T
#     temp.index = df.index.values
#     if discrete: temp.columns = key_maturities
#     else: temp.columns = t
#     return temp

############### PLOTTING CURVES ###############
def plot_curves(df, daily = True, date = None, use_plotly = False, print_summary=False, start = None, end = None, xlim = 30, ylim_lower = 270, ylim_upper = 350, hourly_interval = 1, return_figure = False):
    if daily: 
        start = resolve_date(start)
        end = resolve_date(end)
        data = df.loc[start:end]
        if len(data) == 0:
            print(f'No data for date range {start} - {end}')
            return None
        
    else: 
        timestamps = [pd.to_datetime(f'{date} 09:30') + pd.offsets.Hour(hourly_interval*i) for i in range(1,int(((6/hourly_interval)+1)))]
        
        try: 
            temp = df.loc[date]
        except:
            print(f'{date} not in data')
        
        market_open = temp.iloc[0,:]
        data = market_open.rename('Market Open').to_frame().T

        market_close = temp.iloc[-1,:]
        if market_close.name.hour == 15 and market_close.name.minute == 59:
            data = data.append(market_close.rename('Market Close'))
            if print_summary:
                temp2 = np.round(pd.concat([market_open, market_close, market_close - market_open], axis = 1),2)
                temp2.columns = ['Open','Close','Market Close - Market Open']
                display(temp2)
            
        for timestamp in timestamps:
            if timestamp in temp.index: data = data.append(temp.loc[timestamp].rename(timestamp))
            else: print(f'{timestamp} not in data')
        
    if use_plotly:
        return plotly_curves(data, 
                      date = date,
                      daily = daily, 
                                xlim = xlim, 
                                ylim_lower = ylim_lower, 
                                ylim_upper = ylim_upper,
                            return_figure=return_figure)
    else: 
        return plt_curves(data,
                   daily = daily, 
                   xlim = xlim, 
                   ylim_lower = ylim_lower, 
                   ylim_upper = ylim_upper,
                         return_figure=return_figure)

def plt_curves(df, daily, xlim = 30, ylim_lower = 270, ylim_upper = 350, return_figure = False):
    fig, ax = plt.subplots(figsize=(22, 12))
    for index, row in df.iterrows():
        if isinstance(index, datetime): 
            if daily: label = index.strftime('%Y-%m-%d')
            else: label = index.strftime('%Y-%m-%d %H-%M')
        elif isinstance(index, str): 
            label = index
        else:
            label = None
        ax.plot(df.columns.tolist(), row.values, label = label)
    
    ax.set_xlabel("Maturity")
    ax.set_ylabel("YTW")
    ax.grid()
    ax.legend()
    plt.tight_layout()
    if return_figure: return ax

def plotly_curves(df, date, daily, xlim = 30, ylim_lower = 270, ylim_upper = 350, return_figure=False):
    figure = go.Figure()
    for index, row in df.iterrows():
        if isinstance(index, datetime): 
            if daily: label = index.strftime('%Y-%m-%d')
            else: label = index.strftime('%Y-%m-%d %H-%M')
        elif isinstance(index, str): 
            label = index
        else:
            label = None
            
        figure.add_trace(go.Scatter(x = row.index.tolist(), y= row.values, name = label))
    
    figure.update_xaxes(title="Maturity")
    figure.update_yaxes(title="YTW")
    
    if daily: title = f'Yield Curves for {min(df.index)} to {max(df.index)}'
    else: title = f'Realtime Yield Curves for {date}'
    
    figure.update_layout(
        title_text=title,
        title_x = 0.5,
        title_y = 0.93,
        autosize=False,
        width=1600,
        height=1000)
    
    if return_figure: return figure
    else: figure.show()


# def plot_realtime_curves(date, minute_curves, print_summary=False,  xlim = 30, ylim_lower = 270, ylim_upper = 350, hourly_interval = 2, use_plotly= False ):

#     if use_plotly: figure = go.Figure()
#     if not use_plotly: fig, ax = plt.subplots(figsize=(25,15))

#     timestamps = [pd.to_datetime(f'{date} 09:30') + pd.offsets.Hour(hourly_interval*i) for i in range(1,int(((6/hourly_interval)+1)))]
#     for timestamp in timestamps:
#         try:
#             if use_plotly:
#                 figure.add_trace(go.Scatter(x=minute_curves.loc[timestamp].index, 
#                                         y=minute_curves.loc[timestamp].values,
#                                        mode = 'lines',
#                                        name=timestamp.strftime('%H:%M')))
#             else: minute_curves.loc[timestamp].plot(ax=ax,cmap='viridis')
#         except:
#             print(f'Curve for {timestamp} unavailable')

#     market_open = minute_curves.loc[date].iloc[0,:]

#     if use_plotly: 
#             figure.add_trace(go.Scatter(x=market_open.index, 
#                                 y=market_open.values,
#                                mode = 'lines',
#                                name='Market Open',
#                                        line_color='grey'))
#     else: market_open.plot(ax=ax, label='Market Open', c='orange')

#     market_close = minute_curves.loc[date].iloc[-1,:]

#     if market_close.name.hour == 15 and market_close.name.minute == 59:
#         if use_plotly:
#             figure.add_trace(go.Scatter(x=market_close.index, 
#                                 y=market_close.values,
#                                mode = 'lines',
#                                name='Market Close', line_color='black'))
#         else: market_close.plot(ax=ax, label='Market Close', c='red')

#         if print_summary:
#             temp = np.round(pd.concat([market_open, market_close, market_close - market_open], axis = 1),2)
#             temp.columns = ['Open','Close','Market Close - Market Open']
#             fig.suptitle(f'{temp}')

#     if use_plotly:
#         figure.update_xaxes(title="Maturity")
#         figure.update_yaxes(title="YTW")
#         figure.update_layout(
#                             title_text=f'Realtime Yield Curves for {date}',
#                             title_x = 0.5,
#             title_y = 0.93,
#                             autosize=False,
#                             width=1600,
#                             height=1000,)
#         return figure
#     else: 
#         ax.legend()    
#         ax.set_xlim(0,xlim)
#         ax.set_ylim(ylim_lower, ylim_upper)

# def plot_curves(df, print_summary=False,  xlim = 30, ylim_lower = 270, ylim_upper = 350, hourly_interval = 2, use_plotly= False ):
#     figure = go.Figure()
#     for index, row in df.iterrows():
#         figure.add_trace(go.Scatter(x = t, y= row.values, name = index.strftime('%Y-%m-%d')))

#     figure.update_xaxes(title="Maturity")
#     figure.update_yaxes(title="YTW")
#     figure.update_layout(
#         title_text=f'Yield Curves for {min(df.index)} to {max(df.index)}',
#         title_x = 0.5,
#         title_y = 0.93,
#         autosize=False,
#         width=1600,
#         height=1000)

#     return figure
