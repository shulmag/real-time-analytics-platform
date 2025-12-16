'''
'''
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io

from google.cloud import bigquery
from dateutil import parser
from datetime import datetime 
from pytz import timezone
from sklearn.metrics import mean_absolute_error


# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/user/ficc/ficc/mitas_creds.json'


plt.style.use('fivethirtyeight')
warnings.simplefilter(action='ignore', category=FutureWarning)


BQ_CLIENT = bigquery.Client()
PROJECT_ID = 'eng-reactor-287421'
DATASET_NAME = 'yield_curves_v2'
TABLE_ID = f'{PROJECT_ID}.{DATASET_NAME}.shape_parameters'

t = np.round(np.arange(0.1, 30.1, 0.1), 2)    # rounding to avoid precision issues
KEY_MATURITIES = np.array([1, 2, 5, 10, 15, 30])
eastern = timezone('US/Eastern')
date_hour_format = '%Y-%m-%d %H:%M'
date_format = '%Y-%m-%d'


################## LOADING DATA ##################
def sqltodf(sql, bq_client):
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()


def get_mmd_data(date_start=None, date_end=None):
    if not date_start: date_start = '2021-07-27'
    if not date_end: date_end = datetime.now().strftime(date_format)
    query = f'SELECT date, maturity, AAA FROM `{PROJECT_ID}.yield_curves.mmd_approximation` WHERE date <= "{date_end}" AND date >= "{date_start}" ORDER BY date DESC'
    mmd_data = sqltodf(query, BQ_CLIENT)
    mmd_data.AAA = mmd_data.AAA.astype(float)
    return pd.pivot_table(mmd_data, index = 'date', columns='maturity', values='AAA') * 100


def get_maturity_dict(maturity_df, date):
    temp_df = maturity_df.loc[date].T
    temp_dict = dict(zip(temp_df.index, temp_df.values))
    return temp_dict


def get_scaler_data(date_start=None, date_end=None):
    if not date_start: date_start = '2021-07-27'
    if not date_end: date_end = datetime.now().strftime(date_format)

    scaler_query = f'SELECT * FROM `{PROJECT_ID}.{DATASET_NAME}.standardscaler_parameters_daily` WHERE date <= "{date_end}" AND date >= "{date_start}" ORDER BY date ASC'
    daily_ycl_scaler = sqltodf(scaler_query, BQ_CLIENT)
    daily_ycl_scaler = daily_ycl_scaler.loc[~daily_ycl_scaler.duplicated(subset='date')]
    return daily_ycl_scaler


def get_shape_data(date_start = None, date_end = None):
    if not date_start: date_start = '2021-07-27'
    if not date_end: date_end = datetime.now().strftime(date_format)

    shape_query = f'SELECT * FROM `{PROJECT_ID}.{DATASET_NAME}.shape_parameters` WHERE date <= "{date_end}" AND date >= "{date_start}" ORDER BY date ASC'
    daily_ycl_shape = sqltodf(shape_query, BQ_CLIENT).rename({'Date': 'date'}, axis=1)
    daily_ycl_shape = daily_ycl_shape.loc[~daily_ycl_shape.duplicated(subset='date')]
    return daily_ycl_shape


def load_yield_curve_params(daily: bool, date_start=None, date_end=None, use_redis=False):
    # if either start or end date not available, just fetch earliest possible/most recent data 
    if not date_start: date_start = '2021-07-27'
    if not date_end: date_end = datetime.now().strftime(date_format)

    if daily: 
        yield_query = f'SELECT * FROM `{PROJECT_ID}.{DATASET_NAME}.nelson_siegel_coef_daily` WHERE date <= "{date_end}" AND date >= "{date_start}" ORDER BY date ASC'
    else:
        # note that timestamp must be included for bigquery queries filtering by datetime columns, else bigquery will take 00:00 as default time
        yield_query = f'SELECT * FROM `{PROJECT_ID}.{DATASET_NAME}.nelson_siegel_coef_minute` WHERE date <= "{date_end} 23:59:00" AND date >= "{date_start} 00:00:00" ORDER BY date ASC'
    
    scaler_query = f'SELECT * FROM `{PROJECT_ID}.{DATASET_NAME}.standardscaler_parameters_daily` WHERE date <= "{date_end}" AND date >= "{date_start}" ORDER BY date ASC'
    shape_query = f'SELECT * FROM `{PROJECT_ID}.{DATASET_NAME}.shape_parameters` WHERE date <= "{date_end}" AND date >= "{date_start}" ORDER BY date ASC'
    
    ycl_coef = sqltodf(yield_query, BQ_CLIENT)
    daily_ycl_scaler = sqltodf(scaler_query, BQ_CLIENT)
    daily_ycl_shape = sqltodf(shape_query, BQ_CLIENT).rename({'Date': 'date'}, axis=1)
    
    if daily:
        result = pd.merge(ycl_coef, daily_ycl_scaler, on='date').drop_duplicates()
        result = pd.merge(result, daily_ycl_shape, on='date').drop_duplicates()
    else:
        # for daily yield curve, we get unique dates from the yield curve table, map those to the last available previous day and join them with the scaler and shape tables for efficiency
        date_reference = ycl_coef.date.dt.date.drop_duplicates()
        
        def prev_day(date):
            # gets the last available date in the scaler and shape tables
            refs = daily_ycl_scaler.date.to_list()
            date = date -  pd.Timedelta(days=1)
                
            # if the previous day is outside of the first or last date in the scaler table, then take the earliest or most recent date 
            if date <= min(refs):
                return min(refs)

            if date > max(refs):
                return max(refs)

            # if data not inside the scaler table, go back by one day and try again 
            while date not in refs:
                date = date -  pd.Timedelta(days=1)

            return date
        
        date_reference = dict(zip(date_reference, date_reference.apply(lambda x: prev_day(x))))
        ycl_coef['date_date'] = ycl_coef.date.dt.date.apply(lambda x: date_reference.get(x))
        
        
        result = pd.merge(daily_ycl_shape, daily_ycl_scaler, on='date').drop_duplicates()
        result = pd.merge(ycl_coef, result,  left_on='date_date', right_on='date', suffixes=['', '_right']).drop_duplicates()
        result.drop(['date_date', 'date_right'], axis=1, inplace=True)
        
    result['date'] = pd.to_datetime(result['date'])
    result.set_index('date', drop=True, inplace=True)
    return result


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


def predict_ytw(t: np.array, 
                const: float, 
                exponential: float, 
                laguerre: float, 
                exponential_mean: float, 
                exponential_std: float, 
                laguerre_mean: float, 
                laguerre_std: float, 
                L: float):
    '''This is a wrapper function that takes the prediction inputs, the scaler parameters and the model parameters from a given day. It 
    then scales the input using the get_scaled_features function to obtain the model inputs, and predicts the yield-to-worst implied 
    by the nelson-siegel model on that day. Because the nelson-siegel model is linear, we can do a simple calculation.'''
    X1, X2 = get_scaled_features(t, exponential_mean, exponential_std, laguerre_mean, laguerre_std, L)
    return const + exponential*X1 + laguerre*X2


def decay_transformation(t, L):
    return L*(1-np.exp(-t/L))/t


def laguerre_transformation(t, L):
    return (L*(1-np.exp(-t/L))/t) -np.exp(-t/L)


def get_scaled_features(t: np.array, 
                        exponential_mean: float, 
                        exponential_std: float, 
                        laguerre_mean: float, 
                        laguerre_std: float, 
                        L: float):
    '''This function takes as input the parameters loaded from the scaler parameter table in bigquery on a given day, alongside an 
    array (or a single float) value to be scaled as input to make predictions. It then manually recreate the transformations from the 
    sklearn StandardScaler used to scale data in training by first creating the exponential and laguerre functions then scaling them.'''
    X1 = (decay_transformation(t, L) - exponential_mean) / exponential_std 
    X2 = (laguerre_transformation(t, L) - laguerre_mean) / laguerre_std 
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


############### PLOTTING CURVES ###############
def save_fig_in_memory(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format = 'png')
    buf.seek(0)
    return buf


def plt_curves(df, daily, return_figure = False, title = None):
    fig, ax = plt.subplots(figsize=(22, 12))
    for index, row in df.iterrows():
        if isinstance(index, datetime): 
            if daily: label = index.strftime(date_format)
            else: label = index.strftime('%Y-%m-%d %H-%M')
        elif isinstance(index, str): 
            label = index
        else:
            label = None
        ax.plot(df.columns.tolist(), row.values, label = label)
    
    ax.set_xlabel('Maturity')
    ax.set_ylabel('YTW')
    if title: ax.set_title(title)
    ax.grid(True)
    ax.legend()
    if return_figure: return fig


def plot_series_plus_extreme(data, ax, extremes):
    if extremes:
        idx_max = (data - data[0]).apply(abs).idxmax()
        date_min = data.index[0]
        val_end = data.loc[idx_max]
        val_min = data[0]

        ax.plot([date_min, date_min], [val_min, val_end], c='r', linewidth=2, linestyle='--')
        ax.plot([date_min, idx_max], [val_end, val_end], c='r', linewidth=2, linestyle='-', marker ='o', markersize=3)
    data.plot(linewidth=3, ax=ax)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title(f'Maturity: {int(data.name)}', size=15)


def plot_intraday_extremes(ycdf, today, extremes=False):
    fig, ax = plt.subplots(3, 2, figsize=(20, 20))
    for axes_idx, axes in enumerate(ax.flatten()):
        plot_series_plus_extreme(ycdf[today][KEY_MATURITIES[axes_idx]], axes, extremes)
    fig.suptitle(f'Smoothed Ficc Real-time Yield Curve on {today}\n & Largest Intraday Move from Open to Market High/Low', y=0.935, fontsize=20)
    return fig
