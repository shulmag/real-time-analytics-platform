'''
'''
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import json

from google.cloud import bigquery
from dateutil import parser
from datetime import datetime 
from pytz import timezone
from sklearn.metrics import mean_absolute_error
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday

# Import Redis helper module
import redis_helper

plt.style.use('fivethirtyeight')
warnings.simplefilter(action='ignore', category=FutureWarning)


BQ_CLIENT = bigquery.Client()
PROJECT_ID = 'eng-reactor-287421'
DATASET_NAME = 'yield_curves_v2'
TABLE_ID = f'{PROJECT_ID}.{DATASET_NAME}.shape_parameters'

t = np.round(np.arange(0.1, 30.1, 0.1), 2)    # rounding to avoid precision issues
KEY_MATURITIES = np.array([5, 10, 15, 20, 25, 26, 27, 28, 29, 30])
eastern = timezone('US/Eastern')
date_hour_format = '%Y-%m-%d %H:%M'
date_format = '%Y-%m-%d'

# Initialize Redis client
REDIS_CLIENT = redis_helper.get_redis_client()


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
    """
    Load yield curve parameters from BigQuery or Redis cache.
    
    Args:
        daily (bool): Whether to load daily or minute-level data
        date_start (str): Start date in YYYY-MM-DD format
        date_end (str): End date in YYYY-MM-DD format
        use_redis (bool): Whether to use Redis cache (read-only)
        
    Returns:
        pd.DataFrame: DataFrame containing yield curve parameters
    """
    # if either start or end date not available, just fetch earliest possible/most recent data 
    if not date_start: date_start = '2021-07-27'
    if not date_end: date_end = datetime.now().strftime(date_format)
    
    # Check if Redis is available and should be used
    if use_redis and REDIS_CLIENT is not None:
        # Get current time in Eastern timezone
        now = datetime.now(eastern)
        
        # For Redis, we'll try a few different timestamp formats:
        # 1. Exact current minute
        # 2. Most recent key in Redis
        
        # Try exact current minute
        current_key = redis_helper.generate_key(date=now)
        cached_data = redis_helper.get_cached_dataframe(REDIS_CLIENT, current_key)
        
        if cached_data is not None:
            # Found data for the exact current minute
            return cached_data
        
        # Try most recent key in Redis
        try:
            all_keys = REDIS_CLIENT.keys('2*')  # Get all keys starting with a year
            if all_keys:
                # Sort keys by timestamp (descending)
                sorted_keys = sorted([k.decode('utf-8') for k in all_keys], reverse=True)
                latest_key = sorted_keys[0]
                
                cached_data = redis_helper.get_cached_dataframe(REDIS_CLIENT, latest_key)
                if cached_data is not None:
                    return cached_data
        except Exception as e:
            # If there's an error getting keys, fall back to BigQuery
            pass
        
        # If we got here, Redis didn't have usable data - query BigQuery
    
    # If Redis is not available or no cached data found, query BigQuery
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
    
    # NOTE: We no longer write to Redis - read-only implementation
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


"""Helper functions perviously in spreadsV2. Because these are the only functions used in the server from
    the update spreads cloud function, they are now in auxiliary_functions."""
def get_existing_data(bq_client, business_days, maturity, quantity):
    """Get existing data for the specified business days"""

    date_list = "', '".join([d.strftime('%Y-%m-%d') for d in business_days])
    query = (
        "SELECT "
        "date_et as date, "
        "CONCAT('$', ROUND(avg_spread, 3)) as AvgSpreadDollar, "
        "CONCAT(ROUND(avg_spread_pct, 3), '%') as AvgSpreadPercent, "
        "num_cusips as NumCUSIPs "
        "FROM `eng-reactor-287421.analytics_data_source.daily_dollar_spread_averages` "
        f"WHERE date_et IN ('{date_list}') "
        "AND rating_category = 'IG' "
        f"AND maturity_range = '{maturity}' "
        f"AND quantity = {quantity} "
        "ORDER BY date DESC"
    )
    
    df = bq_client.query(query).to_dataframe()
    existing_dates = set(df['date'].tolist()) if not df.empty else set()
    
    return df, existing_dates

# Calendar and timezone constants
class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    """Custom US Federal Holiday calendar that includes Good Friday"""
    rules = USFederalHolidayCalendar.rules + [GoodFriday]

def get_last_n_business_days(n=10):
    ''' Gets the last 10 business days, including today. '''

    BUSINESS_DAY = CustomBusinessDay(calendar=USHolidayCalendarWithGoodFriday())
    EASTERN = timezone('US/Eastern')
    today = datetime.now(EASTERN).date()
    business_days = [today]  # Start with today
    current_date = today
    
    while len(business_days) < n:
        current_date = (pd.Timestamp(current_date) - BUSINESS_DAY).date()
        business_days.append(current_date)
    return business_days