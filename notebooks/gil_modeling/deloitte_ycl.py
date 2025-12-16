import numpy as np
from google.cloud import bigquery
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar    # used to create a business day defined on the US federal holiday calendar that can be added or subtracted to a datetime
import copy
from functools import wraps
from datetime import datetime

BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())    # used to skip over holidays when adding or subtracting business days

# Constants
PROJECT_ID = 'eng-reactor-287421'
NUM_OF_DAYS_IN_YEAR = 360

# BQ_CLIENT setup
def get_bq_client():
    return bigquery.Client()

BQ_CLIENT = get_bq_client()

# Helper functions
def sqltodf(sql, bq_client):
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()

def diff_in_days_two_dates(date1, date2):
    if isinstance(date1, datetime):
        date1 = date1.date()
    if isinstance(date2, datetime):
        date2 = date2.date()
    return (date1 - date2).days

def cache(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        cache_key = str(args[0]) + str(args[1])
        if cache_key in wrapper.cache:
            output = wrapper.cache[cache_key]
        else:
            output = function(*args)
            wrapper.cache[cache_key] = output
        return output
    wrapper.cache = dict()
    return wrapper

def decay_transformation(t: np.array, L: float):
    return L * (1 - np.exp(-t / L)) / t

def laguerre_transformation(t: np.array, L: float):
    return (L * (1 - np.exp(-t / L)) / t) - np.exp(-t / L)

def load_model_parameters(target_date, nelson_params, scalar_params, shape_parameter):
    target_date = (target_date - (BUSINESS_DAY * 1)).date()
    target_date_shape = copy.deepcopy(target_date)

    while target_date not in nelson_params.keys():
        target_date = (target_date - (BUSINESS_DAY * 1)).date()

    nelson_coeff = nelson_params[target_date].values()
    scalar_coeff = scalar_params[target_date].values()

    while target_date_shape not in shape_parameter.keys():
        target_date_shape = (target_date_shape - (BUSINESS_DAY * 1)).date()
    shape_param = shape_parameter[target_date_shape]['L']

    return nelson_coeff, scalar_coeff, shape_param

def get_scaled_features(t: np.array, exponential_mean: float, exponential_std: float, laguerre_mean: float, laguerre_std: float, shape_paramter: float):
    X1 = (decay_transformation(t, shape_paramter) - exponential_mean) / exponential_std
    X2 = (laguerre_transformation(t, shape_paramter) - laguerre_mean) / laguerre_std
    return X1, X2

def predict_ytw(maturity: np.array,
                const: float,
                exponential: float,
                laguerre: float,
                exponential_mean: float,
                exponential_std: float,
                laguerre_mean: float,
                laguerre_std: float,
                shape_parameter: float):
    X1, X2 = get_scaled_features(maturity, exponential_mean, exponential_std, laguerre_mean, laguerre_std, shape_parameter)
    return const + exponential * X1 + laguerre * X2

@cache
def yield_curve_level(maturity: float, target_date, nelson_params, scalar_params, shape_parameter):
    nelson_siegel_daily_coef, scaler_daily_parameters, shape_param = load_model_parameters(target_date, nelson_params, scalar_params, shape_parameter)
    const, exponential, laguerre = nelson_siegel_daily_coef
    exponential_mean, exponential_std, laguerre_mean, laguerre_std = scaler_daily_parameters
    prediction = predict_ytw(maturity, const, exponential, laguerre, exponential_mean, exponential_std, laguerre_mean, laguerre_std, shape_param)
    return prediction

def get_yield_for_last_duration(row, nelson_params, scalar_params, shape_parameter):
    duration = diff_in_days_two_dates(row['ice_maturity_date'], row['trade_date']) / NUM_OF_DAYS_IN_YEAR
    ycl = yield_curve_level(duration, row['trade_date'], nelson_params, scalar_params, shape_parameter) / 100
    return ycl

def add_yield_curve(data):
    '''Add 'new_ficc_ycl' field to `data`.'''
    nelson_params = sqltodf(f'SELECT * FROM `{PROJECT_ID}.yield_curves_v2.nelson_siegel_coef_daily` order by date desc', BQ_CLIENT)
    nelson_params.set_index('date', drop=True, inplace=True)
    nelson_params = nelson_params[~nelson_params.index.duplicated(keep='first')]
    nelson_params = nelson_params.transpose().to_dict()

    scalar_params = sqltodf(f'SELECT * FROM `{PROJECT_ID}.yield_curves_v2.standardscaler_parameters_daily` order by date desc', BQ_CLIENT)
    scalar_params.set_index('date', drop=True, inplace=True)
    scalar_params = scalar_params[~scalar_params.index.duplicated(keep='first')]
    scalar_params = scalar_params.transpose().to_dict()

    shape_parameter = sqltodf(f'SELECT * FROM `{PROJECT_ID}.yield_curves_v2.shape_parameters` order by Date desc', BQ_CLIENT)
    shape_parameter.set_index('Date', drop=True, inplace=True)
    shape_parameter = shape_parameter[~shape_parameter.index.duplicated(keep='first')]
    shape_parameter = shape_parameter.transpose().to_dict()

    data['new_ficc_ycl'] = data[['ice_maturity_date', 'trade_date']].apply(
        lambda row: get_yield_for_last_duration(row, nelson_params, scalar_params, shape_parameter), axis=1
    )
    data['new_ficc_ycl'] = data['new_ficc_ycl'] * 100
    return data