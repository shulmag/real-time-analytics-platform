'''
Description: Functions to preprocess data and fit model.
'''
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from auxiliary_functions import get_values_for_date_from_df


def get_maturity_dict(maturity_data: pd.DataFrame, date: str) -> dict:
    return get_values_for_date_from_df(date, maturity_data, 'maturity_df').to_dict()


def get_NL_inputs(yield_curve_df: pd.DataFrame, tau: float):
    '''Takes the output of the `get_yield_curve_df` function alongside a shape parameter to generate the features of the Nelson-Siegel 
    yield curve. It then creates the exponential feature and the laguerre feature using the `decay_transformation` and
    `laguerre_transformation` corresponding to the given shape parameter, and returns the features (X1, X2) and the labels (ytw).'''
    temp_df = yield_curve_df.copy()

    temp_df['X1'] = (tau * (1 - np.exp(-temp_df['Weighted_Maturity'] / tau)) / temp_df['Weighted_Maturity'])
    temp_df['X2'] = (tau * (1 - np.exp(-temp_df['Weighted_Maturity'] / tau)) / temp_df['Weighted_Maturity']) - np.exp(-temp_df['Weighted_Maturity'] / tau)

    X = temp_df[['X1', 'X2']]
    y = temp_df['ytw']
    return X, y


def run_NL_ridge(X: pd.DataFrame | np.ndarray, 
                 y: pd.Series | np.ndarray, 
                 alpha: float = 0.001, 
                 scale: bool = True):
    '''Takes the X and Y values and runs a ridge regression to estimate the nelson-siegel yield curve model. If the sklearn StandardScaler 
    is used, then the scaler object is returned alongside the model object so that the scaler parameters can be saved as well.'''
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        ridge = Ridge(alpha=alpha, random_state=1).fit(X, y)
        return scaler, ridge
    else:
        ridge = Ridge(alpha=alpha, random_state=1).fit(X, y)
        return ridge


def get_coefficient_df(model, timestamp_to_the_minute: datetime) -> pd.DataFrame:
    '''Assumes that `model` is from `sklearn.linear_model`.'''
    return pd.DataFrame({'date': pd.to_datetime(timestamp_to_the_minute),
                         'const': model.intercept_,
                         'exponential': model.coef_[0],
                         'laguerre': model.coef_[1]},
                        index=[0])


def scale_X(X, exponential_mean, exponential_std, laguerre_mean, laguerre_std):
    X['X1'] = (X['X1'] - exponential_mean) / exponential_std
    X['X2'] = (X['X2'] - laguerre_mean) / laguerre_std
    return X


def get_scaler_params(date: str, scaler_daily_parameters: pd.DataFrame):
    most_recent_scalar_daily_parameters = get_values_for_date_from_df(date, scaler_daily_parameters, 'scalar daily parameters')
    (exponential_mean, exponential_std, laguerre_mean, laguerre_std) = most_recent_scalar_daily_parameters.values.flatten()
    return exponential_mean, exponential_std, laguerre_mean, laguerre_std
