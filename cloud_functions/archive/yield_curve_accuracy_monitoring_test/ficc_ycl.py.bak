import numpy as np
import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "eng-reactor-287421"


###Functions to transform maturities into the components in the nelson-siegel model
def decay_transformation(t: np.array, L: float):
    '''
    This function takes a numpy array of maturities (or a single float) and a shape parameter. It returns the exponential function
    calculated from those values. This is the first feature of the nelson-siegel model.

    Parameters:
    t:np.array
    L:float
    '''
    return L * (1 - np.exp(-t / L)) / t


def laguerre_transformation(t, L):
    '''
    This function takes a numpy array of maturities (or a single float) and a shape parameter. It returns the laguerre function
    calculated from those values. This is the second feature of the nelson-siegel model.

    Parameters:
    t:np.array
    L:float
    '''
    return (L * (1 - np.exp(-t / L)) / t) - np.exp(-t / L)


###Functions to load the trained daily yield curve model parameters and daily scaler parameters from bigquery
def load_nelson_siegel_daily_bq():
    '''
    This function loads the model coefficients for the daily nelson-siegel model from bigquery.
    '''
    bq_client = bigquery.Client()
    query = '''
            SELECT * FROM yield_curves_v2.nelson_siegel_coef_daily ORDER BY date asc 
            '''

    query_job = bq_client.query(query)
    df = query_job.result().to_dataframe().set_index('date')
    df = df.drop_duplicates(keep='first')
    return df


def load_scaler_daily_bq():
    '''
    This function loads the scaler parameters used in the sklearn StandardScaler to scale the input data for the daily nelson-siegel model
    during training.
    '''
    bq_client = bigquery.Client()
    query = '''
            SELECT * FROM yield_curves_v2.standardscaler_parameters_daily ORDER BY date asc
            '''

    query_job = bq_client.query(query)
    df = query_job.result().to_dataframe().set_index('date')
    df = df.drop_duplicates(keep='first')
    return df


def load_shape_parameter():
    query = ''' SELECT L, Date as date FROM `eng-reactor-287421.yield_curves_v2.shape_parameters` order by Date asc'''
    df = pd.read_gbq(
        query, project_id='eng-reactor-287421', dialect='standard'
    ).set_index('date')
    df = df
    return df


###Functions used for prediction  Function to
def get_scaled_features(
    t: np.array,
    exponential_mean: float,
    exponential_std: float,
    laguerre_mean: float,
    laguerre_std: float,
    L: float,
):
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
    L:float
    '''
    X1 = (decay_transformation(t, L) - exponential_mean) / exponential_std
    X2 = (laguerre_transformation(t, L) - laguerre_mean) / laguerre_std
    return X1, X2


def predict_ytw(
    t: np.array,
    const: float,
    exponential: float,
    laguerre: float,
    exponential_mean: float,
    exponential_std: float,
    laguerre_mean: float,
    laguerre_std: float,
    L: float,
):
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
    L:float
    '''

    X1, X2 = get_scaled_features(
        t, exponential_mean, exponential_std, laguerre_mean, laguerre_std, L
    )
    return const + exponential * X1 + laguerre * X2


def load_mmd_data(last_modified=False):
    bq_client = bigquery.Client()
    query = '''
            SELECT date, maturity, AAA 
            FROM `eng-reactor-287421.yield_curves.mmd_approximation` 
            order by date desc
            '''

    query_job = bq_client.query(query)
    df = query_job.result().to_dataframe()
    df['AAA'] = df['AAA'].astype(float)

    if last_modified:
        table = bq_client.get_table('eng-reactor-287421.yield_curves.mmd_approximation')
        return df, table.modified

    return df
