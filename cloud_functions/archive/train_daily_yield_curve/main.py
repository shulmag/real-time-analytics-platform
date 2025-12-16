
import numpy as np
import pandas as pd
from yieldcurve import *

PROJECT_ID = "eng-reactor-287421"
TABLE_ID_model = "eng-reactor-287421.yield_curves.nelson_siegel_coef_daily"
TABLE_ID_scaler = "eng-reactor-287421.yield_curves.standardscaler_parameters_daily"

# Default ridge regularization penalty size based on initial hyperparameter tuning.
alpha = 0.001

# Default shape parameter based on initial hyperparameter tuning. This affects the curvature and slope of the nelson-siegel curve
# higher values generally imply a straighter, more monotonic yield curve (particularly at maturities < 1)
L = 17


def getSchema_model():
    '''
    This function returns the schema required for the bigquery table storing the nelson siegel coefficients. These entries are dated, and
    contain the model's constant, the coefficient on the exponential and the coefficient on the laguerre components.
    '''

    schema = [
        bigquery.SchemaField("date", "DATE", "REQUIRED"),
        bigquery.SchemaField("const", "FLOAT", "REQUIRED"),
        bigquery.SchemaField("exponential", "FLOAT", "REQUIRED"),
        bigquery.SchemaField("laguerre", "FLOAT", "REQUIRED"),
    ]
    return schema


def getSchema_scaler():
    '''
    This function returns the schema required for the bigquery table storing the sklearn StandardScaler's parameters siegel coefficients.
    These entries are dated, and since we have two variables to scale (the exponential and laguerre), we have two means and two standard
    deviations respectively.
    '''

    schema = [
        bigquery.SchemaField("date", "DATE", "REQUIRED"),
        bigquery.SchemaField("exponential_mean", "FLOAT", "REQUIRED"),
        bigquery.SchemaField("exponential_std", "FLOAT", "REQUIRED"),
        bigquery.SchemaField("laguerre_mean", "FLOAT", "REQUIRED"),
        bigquery.SchemaField("laguerre_std", "FLOAT", "REQUIRED"),
    ]
    return schema


def uploadData(df: pd.DataFrame, TABLE_ID: str):
    '''
    This function updates a given bigquery table by appending a dataframe to it. The write_disposition 'WRITE_APPEND' means that the table
    is appended to the bigquery table rather than replacing it.

    Since we have different schemas for the model parameters and scaler parameters, an additional conditional statement is added to avoid
    needing two different functions

    Parameters
    df:pd.DataFrame
    TABLE_ID:str path of the bigquery table to upload to
    '''
    client = bigquery.Client(project=PROJECT_ID, location="US")
    if TABLE_ID == TABLE_ID_model:
        job_config = bigquery.LoadJobConfig(
            schema=getSchema_model(), write_disposition="WRITE_APPEND"
        )
    elif TABLE_ID == TABLE_ID_scaler:
        job_config = bigquery.LoadJobConfig(
            schema=getSchema_scaler(), write_disposition="WRITE_APPEND"
        )
    else:
        raise ValueError

    job = client.load_table_from_dataframe(df, TABLE_ID, job_config=job_config)

    try:
        job.result()
        print("Upload Successful")
    except Exception as e:
        print("Failed to Upload")
        raise e


def main(args):
    '''
    This is the main function which retrieves the S&P index and maturity data from bigquery and uses the most recent entry to train a new
    nelson-siegel yield curve model. The parameters of that model along with the scaler are recorded and uploaded to bigquery.
    '''

    # Get the last row of the bigquery tables of S&P index values and maturities
    maturity_data = load_maturity_bq()
    index_data = load_index_yields_bq()

    assert (
        maturity_data.index[0] == index_data.index[0]
    )  # check that we got the same dates, this should be the case if both tables are up to date

    coef_df = pd.DataFrame()
    scaler_df = pd.DataFrame()

    # This for loop is just here incase we want to train multiple models for multiple days (ie if we want to refresh the entire bigquery table after changing hyperparameters). For now, it just loops over one row
    # See yieldcurve.py for more details about these functions
    for target_date in list(maturity_data.index.astype(str)):
        maturity_dict = get_maturity_dict(maturity_data, target_date)
        yield_curve_df = get_yield_curve_df(index_data, target_date, maturity_dict)
        X, y = get_NL_inputs(yield_curve_df, L)
        scaler, ridge = run_NL_ridge(X, y, alpha=alpha)
        predictions = ridge.predict(scaler.transform(X))

        # Retrieve model parameters
        const = ridge.intercept_
        exponential = ridge.coef_[0]
        laguerre = ridge.coef_[1]

        # Retrieve scaler parameters
        exponential_mean = scaler.mean_[0]
        exponential_std = np.sqrt(scaler.var_[0])
        laguerre_mean = scaler.mean_[1]
        laguerre_std = np.sqrt(scaler.var_[1])

        # Convert date to datetime object
        date = pd.to_datetime(target_date).date()

        coef = pd.DataFrame(
            {
                'date': date,
                'const': const,
                'exponential': exponential,
                'laguerre': laguerre,
            },
            index=[0],
        )

        scaler = pd.DataFrame(
            {
                'date': date,
                'exponential_mean': exponential_mean,
                'exponential_std': exponential_std,
                'laguerre_mean': laguerre_mean,
                'laguerre_std': laguerre_std,
            },
            index=[0],
        )

        coef_df = coef_df.append(coef)
        scaler_df = scaler_df.append(scaler)

    coef_df = coef_df.reset_index(drop=True)
    scaler_df = scaler_df.reset_index(drop=True)

    # Upload the data to bigquery
    uploadData(coef_df, TABLE_ID_model)

    uploadData(scaler_df, TABLE_ID_scaler)

    return "Done"
