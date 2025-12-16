"""
Description: Implementation of the Nelson-Siegel interest rate model to predict the yield curve. Nelson-Siegel coefficients are used from a dataframe instead of grabbing them from memory store.
10-18: Added function fetch_aaa_coeff to fetch coeffs from aaa tables, modified load_model_parameters to call fetch_aaa_coeff instead of reading from pickle file.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from google.cloud import bigquery

from modules.ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from modules.ficc.utils.auxiliary_functions import cache_output
from modules.ficc.utils.diff_in_days import diff_in_days_two_dates
from modules.ficc.utils.yc_data import get_yc_data


PROJECT_ID = "eng-reactor-287421"


def get_duration(row) -> float:
    """Get the duration in years from `row`.
    TODO: Write a better description of why we use the following heuristic."""
    start_date = row[
        "trade_date"
    ]  # using `trade_date` instead of `settlement_date` because `settlement_date` is a day (or more) after `trade_date` and this may cause it to be equal or after `end_date` in some cases causing the duration to be nonpositive causing `RuntimeWarning: invalid value encountered in scalar divide` in `decay_transformation(...)` and `laguerre_transformation(...)` functions because of division by zero
    maturity_date = row["maturity_date"]
    is_called, is_callable, last_calc_day_cat = (
        row["is_called"],
        row["is_callable"],
        row["last_calc_day_cat"],
    )
    if is_called is True and pd.notna(row["refund_date"]):
        end_date = row["refund_date"]
    elif is_callable is False:
        end_date = maturity_date
    elif last_calc_day_cat == 0 and pd.notna(
        row["next_call_date"]
    ):  # sometimes the case that `next_call_date` is null, but `first_call_date` is not null, and in this case, the correct value is perhaps `first_call_date` but requires complicated upstream code to correct it and deeper investigation before making the change here (1.3% of materialized trade history on 2025-05-13, and only new issues)
        end_date = row["next_call_date"]
    elif last_calc_day_cat == 1 and pd.notna(row["par_call_date"]):
        end_date = row["par_call_date"]
    else:
        end_date = maturity_date
    #   Note: on 8/15/2025 we discovered that reference data provided files included values for next call date
    #   equal to the file date leading to cases where next call date is equal to today. The below fix is
    #   a rough heuristic.

    if (diff_in_days_two_dates(end_date, start_date)) < 1:
        end_date = maturity_date

    return diff_in_days_two_dates(end_date, start_date) / NUM_OF_DAYS_IN_YEAR


@cache_output
def get_ficc_ycl_for_target_trade(row, current_datetime: datetime):
    """Compute the yield curve level for the target trade using the last duration. `current_datetime` is used
    solely for caching."""
    return yield_curve_level(get_duration(row), row["trade_datetime"].strftime("%Y-%m-%d:%H:%M"))


def decay_transformation(t: np.array, L: float):
    """Takes a numpy array of maturities (or a single float) and a shape parameter, and returns the exponential function
    calculated from those values. This is the first feature of the Nelson-Siegel model."""
    return L * (1 - np.exp(-t / L)) / t


def laguerre_transformation(t: np.array, L: float):
    """Takes a numpy array of maturities (or a single float) and a shape parameter, and returns the laguerre function
    calculated from those values. This is the second feature of the Nelson-Siegel model."""
    return (L * (1 - np.exp(-t / L)) / t) - np.exp(-t / L)


def fetch_aaa_coeff(dt: datetime) -> pd.DataFrame:
    """
    Fetch a 1x3 DataFrame (const, exponential, laguerre) from
    eng-reactor-287421.yield_curves_v2.nelson_siegel_coef_minute
    where `date` (BigQuery DATETIME) exactly equals the provided naive datetime.
    Returns:
        pd.DataFrame with one row and columns: ['const', 'exponential', 'laguerre'].
    """
    if not isinstance(dt, datetime):
        raise TypeError("dt must be a datetime.datetime")
    if dt.tzinfo is not None:
        # Strip any tz to keep it naive for BigQuery DATETIME
        dt = dt.replace(tzinfo=None)
    client = bigquery.Client()
    query = """
    SELECT const, exponential, laguerre
    FROM `eng-reactor-287421.yield_curves_aaa.nelson_siegel_coef_minute`
    WHERE `date` = @dt
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("dt", "DATETIME", dt)]
    )

    df = client.query(query, job_config=job_config).to_dataframe()

    if df.empty:
        raise ValueError(f"No row found for date == {dt.isoformat(sep=' ')}")
    # Ensure 1×3 shape and column order
    return df.loc[[df.index[0]], ["const", "exponential", "laguerre"]]


def load_model_parameters(target_datetime):
    """Grabs the Nelson-Siegel coefficients and standard scalar coefficient from the input dataframes."""
    # target_datetime = datetime.strptime(target_datetime, "%Y-%m-%d:%H:%M")
    yield_curve_parameters = get_yc_data(target_datetime)

    nelson_coeff = fetch_aaa_coeff(target_datetime)
    scalar_coeff = yield_curve_parameters["scalar_values"]
    shape_parameter = yield_curve_parameters["shape_parameter"]
    print(nelson_coeff)

    return nelson_coeff, scalar_coeff, shape_parameter


def get_scaled_features(
    t: np.array,
    exponential_mean: float,
    exponential_std: float,
    laguerre_mean: float,
    laguerre_std: float,
    shape_parameter: float,
):
    """This function takes as input the parameters loaded from the scaler parameter table in bigquery on a given day, alongside an array (or a
    single float) value to be scaled as input to make predictions. It then manually recreate the transformations from the sklearn
    StandardScaler used to scale data in training by first creating the exponential and laguerre functions then scaling them.
    """
    X1 = (decay_transformation(t, shape_parameter) - exponential_mean) / exponential_std
    X2 = (laguerre_transformation(t, shape_parameter) - laguerre_mean) / laguerre_std
    return X1, X2


def predict_yield_curve_level(
    maturity: np.array,
    const: float,
    exponential: float,
    laguerre: float,
    exponential_mean: float,
    exponential_std: float,
    laguerre_mean: float,
    laguerre_std: float,
    shape_parameter: float,
):
    """Wrapper function that takes the prediction inputs, the scaler parameters and the model parameters from a given day. It then
    scales the input using the get_scaled_features function to obtain the model inputs, and predicts the yield-to-worst implied by the
    nelson-siegel model on that day. Because the Nelson-Siegel model is linear, we can do a simple calculation.
    """
    X1, X2 = get_scaled_features(
        maturity, exponential_mean, exponential_std, laguerre_mean, laguerre_std, shape_parameter
    )
    results = const + exponential * X1 + laguerre * X2
    print(f"maturity: {maturity}")
    print(f"results: {results}")
    return results


def yield_curve_level(maturity, target_datetime):
    """Takes as input two arguments: the maturity we want the yield-to-worst for and the target datetime from which
    we want the yield curve used in the ytw calculations to be from. There are several conditional statements
    to deal with different types of exceptions.

    The cloud function returns a json containing the status (Failed or Success), the error message (if any)
    and the result (nan if calculation was unsuccessful)."""
    # If a target_datetime is provided but it is in an invalid format, then the correct values from the model and scaler parameters cannot be retrieved, and an error is also returned.
    nelson_siegel_minute_coef, scaler_daily_parameters, shape_parameter = load_model_parameters(
        target_datetime
    )

    if len(nelson_siegel_minute_coef) == 1:
        try:
            const, exponential, laguerre = nelson_siegel_minute_coef.values[0]
        except Exception as e:
            _, const, exponential, laguerre = nelson_siegel_minute_coef.values[0]

    elif len(nelson_siegel_minute_coef.shape) > 1 and len(nelson_siegel_minute_coef) > 1:
        # error = 'Multiple rows for target date in nelson_siegel_coef_daily, taking first one. Check bigquery table.'
        const, exponential, laguerre = nelson_siegel_minute_coef.iloc[0, :]
    elif len(nelson_siegel_minute_coef) > 1:
        const, exponential, laguerre = nelson_siegel_minute_coef
    else:
        raise Exception("Nelson-Siegel coefficients for the selected dates do not exist")

    if len(scaler_daily_parameters) == 1:
        try:
            exponential_mean, exponential_std, laguerre_mean, laguerre_std = (
                scaler_daily_parameters.values[0]
            )
        except Exception as e:
            _, exponential_mean, exponential_std, laguerre_mean, laguerre_std = (
                scaler_daily_parameters.values[0]
            )
    elif len(scaler_daily_parameters.shape) > 1 and len(scaler_daily_parameters) > 1:
        # error = 'Multiple rows for target date in standardscaler_parameters_daily, taking first one. Check bigquery table.'
        exponential_mean, exponential_std, laguerre_mean, laguerre_std = (
            scaler_daily_parameters.iloc[0, :]
        )
    elif len(scaler_daily_parameters) > 1:
        exponential_mean, exponential_std, laguerre_mean, laguerre_std = scaler_daily_parameters
    else:
        raise Exception("Failed to grab scalar coefficient, it does not exist")

    return predict_yield_curve_level(
        maturity,
        const,
        exponential,
        laguerre,
        exponential_mean,
        exponential_std,
        laguerre_mean,
        laguerre_std,
        shape_parameter,
    )
