'''
Description: Functions that support pricing a list of CUSIPs.
'''
import re
from datetime import datetime

import pandas as pd

from google.cloud import aiplatform
from google.api_core.exceptions import InvalidArgument

from modules.point_in_time_pricing import load_model
from modules.ficc.utils.auxiliary_variables import BINARY, NON_CAT_FEATURES
from modules.ficc.utils.auxiliary_functions import run_multiple_times_before_failing
from modules.ficc.pricing.price import compute_price
# from modules.ficc.pricing.yield_rate import compute_yield

from google.api_core.exceptions import GoogleAPIError


BATCH_SIZE = 10000    # empirically determined to give fast batch predictions


def run_ten_times_before_raising_google_api_error(function: callable) -> callable:
    '''This decorator re-runs the `.predict(...)` function for calling the model when the following error appears:
    `ResourceExhausted: 429 Quota exceeded for aiplatform.googleapis.com/online_prediction_request_throughput.`
    This error arose on 2024-07-18 when VanEck was using the product a very large number of times. The decorator 
    also re-runs the `.predict(...)` function when the model is temporarily not found due to a new model being 
    deployed: `<class 'google.api_core.exceptions.NotFound'>: 404 Prediction on deployed model`. The error first 
    appeared on 2024-07-26.'''
    return run_multiple_times_before_failing((GoogleAPIError,), 10, True)(function)


@run_ten_times_before_raising_google_api_error
def _predict(instances: list[dict], endpoint_id: str) -> list[list]:
    '''Calls the model at `endpoint_id` with `instances`.'''
    project = '964018767272'
    location = 'us-central1'
    api_endpoint = f'{location}-aiplatform.googleapis.com'
    client_options = {'api_endpoint': api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    parameters = {}
    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)

    # perform the try...catch to provide a better error message for easier debugging
    try:
        response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
    except InvalidArgument as e:
        error_message = str(e)    # example: "Failed to process element: 0 key: NON_CAT_AND_BINARY_FEATURES of 'instances' list. Error: INVALID_ARGUMENT: JSON Value: \"NaN\" Type: String is not of expected type: float"
        # Extract element index and feature key
        match = re.search(r'Failed to process element:\s*(\d+)\s+key:\s+(\w+)', error_message)
        if match:
            index = int(match.group(1))
            key = match.group(2)
            entry = instances[index][key]
            print(f"Prediction failed at instance index: {index}, feature: {key} which has the following entry:\n\t{entry}")
            try:
                if key == 'NON_CAT_AND_BINARY_FEATURES':
                    nan_indices = [idx for idx, item in enumerate(entry) if pd.isna(item)]
                    if len(nan_indices) > 0:
                        non_cat_and_binary_features = NON_CAT_FEATURES + BINARY
                        print(f"\tThe following indices in the list contain NaN values: {nan_indices}. These correspond to {[non_cat_and_binary_features[idx] for idx in nan_indices]}.")
            except Exception:
                pass
        else:
            print(f"Failed to extract index/key from error message: {error_message}.")
        raise e
    
    return response.predictions


def predict_spread_with_deployed_keras_model(instances: list[dict]) -> list[list]:
    '''Retrieves yield spread estimates from the yield spread with similar trades model on given a set of `instances`. For 
    real-time pricing, we will never be using the yield spread model without similar trades model.'''
    endpoint_id = '1068587863244800000'    # corresponds to yield spread with similar trades model
    return _predict(instances, endpoint_id)


def predict_dollar_price_with_deployed_keras_model(instances: list[dict]) -> list[list]:
    '''Retrieves dollar price estimates from the dollar price model on given a set of `instances`.'''
    endpoint_id = '1672070213312446464'    # corresponds to dollar price model
    return _predict(instances, endpoint_id)


def predict_with_archived_keras_model(archived_keras_model, instances: list[dict]) -> list[list]:
    'Retrieves estimates from the `archived_keras_model` on a given set of `instances`.'
    return archived_keras_model.predict(instances, batch_size=BATCH_SIZE)


def predict_spread(instances: list[dict], use_yield_spread_with_similar_trades_model: bool = True, datetime_for_point_in_time_pricing: datetime = None):
    '''`datetime_for_point_in_time_pricing` is a datetime object iff we are using an archived model 
    for point-in-time pricing, otherwise it is `None`.'''
    if datetime_for_point_in_time_pricing is None: return predict_spread_with_deployed_keras_model(instances)    # real-time pricing
    archived_model_predicting_yield_spread = load_model(datetime_for_point_in_time_pricing, 'similar_trades_model' if use_yield_spread_with_similar_trades_model else 'yield_spread_model')
    return predict_with_archived_keras_model(archived_model_predicting_yield_spread, instances)


def predict_dollar_price(instances: list[dict], datetime_for_point_in_time_pricing: datetime = None):
    '''`datetime_for_point_in_time_pricing` is a datetime object iff we are using an archived model 
    for point-in-time pricing, otherwise it is `None`.'''
    if datetime_for_point_in_time_pricing is None: return predict_dollar_price_with_deployed_keras_model(instances)    # real-time pricing
    archived_dollar_price_model = load_model(datetime_for_point_in_time_pricing, 'dollar_price_model')
    return predict_with_archived_keras_model(archived_dollar_price_model, instances)


def get_trade_price_from_yield_spread_model(trade) -> tuple[float, datetime | pd.Timestamp]:
    price, calc_date = compute_price(trade, trade.ficc_ytw)
    return abs(price), calc_date    # price should always be positive


## commented out since this function is not used, since we do not display to converted yield from price on the front end when using the dollar price model
# def get_estimated_yield(row):
#     try:
#         ytw, calc_date = compute_yield(row, price=row.price)
#     except Exception as e:
#         print(f'Failed to convert price to yield for {row.cusip}, using yield spread model.')
#         print(f'Error: {e}')
#         ytw, calc_date = None, None
#     return ytw, calc_date
