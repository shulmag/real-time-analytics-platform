'''
Updated on 05/26/2023 by Developer. 

This module contains functions that return data for our product:
1) YTW and Price for CUSIP
2) Batch Pricing
3) Yield Curve 
4) Stats
5) Other functions, not all has a UI right now
'''
import aiohttp
import asyncio

import redis
import pandas as pd
pd.options.mode.chained_assignment = None    # default='warn'; this helps clean up the logs when debugging
import numpy as np
import requests
import holidays
import time
import math
import multiprocess as mp

from google.cloud import bigquery, workflows_v1beta, storage
from google.cloud.workflows import executions_v1beta
from google.cloud.workflows.executions_v1beta.types import executions

from pickle5 import pickle
from google.cloud import aiplatform
from flask import jsonify, make_response
from datetime import datetime, timedelta
from pytz import timezone

from modules.ficc.utils.auxiliary_functions import sqltodf, compare_dates
from modules.ficc.utils.diff_in_days import diff_in_days, diff_in_days_two_dates
from modules.ficc.utils.auxiliary_variables import PREDICTORS, CATEGORICAL_FEATURES, NON_CAT_FEATURES, BINARY, NUM_OF_DAYS_IN_YEAR
from modules.ficc.data.process_data import process_data
from modules.ficc.pricing.price import compute_price
from modules.ficc.utils.nelson_siegel_model import yield_curve_level
from modules.ficc.utils.gcp_storage_functions import download_pickle_file

import smtplib
from email.mime.text import MIMEText

storage_client = storage.Client()
get_treasury_rate_df = lambda: download_pickle_file(storage_client, 'treasury_rate_df', 'treasury_rate_df.pkl')    # create function instead of loading as a global variable so we can get the most updated value every time we price a CUSIP
get_encoders = lambda: download_pickle_file(storage_client, 'automated_training', 'encoders.pkl')    # create function instead of loading as a global variable so we can get the most updated value every time we price a CUSIP

SEQUENCE_LENGTH = 2    # number of trades in the same CUSIP trade history
NUM_FEATURES = 6    # number of features per trade in the history

EASTERN = timezone('US/Eastern')    # all datetime objects must use ET time zone

# Variables for date functions
ONE_DAY = timedelta(days=1)
TWO_DAYS = timedelta(days=2)
HOLIDAYS_US = holidays.US()

# Variables for string formatting of datetime objects
YEAR_MONTH_DAY = '%Y-%m-%d'
HOUR_MIN = '%H:%M'
HOUR_MIN_SEC = '%H:%M:%S'
MONTH_DAY_YEAR = '%m-%d-%Y'
YEAR_MONTH_DAY_HOUR_MIN_SEC = YEAR_MONTH_DAY + ' ' + HOUR_MIN_SEC
datetime_display_format = MONTH_DAY_YEAR + ' ' + HOUR_MIN_SEC

NUM_CHARS_IN_YEAR_MONTH_DAY = 10    # 4 characters for the year, 2 characters for the month, 2 characters for the day, and two additional characters for the separators

# Predictors list for the neural network
NN_PRED_LIST = ['trade_history_input', 'target_attention_input', 'NON_CAT_AND_BINARY_FEATURES'] + CATEGORICAL_FEATURES
TRADE_MAPPING = {'D': [0, 0], 'S': [0, 1], 'P': [1, 0]}
NUMERICAL_ERROR = -1    # value used to fill in a numerical field when there is an error

# Precision of price and yield values
DISPLAY_PRECISION = 3
LOGGING_PRECISION = 5
round_for_logging = lambda value: np.round(value, LOGGING_PRECISION)

# Used to create the scheme for logging
LOGGING_FEATURES = {'user': 'string', 
                    'api_call': 'BOOLEAN', 
                    'time': 'timestamp', 
                    'cusip': 'string', 
                    'direction': 'string', 
                    'quantity': 'integer', 
                    'ficc_price': 'numeric', 
                    'ficc_ytw': 'numeric', 
                    'yield_spread': 'numeric', 
                    'ficc_ycl': 'numeric', 
                    'calc_date': 'string', 
                    'daily_schoonover_report': 'BOOLEAN', 
                    'real_time_yield_curve': 'BOOLEAN', 
                    'batch': 'BOOLEAN', 
                    'show_similar_bonds': 'BOOLEAN', 
                    'error': 'BOOLEAN'}
RECENT_FEATURES = ['yield_spread', 'yield_spread2', 'yield_spread3', 'yield_spread4', 'yield_spread5']

# Features used in the final batch pricing csv
FEATURES_FOR_OUTPUT_CSV = ['cusip', 'quantity', 'trade_type', 'ytw', 'price', 'yield_to_worst_date', 'coupon', 'security_description', 'maturity_date']
TRADE_TYPE_CODE_TO_TEXT = {'D': 'Inter-Dealer', 'P': 'Purchase from Customer', 'S': 'Sale to Customer'}    # used to display a human-readable trade type in the output csv for batch pricing; NOTE: this is WET and needs to match those on the front end, perform cmd+f on `const tradeType = ` in `pricing.jsx` and make sure the values here match those

# Upper and lower bounds for what the user may enter for the quantity; NOTE: this is WET and needs to match those on the front end, perform cmd+f on `required min=` in both `pricing.jsx` and `tabsCusipSearchForm.jsx`
QUANTITY_LOWER_BOUND = 5    # in thousands
QUANTITY_UPPER_BOUND = 10000    # in thousands

REDEMPTION_VALUE_AT_MATURITY = 100    # hard coded for price at maturity
# FAILURE_EMAIL_RECIPIENTS = ['eng@ficc.ai', 'eng@ficc.ai', 'eng@ficc.ai']
FAILURE_EMAIL_RECIPIENTS = ['eng@ficc.ai']
# trade_data = None    # TODO: we create this variable with the goal was of not having to reload the data if the same CUSIP was called; currently not used, but maybe worth bringing back?

def send_email(subject, message, recipients=FAILURE_EMAIL_RECIPIENTS):
    sender_email = 'notifications@ficc.ai'
    password = 'ztwbwrzdqsucetbg'
    smtp_server = 'smtp.gmail.com'
    port = 587
    
    server = smtplib.SMTP(smtp_server,port)
    server.starttls()
    server.login(sender_email, password)
    
    message = MIMEText(message)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = ', '.join(recipients)
 
    try:
        server.sendmail(sender_email, recipients, message.as_string())
    except Exception as e:
        print(e)
    server.quit()


def datetime_as_string(datetime, separator=' ', precision='sec', display=False):
    '''Return the datetime as a string for a specified `precision` and `separator`.'''
    if precision == 'sec':
        time = HOUR_MIN_SEC
    elif precision == 'min':
        time = HOUR_MIN
    elif precision == 'day':
        time = ''
        separator = ''    # no need for a separator between the date and time when there is no time for the precision of `day`
    else:
        raise ValueError(f'Precision: {precision} not supported')
    year_month_day = MONTH_DAY_YEAR if display else YEAR_MONTH_DAY
    return datetime.strftime(year_month_day + separator + time)

get_current_datetime = lambda: datetime.now(EASTERN)

def current_datetime_as_string(separator=' ', precision='sec'):
    return datetime_as_string(get_current_datetime(), separator, precision)

# These messages are used when a CUSIP cannot be priced
CUSIP_ERROR_MESSAGE = {'invalid': 'CUSIP is invalid', 
                       'not_found': 'CUSIP not supported', 
                       'not_outstanding': 'CUSIP is no longer outstanding', 
                       'defaulted': 'CUSIP has defaulted', 
                       'maturing_before_settlement_date': 'CUSIP is maturing very soon or has already matured', 
                       'not_bonds': 'CUSIP is not supported because we do not support Anticipation Notes, Certificates of Obligation, Warrants, or Commercial Paper', 
                       'insufficient_data': 'One or more of the following dates necessary to compute yield has not been reported for this CUSIP: dated date, interest payment/coupon date, maturity date, coupon (interest) rate'}

bq_client = bigquery.Client()

def log_usage(usage_dict=None,    # if this value is not None, then it will overwrite all other arguments passed in 
              user=None, 
              api_call=False, 
              time=None, 
              cusip=None, 
              direction=None, 
              quantity=None, 
              ficc_price=None, 
              ficc_ytw=None, 
              yield_spread=None, 
              ficc_ycl=None, 
              calc_date=None, 
              daily_schoonover_report=False, 
              real_time_yield_curve=False, 
              batch=False, 
              show_similar_bonds=False, 
              error=False, 
              recent=None):    # TODO: make sure logging occurs in an asynchronic way since logging needs to occur always; `load_table_from_json` may not be asynchronic
    '''Logs usage whenever demo is used.'''
    if usage_dict is None and user is None:
        print('Logging failed. `usage_dict` and `user` can not both be `None`.')
        return None

    if time is None: time = current_datetime_as_string()

    try:
        if usage_dict is None:
            if type(recent) == list and len(recent) <= 5:
                recent = [{feature: round_for_logging(recent[idx]) if idx < len(recent) else None for idx, feature in enumerate(RECENT_FEATURES)}]
            else:
                print(f'`recent` ({recent}) is not a list of length less than or equal to 5 and so it will not be logged')
                recent = [{feature: None for feature in RECENT_FEATURES}]

            usage_dict = {'user': user, 
                          'api_call': api_call, 
                          'time': time, 
                          'cusip': cusip, 
                          'direction': direction, 
                          'quantity': quantity, 
                          'ficc_price': ficc_price, 
                          'ficc_ytw': ficc_ytw, 
                          'yield_spread': yield_spread, 
                          'ficc_ycl': ficc_ycl, 
                          'calc_date': calc_date, 
                          'daily_schoonover_report': daily_schoonover_report, 
                          'real_time_yield_curve': real_time_yield_curve, 
                          'batch': batch, 
                          'show_similar_bonds': show_similar_bonds, 
                          'error': error, 
                          'recent': recent}
            print(usage_dict)

        table_id = 'eng-reactor-287421.api_calls_tracker.usage_data'

        # create schema
        logging_schema = [bigquery.SchemaField(feature, dtype) for feature, dtype in LOGGING_FEATURES.items()]
        recent_schema = bigquery.SchemaField('recent', 'RECORD', mode='REPEATED', fields=[bigquery.SchemaField(feature, 'numeric') for feature in RECENT_FEATURES])
        logging_schema.append(recent_schema)
        job_config = bigquery.LoadJobConfig(schema=logging_schema)
        
        if type(usage_dict) is dict:    # if you're a customer and you just priced an individual bond, `usage_dict` will be the particular the fields as a dict, but `load_table_from_json` requires a list of dicts. For batch pricing, we call `log_usage` with a list of dicts so no conversion is necesssary
            usage_dict = [usage_dict]

        # assigning the result of this command and calling `load_job.result()` waits for the job to complete before proceeding
        bq_client.load_table_from_json(usage_dict, table_id, job_config=job_config)
        # load_job = bq_client.load_table_from_json(usage_dict, table_id, job_config=job_config)
        # try:
        #     load_job.result()    # Waits for the job to complete.
        # except Exception as e:
        #     print(load_job.errors)
        #     raise e
    except Exception as e:
        print(f'\nLogging error. {type(e)}: {e}\n')    # print the exception so to not terminate the program in case there is a logging error


cusip_is_invalid = lambda cusip: len(cusip) < 8 or not cusip.isalnum()    # can handle weird Excel scientific notation CUSIPs, so remove invalid condition of `len(cusip) > 9`; see `fix_cusip_improperly_formatted_from_excel_automatic_scientific_notation(...)`

def fix_cusip_improperly_formatted_from_excel_automatic_scientific_notation(cusip):
    '''There are some CUSIPs that end in “E#” where # is a digit. This causes Excel to represent 
    the CUSIP in scientific notation where “E#” represents 10^#. For example, 1073356E6 becomes 
    1073356000000. We can recover the original CUSIP, when the CUSIP has greater than 9 characters 
    (where all the trailing characters are 0s). Similarly, we can also recover the original CUSIP 
    when it has 7 characters, where the CUSIP should have ended in “E0”. In contrast, when we have 
    8 digits or 9 digits, we should not perform any additional modification beyond adding the check 
    digit for the 8 digit CUSIP since we cannot distinguish between an 8 digit CUSIP entered by the 
    user and one where Excel has mutated it based on scientific notation.'''
    if len(cusip) <= 9: return cusip    # no modification can be done to the CUSIP
    for idx, char in enumerate(cusip):
        if not ((idx >= 7 and char == '0') or (idx < 7 and char.isdigit())):    # if idx is greater than 7, then the character must be 0, otherwise the character must be a digit
            return cusip    # no modification can be done to the CUSIP since it has alphanumeric characters or does not have trailing 0's
    num_zeros = len(cusip) - 7
    corrected_cusip = cusip[:7] + f'E{num_zeros}'
    print(f'*** Excel altered CUSIP of {cusip} was converted to: {corrected_cusip} ***')
    return corrected_cusip

def get_data_from_redis(cusips):
    '''Return the data found in the redis from a list of cusips. If `return_cusips_not_found` 
    is True, then we return the data not found in the redis.
    NOTE: experiments with parallelization for getting the data from redis did not give any speedup.'''
    redis_client = redis.Redis(host='10.14.140.37', port=6379, db=0)
    if type(cusips) != list: cusips = [cusips]    # this means that a single cusip was passed in, but not in a list

    cusips_can_be_priced_df = []
    cusips_cannot_be_priced_df = []
    def get_data_for_single_cusip(cusip_idx, cusip):
        '''Get redis data for a single CUSIP. Put the data into the correct list
        based on if the data was found or not.'''
        if len(cusip) == 0: return None    # ignore an empty line

        def missing_important_dates(single_cusip_data):
            '''Checks whether important dates needed for pricing are null. To price a CUSIP, we need 
            `next_coupon_payment_date`, `maturity_date`, `first_coupon_date` and `accrual_date`. However, 
            we only need these features if the CUSIP is neither called nor a zero coupon bond.'''
            is_called = single_cusip_data['is_called'] == True
            is_zero_coupon = single_cusip_data['coupon_type'] == 17 or \
                             single_cusip_data['interest_payment_frequency'] == 16 or \
                             single_cusip_data['coupon'] == 0
            is_missing_important_dates = pd.isna(single_cusip_data['next_coupon_payment_date']) or \
                                         pd.isna(single_cusip_data['maturity_date']) or \
                                         pd.isna(single_cusip_data['first_coupon_date']) or \
                                         pd.isna(single_cusip_data['accrual_date'])
            return not is_called and not is_zero_coupon and is_missing_important_dates
                
        get_cusip_cannot_be_priced_series = lambda cusip_idx, cusip, message: pd.Series({'cusip': cusip, 'message': message}, name=cusip_idx)

        if cusip_is_invalid(cusip):    # all cusips must have length >= 8
            cusips_cannot_be_priced_df.append(get_cusip_cannot_be_priced_series(cusip_idx, cusip, 'invalid'))
        else:
            if len(cusip) == 8:
                check_digit = calculate_cusip_check_digit(cusip)
                orig_cusip = cusip    # `orig_cusip` used for print statement
                cusip = cusip + str(check_digit)
                print(f'*** 8 digit CUSIP of {orig_cusip} was converted to 9 digit CUSIP: {cusip} ***')
            try:
                cusip = fix_cusip_improperly_formatted_from_excel_automatic_scientific_notation(cusip)
                data = pickle.loads(redis_client.get(cusip))
                if data['outstanding_indicator'] is False:
                    cusips_cannot_be_priced_df.append(get_cusip_cannot_be_priced_series(cusip_idx, cusip, 'not_outstanding'))
                elif missing_important_dates(data) or pd.isna(data['coupon']):
                    cusips_cannot_be_priced_df.append(get_cusip_cannot_be_priced_series(cusip_idx, cusip, 'insufficient_data'))
                # elif data['default_exists'] is True:
                #     cusips_cannot_be_priced_df.append(get_cusip_cannot_be_priced_series(cusip_idx, cusip, 'defaulted'))
                # elif data['maturity_description_code'] != 2:
                #     cusips_cannot_be_priced_df.append(get_cusip_cannot_be_priced_series(cusip_idx, cusip, 'not_bonds'))
                else:    # no problems with this cusip
                    cusips_can_be_priced_df.append(data.rename(cusip_idx))    # change name to be `cusip_idx` in order to preserve original ordering
            except Exception as e:    # this means that the cusip was not found in the redis
                print(e) # #EDITED FROM ORIGINAL FINANCE.PY
                # print(f'cusip not in ref data Redis: {cusip}')
                cusips_cannot_be_priced_df.append(get_cusip_cannot_be_priced_series(cusip_idx, cusip, 'not_found'))

    for cusip_idx, cusip in enumerate(cusips):
        get_data_for_single_cusip(cusip_idx, cusip)
    cusips_can_be_priced_df = pd.concat(cusips_can_be_priced_df, axis=1).T if cusips_can_be_priced_df != [] else pd.DataFrame()    # list of series to dataframe: https://stackoverflow.com/questions/55478191/list-of-series-to-dataframe
    cusips_cannot_be_priced_df = pd.concat(cusips_cannot_be_priced_df, axis=1).T if cusips_cannot_be_priced_df != [] else pd.DataFrame()     # list of series to dataframe: https://stackoverflow.com/questions/55478191/list-of-series-to-dataframe
    return cusips_can_be_priced_df, cusips_cannot_be_priced_df


def process_ref_data(df, quantity, trade_type, current_date, settlement_date, trade_datetime, is_batch_pricing=False):
    df['par_traded'] = quantity
    df['trade_type'] = trade_type
    df['trade_date'] = current_date
    df['settlement_date'] = settlement_date
    df['trade_datetime'] = trade_datetime
    df['transaction_type'] = 'I'
    try:
        df = process_data(df,
                          trade_datetime,
                          bq_client,
                          SEQUENCE_LENGTH,
                          NUM_FEATURES,
                          'FICC',
                          remove_short_maturity=True,
                          min_trades_in_history=0,
                          process_ratings=False, 
                          treasury_rate_df=get_treasury_rate_df(), 
                          is_batch_pricing=is_batch_pricing)
    except Exception as e:
        print(f'process_data failed with error: {e}')
        raise e
    return df


def get_ref_data(cusip, quantity, current_date, trade_datetime, settlement_date, trade_type):
    '''This function retrives reference data from our managed Redis instance. Currently,
    the key for this Redis instance is simply a CUSIP, and the reference data retrieved 
    is the most recent reference data for that CUSIP. In future, this will be CUSIP and
    timestamp.'''
    cusip_can_be_priced_df, cusip_cannot_be_priced_df = get_data_from_redis(cusip)
    if len(cusip_cannot_be_priced_df) != 0:
        error_key = cusip_cannot_be_priced_df['message'].values[0]
        raise CustomMessageError(CUSIP_ERROR_MESSAGE[error_key])
    
    maturity_date = cusip_can_be_priced_df['maturity_date'].values[0]
    if compare_dates(maturity_date, settlement_date) <= 0:    # checks if maturity_date is before or the same as settlement_date
        # raise CustomMessageError(f'This cusip matures on {datetime_as_string(maturity_date, precision="day")}')
        raise CustomMessageError(CUSIP_ERROR_MESSAGE['maturing_before_settlement_date'])    # the above error message provides more detail but this keeps it consistent with that in batch pricing
    
    cusip_can_be_priced_df = process_ref_data(cusip_can_be_priced_df, quantity, trade_type, current_date, settlement_date, trade_datetime)
    cusip_can_be_priced_df['orig_interest_payment_frequency'] = cusip_can_be_priced_df.interest_payment_frequency
    return cusip_can_be_priced_df


def target_trade_processing_for_attention(row):
    target_trade_features = []
    target_trade_features.append(row['quantity'])
    target_trade_features = target_trade_features + TRADE_MAPPING[row['trade_type']]
    return np.tile(target_trade_features, (1, 1))


get_trade_price = lambda trade: compute_price(trade, trade.ficc_ytw)    # compute price does not need to return the calc_date, if we are using the calc_date model


def get_access_token(audience):
    '''This function retrieves a Firebase authentication token.'''
    token_response = requests.get(
        'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=' + audience,
        headers={'Metadata-Flavor': 'Google'})     
    return token_response.content.decode('utf-8')


def get_settlement_date(trade_date):
    '''Maturity should be calculated based on a given settlement date, rather than a trade date.
    This function calculates a settlement date given a trade date in order to calculate maturity.
    NB: this needs to be converted to datetime in ET.'''
    trade_date = pd.to_datetime(trade_date, format=YEAR_MONTH_DAY)
    settlement_date = trade_date + TWO_DAYS
    while settlement_date.weekday() in holidays.WEEKEND or settlement_date in HOLIDAYS_US:
        settlement_date += ONE_DAY
    return settlement_date


def pre_processing(df):
    '''This function performs pre-processing for trade and reference data before it is sent 
    to the model.'''
    if type(df) == str: return df    # NB: When df is a str, it is an error message, but these error messages don't correspond to those in process_data

    df['target_attention_features'] = df.apply(target_trade_processing_for_attention, axis=1)

    if 'target_attention_features' not in PREDICTORS:
        PREDICTORS.append('target_attention_features')
    if 'ficc_treasury_spread' not in PREDICTORS:
        PREDICTORS.append('ficc_treasury_spread')
        NON_CAT_FEATURES.append('ficc_treasury_spread')
    # if 'ted-rate' not in PREDICTORS:
    #     PREDICTORS.append('ted-rate')
    #     NON_CAT_FEATURES.append('ted-rate')
    return df


def create_input(df):
    encoders = get_encoders()    # do not make `encoders` a global variable because the only way to use the updated encoders is to re-deploy the server, and we would like to use updated encoders even if there is no server code change
    datalist = []

    noncat_and_binary = []
    for f in NON_CAT_FEATURES + BINARY:
        noncat_and_binary.append(np.expand_dims(df[f].to_numpy().astype('float64'), axis=1))
    datalist.append(np.concatenate(noncat_and_binary, axis=-1))
    
    for f in CATEGORICAL_FEATURES:
        encoded = encoders[f].transform(df[f])
        datalist.append(encoded.astype('float64'))

    return datalist


def predict_spread(instances):
    '''This function retrieves yield spread estimates from the model given a set of instances.'''
    api_endpoint = 'us-east4-aiplatform.googleapis.com'
    project = '964018767272'
    endpoint_id = '5835310518746742784'
    location = 'us-east4'
    client_options = {'api_endpoint': api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    parameters = {}
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    predictions=response.predictions
    return predictions


def get_inputs_for_nn(df, return_values_only=False):
    '''Returns inputs for the neural network as a dictionary.'''
    inputs = create_input(df)
    inputs = [list(a) for a in list(inputs)]

    trade_history_input = np.stack(df['trade_history'].to_numpy())
    trade_history_input = trade_history_input.tolist()

    target_attention_features_input = np.stack(df['target_attention_features'].to_numpy())
    target_attention_features_input = target_attention_features_input.tolist()

    inputs = trade_history_input + target_attention_features_input + inputs

    if return_values_only: return inputs

    input_dict = {}
    for idx, nn_pred in enumerate(NN_PRED_LIST):
        input_dict[nn_pred] = inputs[idx][0].tolist() if nn_pred == 'NON_CAT_AND_BINARY_FEATURES' else inputs[idx]
    
    return input_dict


def _get_spread(df):
    '''This function takes a dataframe, encodes the features, and returns yield spread estimates.
    NB: This is only used by get_price_and_ytw. In order to be DRY, get_BB11_table and get_ytw_curve
    should also use this function.'''
    input_dict = get_inputs_for_nn(df)
    spread = predict_spread([input_dict])    # functions in `predict_spread` expect a list of dictionaries, so `input_dict` must be wrapped with list brackets
    # print(spread)    # FIXME: it may be a bug that we return the (0, 0) item of spread since this may fail for batch pricing in which we do not want just the 0 indexed item
    return spread[0][0]    # TODO: why do we return the (0, 0) index?


def get_price_and_ytw(cusip, quantity, trade_type, user, api_call):
    '''This function takes aspects of a hypothetical trade for a particular CUSIP
    and returns estimations of price and Yield to Worst.'''
    try:    # wrap in try except to perform logging even when there is an error
        current_datetime = get_current_datetime()
        current_date = datetime_as_string(current_datetime, precision='day')
        current_datetime = datetime_as_string(current_datetime)    # current_datetime is now a string; making it in precision minutes since this is what get_yield_curve requires

        settlement_date = get_settlement_date(current_date)
        quantity = int(quantity) * 1000    # wrap this in int since `quantity` is passed in as a string, but for this function, we need `quantity` to be a numerical value

        # FIXME: remove calling `process_data` inside `get_ref_data(...)` and instead call `get_ytw_for_list(...)` in side this function
        # Passing the date for the trade
        df = get_ref_data(cusip, 
                          quantity, 
                          pd.to_datetime(current_date, format=YEAR_MONTH_DAY),
                          pd.to_datetime(current_datetime, format=YEAR_MONTH_DAY_HOUR_MIN_SEC),
                          settlement_date,
                          trade_type)

        # Changed yield_curve_level to ficc_ycl. This now comes from the data package
        yield_curve_level = df['ficc_ycl'].iloc[0] / 100    # extract the yield_curve_level from the dataframe, and since the dataframe only has one item, so we are isolating the value by doing .iloc[0]

        df = pre_processing(df)
        if type(df) == str: return df
        ys = _get_spread(df) / 100
        if ys == 'error': raise ValueError(f'Cannot compute the spread since certain columns are null:\n{df}')

        ytw = yield_curve_level + ys
        df['ficc_ytw'] = ytw
        df['interest_payment_frequency'] = df.orig_interest_payment_frequency
        
        price, calc_date = get_trade_price(df.iloc[0])    # the dataframe only has one item, so we are isolating the value by doing .iloc[0]
        price = np.round(price, DISPLAY_PRECISION)
        df['price'] = price

        print('\n*****************')
        print(f'ytw: {ytw}')
        print(f'ys: {ys}')
        print(f'price: {price}')
        print('*****************\n')

        df['calc_date'] = datetime.strftime(calc_date, MONTH_DAY_YEAR)    # calc_date is now changed to the predicted calc_date instead of last_calc_date which is what it was assigned in `_add_calc_date_and_ficc_treasury_spread(...)`; this format is MONTH_DAY_YEAR, instead of YEAR_MONTH_DAY for presentation purposes
        maturity_date = datetime.strptime(np.datetime_as_string(df.maturity_date.values[0], unit='s'), YEAR_MONTH_DAY + 'T' + HOUR_MIN_SEC)
        df['maturity_date'] = datetime.strftime(maturity_date, YEAR_MONTH_DAY)    # use this maturity date for similar bonds; change to MONTH_DAY_YEAR on the front end since the backend needs it in format YEAR_MONTH_DAY to find similar bonds

        try:
            next_call_date = datetime.strptime(np.datetime_as_string(df.next_call_date.values[0], unit='s'), YEAR_MONTH_DAY + 'T' + HOUR_MIN_SEC)
            df['next_call_date'] = datetime.strftime(next_call_date, YEAR_MONTH_DAY)    # use this to display the call date on the front end
        except ValueError as e:
            next_call_date = None
            df['next_call_date'] = None

        try:
            refund_date = datetime.strptime(np.datetime_as_string(df.refund_date.values[0], unit='s'), YEAR_MONTH_DAY + 'T' + HOUR_MIN_SEC)
            df['refund_date'] = datetime.strftime(refund_date, YEAR_MONTH_DAY)    # use this to display the call date on the front end
        except ValueError as e:
            refund_date = None
            df['refund_date'] = None

        display_text_for_ytw = 'Worst'    # display the yield to _ on the front end
        display_price = REDEMPTION_VALUE_AT_MATURITY    # display the redemption value for the associated redemption date on the front end
        if calc_date == maturity_date:
            display_text_for_ytw = 'Maturity'
        elif calc_date == refund_date:    # give priority to refund instead of call because there are bonds that are called at CAV and called at premium where the call date and fefund date and prices are the same, but the bond is called so we should show refund date and price
            display_text_for_ytw = 'Refund'
            display_price = df.refund_price
        elif calc_date == next_call_date:
            display_text_for_ytw = 'Call'
            display_price = df.next_call_price

        try:    # wrap this in a try...except statement so that an invalid `df.next_call_price` or `df.refund_price` are not sent to the front end
            display_price = np.round(float(display_price), DISPLAY_PRECISION)
            if display_price % 1 == 0: display_price = int(display_price)    # shave off the .0 if it exists
        except ValueError as e:
            display_price = REDEMPTION_VALUE_AT_MATURITY

        issue_date = datetime.strptime(np.datetime_as_string(df.issue_date.values[0], unit='s'), YEAR_MONTH_DAY + 'T' + HOUR_MIN_SEC)
        df['issue_date'] = datetime.strftime(issue_date, YEAR_MONTH_DAY)    # use this to display the dated date on the front end

        # extract the previous trade features which is a pandas series of length 1
        previous_trade_features_array = df.previous_trades_features.iloc[0]

        # map of each feature in `previous_trade_features_array` to position in array
        feature_to_idx = {'yield_spread': 0, 
                          'ficc_ycl': 1, 
                          # 'rtrs_control_number': 2, 
                          'yield_to_worst': 3, 
                          'dollar_price': 4, 
                          # 'seconds_ago': 5, 
                          'size': 6, 
                          'calc_date': 7, 
                          # 'maturity_date': 8, 
                          # 'next_call_date': 9, 
                          # 'par_call_date': 10, 
                          # 'refund_date': 11, 
                          'trade_datetime': 12, 
                          # 'calc_day_cat': 13, 
                          # 'settlement_date': 14, 
                          'trade_type': 15}

        # since now we process trades which do not have a history
        if len(previous_trade_features_array) > 0:
            # convert all trade_datetime features to a easy to read string for display on the front end
            previous_trade_features_array[:, feature_to_idx['trade_datetime']] = np.vectorize(lambda datetime: datetime_as_string(datetime, display=True))(previous_trade_features_array[:, feature_to_idx['trade_datetime']])    # np.vectorize allows us to apply this function to each item in the array

            num_trades, _ = previous_trade_features_array.shape    # second item in the shape tuple is the number of features per trade, which should be equal to len(feature_to_idx)
            num_trades = min(num_trades, 32)    # 32 is the maximum number of trades in the same CUSIP trade history; change this value to display fewer than 32 trades on the front end if desired

            # put previous trade features into a list of dictionaries
            def create_previous_trade_dict(trade_idx):
                previous_trade = {}
                for feature, feature_idx in feature_to_idx.items():
                    feature_value = previous_trade_features_array[trade_idx, feature_idx]
                    if feature == 'calc_date': feature_value = datetime_as_string(feature_value, precision='day', display=True)
                    previous_trade[feature] = feature_value
                return previous_trade
            
            previous_trades = [create_previous_trade_dict(trade_idx) for trade_idx in range(num_trades)]
        else:
            previous_trades = []

        df['previous_trades_features'] = [previous_trades]    # wrapping this in a list since the original df expects a single value for each column
        # create these next two variables as sets so we can easily perform an intersection
        features_to_display = {'security_description', 'price', 'ficc_ytw', 'calc_date', 'coupon', 'issue_date', 'next_call_date', 'previous_trades_features'}
        features_for_similar_bonds = {'purpose_class', 'purpose_sub_class', 'coupon', 'incorporated_state_code', 'rating', 'maturity_date', 'moodys_long'}

        df['rating'] = df['rating'].replace('MR', 'NR')    # change rating from MR to NR since MR means no rating from any agency, and NR means no rating from S&P, but for the user, this should be the same
        df['moodys_long'] = df['moodys_long'].fillna('NR')    # change rating from null to NR since null means no rating from any agency, and NR means no rating from Moody's, but for the user, this should be the same

        df['ficc_ytw'] = np.round(ytw, DISPLAY_PRECISION)    # this is to display the predicted yield to worst with the correct number of decimal points on the front end

        df_dict_list = df[list(features_to_display | features_for_similar_bonds)].to_dict('records')    # returns a dictionary inside a list; since there is only one row in `df`, there is only one item in this list (the dictionary maps the feature to the feature value for each row)
        df_dict = df_dict_list[0]    # extract the dictionary from the list by choosing the 0-th index
        features_to_remove_where_feature_value_is_null = [feature for feature, feature_value in df_dict.items() if type(feature_value) != list and pd.isnull(feature_value)]    # first check whether the feature value is a list which is the case with previous_trade_features, since we cannot check if a list is null type
        for feature in features_to_remove_where_feature_value_is_null:    # need to perform this procedure since purpose_sub_class sometimes has a null feature_value
            df_dict.pop(feature)    # remove all features from the dictionary where the feature value is null

        df_dict['display_text_for_ytw'] = display_text_for_ytw
        df_dict['display_price'] = display_price
        response = make_response(jsonify(df_dict_list), 200)    # need to listify the resulting set after the intersection when selecting columns in `df` due to `FutureWarning: Passing a set as an indexer is deprecated and will raise in a future version. Use a list instead.`

        log_usage(user=user, 
                  api_call=api_call, 
                  cusip=cusip, 
                  direction=trade_type, 
                  quantity=quantity // 1000, 
                  ficc_price=price, 
                  ficc_ytw=round_for_logging(ytw),    # need to be rounded to LOGGING_PRECISION (can perhaps do more but not too many more digits) decimal places otherwise will be detected as an invalid numerical value
                  yield_spread=round_for_logging(ys),    # need to be rounded to LOGGING_PRECISION (can perhaps do more but not too many more digits) decimal places otherwise will be detected as an invalid numerical value
                  ficc_ycl=round_for_logging(yield_curve_level),    # need to be rounded to LOGGING_PRECISION (can perhaps do more but not too many more digits) decimal places otherwise will be detected as an invalid numerical value
                  calc_date=datetime.strftime(calc_date, MONTH_DAY_YEAR), 
                  recent=[np.float64(trade[0]) for trade in df['trade_history'].iloc[0]])    # `.iloc[0]` selects the first (and only) row in the dataframe since we only priced a single CUSIP, and trade[0] chooses the first item (corresponding to the yield spread) in each trade in the trade history; need to convert to float otherwise the following exception is raised: `TypeError: Object of type int64 is not JSON serializable`

        return response
    except Exception as e:
        log_usage(user=user, 
                  api_call=api_call, 
                  cusip=cusip, 
                  direction=trade_type, 
                  quantity=quantity // 1000, 
                  error=True)
        if type(e) == CustomMessageError: return e.get_json_response()
        else: return CustomMessageError(DEFAULT_PRICING_ERROR_MESSAGE).get_json_response()



def _create_df_list(df):
    # TODO: what is going on here logically, in this for loop? Basically doing a transpose. Create a description for this function.
    # FIXME: add assert statements as to what features we expect in `df`. Add assert statement and see what happens. Should we use assert statements in this code? I'm worried that it will cause problems in production. What is the best way to do error handling in this way?
    nn_inputs = get_inputs_for_nn(df, return_values_only=True)
    df_list = []
    number_of_cusips = len(nn_inputs[-1])
    nn_pred_list_to_idx = {nn_pred: idx for idx, nn_pred in enumerate(NN_PRED_LIST + ['last_calc_day_cat'])}    # maps the nn predictor to its index in the NN_PRED_LIST which will can be used to position the nn predictor in `input_dict`
    for cusip_idx in range(number_of_cusips):
        input_dict = {}             
        for nn_pred in NN_PRED_LIST:
            if nn_pred == 'trade_history_input':
                input_dict[nn_pred] = nn_inputs[cusip_idx]
            elif nn_pred == 'target_attention_input':
                input_dict[nn_pred] = nn_inputs[cusip_idx + number_of_cusips]
            else:    # TODO: when using this code, we are not using `last_calc_day_cat` which is not in the CATEGORICAL_FEATURES, so figure out: should add in `last_calc_day_cat` when constructing `nn_pred_list_to_idx` or should we add this to CATEGORICAL_FEATURES?
                nn_pred_idx = nn_pred_list_to_idx[nn_pred]
                input_dict[nn_pred] = nn_inputs[2 * number_of_cusips + nn_pred_idx - 2][cusip_idx].tolist()    # the `- 2` comes from the first two items being 'trade_history_input' and 'target_attention_input' so we are making an index adjustment
        
        df_list.append(input_dict)
    return df_list


def get_ytw_for_list(df, trade_datetime, quantity_list, trade_type, current_date, settlement_date):
    '''Return a list of ytw values for a dataframe on `df`. The `reference_datetime` 
    is used to get the yield curve level.'''
    df = process_ref_data(df, quantity_list, trade_type, current_date, settlement_date, trade_datetime, True)
    print(df.columns)
    # Changing yield_curve_level to ficc_ycl. This now comes from the data package
    yc = np.array(df['ficc_ycl']) / 100

    df = pre_processing(df)
    if type(df) == str: return df
    df_list = _create_df_list(df)

    ys = predict_spread(df_list)
    ys = np.array(ys) / 100
    ys = ys.ravel()    # `np.ravel` returns a contiguous flattened array
    ytw = np.add(ys, yc)

    # print('\n*****************')
    # print(f'ytw: {ytw}')
    # print(f'ys: {ys}')
    # print('*****************\n')

    return ytw, ys, yc, df    # `ys` and `yc` are used for logging


def add_ytw_price_calculationdate_coupon(df, ytw):
    '''Add the features: (1) 'ficc_ytw', (2) 'price', (3) 'yield_to_worst_date', 
    (4) 'ytw', and (5) 'coupon', and return the new dataframe.
    TODO: refactor this so that it can be used in `get_price_and_ytw`.'''
    df['ficc_ytw'] = ytw
    df['interest_payment_frequency'] = df.orig_interest_payment_frequency    # TODO: standardize so that we only have one interest_payment_frequency throughout
    temp_df = df.apply(lambda trade: get_trade_price(trade), axis=1)
    temp_df = temp_df.to_frame()
    temp_df[['price', 'calc_date']] = pd.DataFrame(temp_df[0].tolist(), index=temp_df.index)
    df['price'] = np.round(temp_df.price, DISPLAY_PRECISION)
    df['yield_to_worst_date'] = temp_df['calc_date'].dt.strftime(MONTH_DAY_YEAR)
    df['ytw'] = np.round(df.ficc_ytw, DISPLAY_PRECISION)
    df['ytw_LOGGING_PRECISION'] = round(df.ficc_ytw, LOGGING_PRECISION)    # used for logging
    df['coupon'] = np.round(df['coupon'].astype(float), DISPLAY_PRECISION)
    return df


def make_response_if_cusip_found(df, cusip):
    if len(df) == 0:
        return f'No results for {cusip}'
    response = make_response(jsonify(df.to_dict('records')), 200)
    return response


def price_cusips_list(cusip_list, quantity_list, trade_type):
    '''This function takes a list of CUSIPs and returns price and YTW estimates for each.'''
    start = time.time()
    cusips_can_be_priced_df, cusips_cannot_be_priced_df = get_data_from_redis(cusip_list)
    # print(f'Loading ref data took {time.time()-start} seconds')
    if len(cusips_cannot_be_priced_df) != 0: 
        print(f'Following cusips cannot be priced: {cusips_cannot_be_priced_df.cusip.to_list()}')
        send_email(subject = 'MSRB Real Time Trades in Unpriceable CUSIPS', message = 'Following cusips cannot be priced: ' + ', '.join(cusips_cannot_be_priced_df.cusip.to_list())) #EDITED FROM ORIGINAL FINANCE.PY
    quantity_list = np.array(quantity_list)    # converted to numpy list in order to easily index by list

    cusip_indices_that_can_be_priced = list(cusips_can_be_priced_df.index.values)
    quantities_for_cusips_that_can_be_priced = quantity_list[cusip_indices_that_can_be_priced]
    cusips_can_be_priced_df['quantity'] = quantities_for_cusips_that_can_be_priced
    cusips_can_be_priced_df['non_log_transformed_quantity'] = quantities_for_cusips_that_can_be_priced    # this is used for later restoring the non-log10 transformed quantities and quantities for CUSIPs not priced to the dataframe
    
    cusip_indices_that_cannot_be_priced = list(cusips_cannot_be_priced_df.index.values)
    quantities_for_cusips_that_cannot_be_priced = quantity_list[cusip_indices_that_cannot_be_priced]
    cusips_cannot_be_priced_df['quantity'] = quantities_for_cusips_that_cannot_be_priced

    def _fill_basic_error_columns(df):
        df['ytw'] = NUMERICAL_ERROR
        df['ytw_LOGGING_PRECISION'] = NUMERICAL_ERROR
        df['price'] = NUMERICAL_ERROR
        df['yield_spread'] = NUMERICAL_ERROR
        df['ficc_ycl'] = NUMERICAL_ERROR
        df['coupon'] = pd.NA
        df['security_description'] = pd.NA
        df['maturity_date'] = pd.NA
        return df

    def fill_error_columns(df, message_key):
        if len(df) == 0: return df
        message = CUSIP_ERROR_MESSAGE[message_key]
        df['yield_to_worst_date'] = message
        return _fill_basic_error_columns(df)
    
    def fill_all_error_columns(df):
        if len(df) == 0: return df
        grouped_by_message = df.groupby('message')
        df['yield_to_worst_date'] = grouped_by_message.message.transform(lambda x: CUSIP_ERROR_MESSAGE[x.name])    # assign a value to each group: https://stackoverflow.com/questions/69951813/groupby-specific-column-then-assign-new-values-base-on-conditions
        return _fill_basic_error_columns(df)

    if len(cusips_can_be_priced_df) != 0:
        current_datetime = get_current_datetime()
        current_date = datetime_as_string(current_datetime, precision='day')
        current_datetime = datetime_as_string(current_datetime)    # current_datetime is now a string; making it in precision minutes since this is what get_yield_curve requires
        settlement_date = get_settlement_date(current_date)

        settlement_date_after_maturity_date = cusips_can_be_priced_df['maturity_date'] <= settlement_date
        cusips_can_be_priced_df_settlement_date_after_maturity_date = cusips_can_be_priced_df[settlement_date_after_maturity_date]
        cusips_can_be_priced_df_settlement_date_after_maturity_date = fill_error_columns(cusips_can_be_priced_df_settlement_date_after_maturity_date, 'maturing_before_settlement_date')
        cusips_can_be_priced_df_settlement_date_before_maturity_date = cusips_can_be_priced_df[~settlement_date_after_maturity_date]
        
        if len(cusips_can_be_priced_df_settlement_date_before_maturity_date) > 0:    # only attempt to price cusips if there are any remaining after removing those where the settlement date is after the maturity date
            cusips_can_be_priced_df_settlement_date_before_maturity_date['orig_interest_payment_frequency'] = cusips_can_be_priced_df_settlement_date_before_maturity_date.interest_payment_frequency    # TODO: why do we have both 'orig_interest_payment_frequency' and 'interest_payment_frequency' in `df`?
            ytw, spreads, ycl, cusips_can_be_priced_df_settlement_date_before_maturity_date = get_ytw_for_list(cusips_can_be_priced_df_settlement_date_before_maturity_date, 
                                                                                                               pd.to_datetime(current_datetime, format=YEAR_MONTH_DAY_HOUR_MIN_SEC), 
                                                                                                               cusips_can_be_priced_df_settlement_date_before_maturity_date['quantity'].values, 
                                                                                                               trade_type, 
                                                                                                               pd.to_datetime(current_date, format=YEAR_MONTH_DAY), 
                                                                                                               settlement_date)
            cusips_can_be_priced_df_settlement_date_before_maturity_date = add_ytw_price_calculationdate_coupon(cusips_can_be_priced_df_settlement_date_before_maturity_date, ytw)

            # below features are used for logging
            cusips_can_be_priced_df_settlement_date_before_maturity_date['yield_spread'] = round_for_logging(spreads)
            cusips_can_be_priced_df_settlement_date_before_maturity_date['ficc_ycl'] = round_for_logging(ycl)

            cusips_can_be_priced_df = pd.concat([cusips_can_be_priced_df_settlement_date_before_maturity_date, cusips_can_be_priced_df_settlement_date_after_maturity_date])
            cusips_can_be_priced_df['quantity'] = cusips_can_be_priced_df['non_log_transformed_quantity']    # put the non-log10 transformed quantity back into the dataframe
    cusips_cannot_be_priced_df = fill_all_error_columns(cusips_cannot_be_priced_df)
    cusips_df = pd.concat([cusips_can_be_priced_df, cusips_cannot_be_priced_df])
    return cusips_df.sort_index()


def get_batch_preds(file_obj_or_list_tuple, default_quantity, trade_type, user, api_call, access_token=None):
    import codecs
    import csv
    default_quantity = int(default_quantity)

    try:    # wrap in try except to perform logging even when there is an error
        def add_logging_fields(error=False):    # use `df` directly for logging since the `.to_dict('records')` function will format this nicely into a list of dicts, and when needing to assign a single value (e.g., setting 'direction' to 'D') to all rows, it is syntactically clean to assign it to a column of a dataframe
            # do not need to populate the quantity field since this is already done in `price_cusips_list`
            df['user'] = user
            df['api_call'] = api_call
            df['time'] = current_datetime_as_string()
            # df['direction'] = df.direction
            df['daily_schoonover_report'] = False
            df['real_time_yield_curve'] = False
            df['batch'] = True
            df['show_similar_bonds'] = False
            df['ficc_price'] = NUMERICAL_ERROR if error else df.price
            df['ficc_ytw'] = NUMERICAL_ERROR if error else df.ytw_LOGGING_PRECISION
            df['calc_date'] = 'None' if error else df.yield_to_worst_date
            if 'error' not in df.columns: df['error'] = error

        file_passed_in = True
        if type(file_obj_or_list_tuple) == tuple:
            cusip_list, quantity_list = file_obj_or_list_tuple
            assert type(cusip_list) == list, f'cusip_list shoud be of type list, but is instead {type(cusip_list)} and has value {cusip_list}'
            if quantity_list == None: quantity_list = [None] * len(cusip_list)
            file_passed_in = False
        else:
            reader = csv.reader(codecs.iterdecode(file_obj_or_list_tuple, 'utf-8'))
            cusip_list, quantity_list = [], []
            for row in reader:
                if len(row) > 0:    # first check if the row is empty before processing
                    cusip_list.append(row[0].upper())    # uppercase each cusip
                    quantity = row[1] if len(row) > 1 else None    # if statement is True if the user has inputted a quantity
                    quantity_list.append(quantity)
        assert len(cusip_list) == len(quantity_list)    # these lengths are assumed to be equal when batch pricing in chunks

        def filename_or_df(df):    # return filename if input was csv or return dataframe if user inputted lists
            if not file_passed_in: return df
            temporary_filename = '/tmp/results.csv'
            df.to_csv(temporary_filename, index=False)    # create `add_pricing_logging_fields_to_df()` function before saving `df` to a csv so that we have access to the function in the try except
            return temporary_filename

        if len(cusip_list) == 0: return filename_or_df(pd.DataFrame(columns=FEATURES_FOR_OUTPUT_CSV))    # create empty dataframe with just column headers

        LARGE_BATCH_SIZE = 10000
        is_large_batch = len(cusip_list) > LARGE_BATCH_SIZE    # used to control logging and initializations for making API call to function
        if is_large_batch == False: default_quantity *= 1000    # used to fill in the quantity of a hypothetical trade if there is no quantity provided in the csv
        
        def process_quantity(user_quantity):
            if user_quantity == None:    # no quantity was provided, so return `default_quantity` which has been modified if it is a large batch or not
                return default_quantity
            elif is_large_batch:    # quantity was provided by the user, but no need to modify it here, can do it in the API call
                return user_quantity
            try:    # quantity provided, and needs to be handled since there is no API call that is going to be made to handle it
                quantity = int(user_quantity) * 1000
                return max(min(quantity, QUANTITY_UPPER_BOUND * 1000), QUANTITY_LOWER_BOUND * 1000)    # if quantity is outside of the range [QUANTITY_LOWER_BOUND * 1000, QUANTITY_UPPER_BOUND * 1000], then put it back into the range
            except ValueError:    # catches the case where quantity value is not a valid integer
                return default_quantity    # `quantity` is initialized to `default_quantity` which has been modified if it is a large batch or not
        quantity_list = [process_quantity(quantity) for quantity in quantity_list]    # override `quantity_list`

        async def get_df_from_api_calls(cusip_quantity_chunks):
            async with aiohttp.ClientSession() as session:    # TODO: consider if there is a size limit since pricing 10k CUSIPs returns a CSV of size 1.2 megabytes
                tasks = [asyncio.ensure_future(api_call_func(session, cusip_chunk, quantity_chunk)) for cusip_chunk, quantity_chunk in cusip_quantity_chunks]
                df_list = await asyncio.gather(*tasks)
            return df_list

        async def api_call_func(session, cusip_chunk, quantity_chunk):
            url = 'https://server-3ukzrmokpq-uc.a.run.app/api/batchpricing'
            data = {'access_token': access_token, 
                    'amount': default_quantity, 
                    'tradeType': trade_type}
            data['cusipList'] = cusip_chunk
            data['quantityList'] = quantity_chunk
            async with session.post(url, data=data, timeout=1800) as resp:    # timeout=900 means that there is a 1800 second (30 min) threshold before raising `asyncio.TimeoutError`
                resp_json = await resp.json()
                return pd.read_json(resp_json)
        
        if is_large_batch:
            chunk_size = LARGE_BATCH_SIZE
        else:
            chunk_size = 1000
            if len(cusip_list) > chunk_size:
                num_chunks = 8    # number of CPUs on the gcloud app deploy
                chunk_size = math.ceil(len(cusip_list) / num_chunks)
        cusip_quantity_chunks = [(cusip_list[chunk_start_idx : chunk_start_idx + chunk_size], quantity_list[chunk_start_idx : chunk_start_idx + chunk_size]) for chunk_start_idx in range(0, len(cusip_list), chunk_size)]
        
        start_time = time.time()
        if is_large_batch:
            df_list = asyncio.run(get_df_from_api_calls(cusip_quantity_chunks))
        else:
            # df_list = [func(cusip_chunk, quantity_chunk) for cusip_chunk, quantity_chunk in cusip_quantity_chunks]
            with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
                df_list = pool_object.starmap(lambda cusip_chunk, quantity_chunk: price_cusips_list(cusip_chunk, quantity_chunk, trade_type), cusip_quantity_chunks)    # need to use starmap since lambda function has multiple arguments: https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
        print(f'Took {timedelta(seconds=time.time() - start_time)} to price {len(cusip_list)} CUSIPs')
        df = pd.concat(df_list).reset_index(drop=True)    # need .reset_index(...) to avoid `ValueError: DataFrame index must be unique for orient='columns'`
        
        # create `add_basic_logging_fields()` function before calling `price_cusips_list(...)` so that we have access to the function in the try except
        df['trade_type'] = trade_type    # if 'trade_type' not in df.columns else df['trade_type'].fillna(trade_type)    # fills in all trade_type's for CUSIPs that were not priced
        df['direction'] = df['trade_type']    # used for logging
        df['trade_type'] = df['trade_type'].map(TRADE_TYPE_CODE_TO_TEXT)    # replace trade_type code with human readable text for output csv
        df['quantity'] = df['quantity'].astype(int)
        df['quantity_output'] = df['quantity']

        # handle cusips that have errored by first marking all as not having errored, then isolating the ones that have errored and marking them as such
        df['error'] = False
        df.loc[df['price'] == NUMERICAL_ERROR, 'error'] = True    # could have chosen either the `price` field or the `ytw` field since both are filled with `NUMERICAL_ERROR` when there is a pricing error

        if is_large_batch is False:
            df['quantity'] = df['quantity'] // 1000    # when logging, quantity is in thousands
            add_logging_fields()
            log_usage(df[LOGGING_FEATURES.keys()].to_dict('records'))    # LOGGING_FEATURES.keys() gets each of the features needed for logging

        df['quantity'] = df['quantity_output']
        df = df[FEATURES_FOR_OUTPUT_CSV]
        df['maturity_date'] = pd.to_datetime(df['maturity_date']).dt.strftime(MONTH_DAY_YEAR)    # first convert the date object to a datetime object and then do a string format
        return filename_or_df(df)
    except Exception as e:
        if 'df' not in locals():
            df = pd.DataFrame()
            if 'cusip_list' in locals():    # `df` was not successfully created, i.e., `price_cusips_list(...)` did not work
                df['cusip'] = cusip_list
            else:    # file could not be read, so there are no cusips to log
                df['cusip'] = ['None']

        if 'quantity_list' in locals():
            df['quantity'] = quantity_list
        else:
            df['quantity'] = [default_quantity]
        df['quantity'] = df['quantity'] // 1000
        add_logging_fields(error=True)    # we know this function exists since `cusip_list` exists
        log_usage(df[LOGGING_FEATURES.keys()].to_dict('records'))
        raise e


