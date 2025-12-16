'''
 # @ Create date: 2023-12-18
 # @ Modified date: 2024-04-18
 '''
import warnings
import os
import shutil
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay
from sklearn import preprocessing
from pickle5 import pickle
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import traceback 

from google.cloud import bigquery
from google.cloud import storage

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ficc.utils.gcp_storage_functions import upload_data, download_data
from ficc.data.process_data import process_data
from ficc.utils.auxiliary_functions import function_timer, sqltodf, get_ys_trade_history_features, get_dp_trade_history_features
from ficc.utils.nelson_siegel_model import yield_curve_level
from ficc.utils.diff_in_days import diff_in_days_two_dates
from yield_model import yield_spread_model
from automated_training_auxiliary_variables import NUM_OF_DAYS_IN_YEAR, \
                                                   CATEGORICAL_FEATURES, \
                                                   CATEGORICAL_FEATURES_DOLLAR_PRICE, \
                                                   NON_CAT_FEATURES, \
                                                   NON_CAT_FEATURES_DOLLAR_PRICE, \
                                                   BINARY, \
                                                   BINARY_DOLLAR_PRICE, \
                                                   PREDICTORS, \
                                                   PREDICTORS_DOLLAR_PRICE, \
                                                   YS_VARIANTS, \
                                                   YS_FEATS, \
                                                   DP_VARIANTS, \
                                                   DP_FEATS, \
                                                   YEAR_MONTH_DAY, \
                                                   HOUR_MIN_SEC, \
                                                   QUERY_FEATURES, \
                                                   QUERY_CONDITIONS, \
                                                   ADDITIONAL_QUERY_CONDITIONS_FOR_YIELD_SPREAD_MODEL, \
                                                   ADDITIONAL_QUERY_FEATURES_FOR_DOLLAR_PRICE_MODEL, \
                                                   EASTERN, \
                                                   NUM_TRADES_IN_HISTORY_YIELD_SPREAD_MODEL, \
                                                   NUM_TRADES_IN_HISTORY_DOLLAR_PRICE_MODEL, \
                                                   CATEGORICAL_FEATURES_VALUES, \
                                                   SAVE_MODEL_AND_DATA, \
                                                   HOME_DIRECTORY, \
                                                   WORKING_DIRECTORY, \
                                                   BUCKET_NAME, \
                                                   EARLIST_TRADE_DATETIME, \
                                                   CUMULATIVE_DATA_PICKLE_FILENAME_YIELD_SPREAD, \
                                                   CUMULATIVE_DATA_PICKLE_FILENAME_DOLLAR_PRICE, \
                                                   OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA_YIELD_SPREAD, \
                                                   OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA_DOLLAR_PRICE, \
                                                   TTYPE_DICT, \
                                                   LONG_TIME_AGO_IN_NUM_SECONDS, \
                                                   MIN_TRADES_NEEDED_TO_BE_CONSIDERED_BUSINESS_DAY, \
                                                   SENDER_EMAIL, \
                                                   BATCH_SIZE, \
                                                   NUM_EPOCHS, \
                                                   DROPOUT, \
                                                   LEARNING_RATE, \
                                                   TESTING, \
                                                   EMAIL_RECIPIENTS, \
                                                   MODEL_FOLDERS, \
                                                    MAX_NUM_BUSINESS_DAYS_IN_THE_PAST_TO_CHECK


def get_creds():
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/user/ficc/mitas_creds.json'
    return None

def get_storage_client():
    get_creds()
    return storage.Client()


def get_bq_client():
    get_creds()
    return bigquery.Client()


STORAGE_CLIENT = get_storage_client()
BQ_CLIENT = get_bq_client()


def setup_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) == 0:
        print('No GPUs')
    else:
        for gpu in gpus:    # https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
            tf.config.experimental.set_memory_growth(gpu, True)


D_prev = dict()
P_prev = dict()
S_prev = dict()


def get_trade_history_columns(model: str) -> list:
    '''Creates a list of columns.'''
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    if model == 'yield_spread':
        variants = YS_VARIANTS
        feats = YS_FEATS
    else:
        variants = DP_VARIANTS
        feats = DP_FEATS
    
    columns = []
    for prefix in variants:
        for suffix in feats:
            columns.append(prefix + suffix)
    return columns


def target_trade_processing_for_attention(row):
    trade_mapping = {'D': [0,0], 'S': [0,1], 'P':[1,0]}
    target_trade_features = []
    target_trade_features.append(row['quantity'])
    target_trade_features = target_trade_features + trade_mapping[row['trade_type']]
    return np.tile(target_trade_features, (1, 1))


def replace_ratings_by_standalone_rating(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[data.sp_stand_alone.isna(), 'sp_stand_alone'] = 'NR'
    data.rating = data.rating.astype('str')
    data.sp_stand_alone = data.sp_stand_alone.astype('str')
    data.loc[(data.sp_stand_alone != 'NR'), 'rating'] = data[(data.sp_stand_alone != 'NR')]['sp_stand_alone'].loc[:]
    return data


def get_yield_for_last_duration(row, nelson_params, scalar_params, shape_parameter):
    if pd.isnull(row['last_calc_date'])or pd.isnull(row['last_trade_date']):
        # if there is no last trade, we use the duration of the current bond
        duration = diff_in_days_two_dates(row['maturity_date'], row['trade_date']) / NUM_OF_DAYS_IN_YEAR
        ycl = yield_curve_level(duration, row['trade_date'].date(), nelson_params, scalar_params, shape_parameter) / 100
        return ycl
    duration =  diff_in_days_two_dates(row['last_calc_date'], row['last_trade_date']) / NUM_OF_DAYS_IN_YEAR
    ycl = yield_curve_level(duration, row['trade_date'].date(), nelson_params, scalar_params, shape_parameter) / 100
    return ycl


@function_timer
def add_yield_curve(data):
    '''Add 'new_ficc_ycl' field to `data`.'''
    nelson_params = sqltodf('SELECT * FROM `eng-reactor-287421.ahmad_test.nelson_siegel_coef_daily` order by date desc', BQ_CLIENT)
    nelson_params.set_index('date', drop=True, inplace=True)
    nelson_params = nelson_params[~nelson_params.index.duplicated(keep='first')]
    nelson_params = nelson_params.transpose().to_dict()

    scalar_params = sqltodf('SELECT * FROM `eng-reactor-287421.ahmad_test.standardscaler_parameters_daily` order by date desc', BQ_CLIENT)
    scalar_params.set_index('date', drop=True, inplace=True)
    scalar_params = scalar_params[~scalar_params.index.duplicated(keep='first')]
    scalar_params = scalar_params.transpose().to_dict()

    shape_parameter = sqltodf('SELECT * FROM `eng-reactor-287421.ahmad_test.shape_parameters` order by Date desc', BQ_CLIENT)
    shape_parameter.set_index('Date', drop=True, inplace=True)
    shape_parameter = shape_parameter[~shape_parameter.index.duplicated(keep='first')]
    shape_parameter = shape_parameter.transpose().to_dict()

    data['last_trade_date'] = data['last_trade_datetime'].dt.date
    data['new_ficc_ycl'] = data[['last_calc_date',
                                 'last_settlement_date',
                                 'trade_date',
                                 'last_trade_date',
                                 'maturity_date']].parallel_apply(lambda row: get_yield_for_last_duration(row, nelson_params, scalar_params, shape_parameter), axis=1)
    data['new_ficc_ycl'] = data['new_ficc_ycl'] * 100
    return data


def decrement_business_days(date: str, num_business_days: int) -> str:
    '''Subtract `num_business_days` from `date`.'''
    return (datetime.strptime(date, YEAR_MONTH_DAY) - BusinessDay(num_business_days)).strftime(YEAR_MONTH_DAY)


def earliest_trade_from_new_data_is_same_as_last_trade_date(new_data: pd.DataFrame, last_trade_date) -> bool:
    '''Checks whether `last_trade_date` is the same as the date of the earliest trade in `new_data`. This 
    situation arises materialized trade history is created in the middle of the day, and so there are trades 
    on the same day that are still coming in. If we do not account for this case, then the automated training 
    fails since it searches for trades to populate the testing set as those after the `last_trade_date`.'''
    return new_data.trade_date.min().date().strftime(YEAR_MONTH_DAY) == last_trade_date


@function_timer
def get_new_data(file_name, model: str, use_treasury_spread: bool = False, optional_arguments_for_process_data: dict = {}):
    assert model in ('yield_spread', 'dollar_price'), f'Invalid value for model: {model}'
    query_features = QUERY_FEATURES
    query_conditions = QUERY_CONDITIONS
    if model == 'yield_spread':
        query_conditions = ADDITIONAL_QUERY_CONDITIONS_FOR_YIELD_SPREAD_MODEL + query_conditions
    else:
        query_features = query_features + ADDITIONAL_QUERY_FEATURES_FOR_DOLLAR_PRICE_MODEL
    
    old_data, last_trade_datetime, last_trade_date = get_data_and_last_trade_datetime(BUCKET_NAME, file_name)
    print(f'last trade datetime: {last_trade_datetime}')
    DATA_QUERY = get_data_query(last_trade_datetime, query_features, query_conditions)
    file_timestamp = datetime.now(EASTERN).strftime(YEAR_MONTH_DAY + '-%H:%M')

    trade_history_features = get_ys_trade_history_features(use_treasury_spread) if model == 'yield_spread' else get_dp_trade_history_features()
    num_features_for_each_trade_in_history = len(trade_history_features)
    num_trades_in_history = NUM_TRADES_IN_HISTORY_YIELD_SPREAD_MODEL if model == 'yield_spread' else NUM_TRADES_IN_HISTORY_DOLLAR_PRICE_MODEL
    raw_data_filepath = f'raw_data_{file_timestamp}.pkl'
    data_from_last_trade_datetime = process_data(DATA_QUERY, 
                                                 BQ_CLIENT, 
                                                 num_trades_in_history, 
                                                 num_features_for_each_trade_in_history, 
                                                 raw_data_filepath, 
                                                 save_data=SAVE_MODEL_AND_DATA, 
                                                 **optional_arguments_for_process_data)
    
    if data_from_last_trade_datetime is not None:
        if earliest_trade_from_new_data_is_same_as_last_trade_date(data_from_last_trade_datetime, last_trade_date):    # see explanation in docstring for `earliest_trade_from_new_data_is_same_as_last_trade_date(...)` as to why this scenario is important to handle
            decremented_last_trade_date = decrement_business_days(last_trade_date, 1)
            warnings.warn(f'Since the earliest trade from the new data is the same as the last trade date, we are decrementing the last trade date from {last_trade_date} to {decremented_last_trade_date}. This occurs because materialized trade history was created in the middle of the work day. If materialized trade history was not created during the middle of the work day, then investigate why we are inside this `if` statement.')
            last_trade_date = decremented_last_trade_date
        
        if model == 'dollar_price': data_from_last_trade_datetime = data_from_last_trade_datetime.rename(columns={'trade_history': 'trade_history_dollar_price'})    # change the trade history column name to match with `PREDICTORS_DOLLAR_PRICE`
    return old_data, data_from_last_trade_datetime, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath


def remove_old_trades(data: pd.DataFrame, num_days_to_keep: int, most_recent_trade_date: str = None, dataset_name: str = None) -> pd.DataFrame:
    '''Only keep `num_days_to_keep` days from the most recent trade in `data`.'''
    from_dataset_name = f' from {dataset_name}' if dataset_name is not None else ''
    most_recent_trade_date = data.trade_date.max() if most_recent_trade_date is None else pd.to_datetime(most_recent_trade_date)
    days_to_most_recent_trade = diff_in_days_two_dates(most_recent_trade_date, data.trade_date, 'exact')
    print(f'Removing trades{from_dataset_name} older than {num_days_to_keep} days before {most_recent_trade_date}')
    return data[days_to_most_recent_trade < num_days_to_keep]


@function_timer
def combine_new_data_with_old_data(old_data: pd.DataFrame, new_data: pd.DataFrame, model: str) -> pd.DataFrame:
    assert model in ('yield_spread', 'dollar_price'), f'Invalid value for model: {model}'
    if new_data is None: return old_data    # there is new data since `last_trade_date`

    num_trades_in_new_data = len(new_data)
    num_trades_in_old_data = 0 if old_data is None else len(old_data)
    print(f'Old data has {num_trades_in_old_data} trades. New data has {num_trades_in_new_data} trades')
    trade_history_feature_name = 'trade_history' if model == 'yield_spread' else 'trade_history_dollar_price'
    num_trades_in_history = NUM_TRADES_IN_HISTORY_YIELD_SPREAD_MODEL if model == 'yield_spread' else NUM_TRADES_IN_HISTORY_DOLLAR_PRICE_MODEL
    print(f'Restricting history to {num_trades_in_history} trades')
    new_data[trade_history_feature_name] = new_data[trade_history_feature_name].apply(lambda x: x[:num_trades_in_history])
    if old_data is not None: old_data[trade_history_feature_name] = old_data[trade_history_feature_name].apply(lambda x: x[:num_trades_in_history])    # done in case `num_trades_in_history` has decreased from before

    new_data = replace_ratings_by_standalone_rating(new_data)
    new_data['yield'] = new_data['yield'] * 100
    if model == 'yield_spread': new_data = add_yield_curve(new_data)
    new_data['target_attention_features'] = new_data.parallel_apply(target_trade_processing_for_attention, axis=1)

    new_data['trade_history_sum'] = new_data[trade_history_feature_name].parallel_apply(lambda x: np.sum(x))
    new_data.dropna(inplace=True, subset=['trade_history_sum'])
    print(f'Removed {num_trades_in_new_data - len(new_data)} trades, since these have null values in the trade history')
    new_data.issue_amount = new_data.issue_amount.replace([np.inf, -np.inf], np.nan)

    data = pd.concat([new_data, old_data]) if old_data is not None else new_data    # concatenating `new_data` to the original `data` dataframe
    if model == 'yield_spread': data['new_ys'] = data['yield'] - data['new_ficc_ycl']
    print(f'{len(data)} trades after combining new and old data')
    return data


@function_timer
def add_trade_history_derived_features(data: pd.DataFrame, model: str, use_treasury_spread: bool = False) -> pd.DataFrame:
    assert model in ('yield_spread', 'dollar_price'), f'Invalid value for model: {model}'
    data.sort_values('trade_datetime', inplace=True)    # when calling `trade_history_derived_features...(...)` the order of trades needs to be ascending for `trade_datetime`
    trade_history_derived_features = trade_history_derived_features_yield_spread(use_treasury_spread) if model == 'yield_spread' else trade_history_derived_features_dollar_price
    trade_history_feature_name = 'trade_history' if model == 'yield_spread' else 'trade_history_dollar_price'
    
    temp = data[['cusip', trade_history_feature_name, 'quantity', 'trade_type']].parallel_apply(trade_history_derived_features, axis=1)
    cols = get_trade_history_columns(model)
    data[cols] = pd.DataFrame(temp.tolist(), index=data.index)
    del temp

    data.sort_values('trade_datetime', ascending=False, inplace=True)
    return data


@function_timer
def drop_features_with_null_value(df: pd.DataFrame, features: list) -> pd.DataFrame:
    # df = df.dropna(subset=features)
    for feature in features:    # perform the procedure feature by feature to output how many trades are being removed for each feature
        num_trades_before = len(df)
        df = df.dropna(subset=[feature])
        num_trades_after = len(df)
        if num_trades_before != num_trades_after: print(f'Removed {num_trades_before - num_trades_after} trades for having a null value in feature: {feature}')
    return df


@function_timer
def save_data(data: pd.DataFrame, file_name: str) -> None:
    file_path = f'{WORKING_DIRECTORY}/files/{file_name}'
    data = remove_old_trades(data, 390, dataset_name='entire processed data file')    # 390 = 13 * 30, so we are keeping approximately 13 months of data in the file; decided to keep the last 13 months of data to go beyond one year and allow for future experiments with annual patterns without having to re-create the entire dataset
    print(f'Saving data to pickle file with name {file_path}')
    data.to_pickle(file_path)
    upload_data(STORAGE_CLIENT, BUCKET_NAME, file_name, file_path)


def _get_trade_date_where_data_exists(date, data: pd.DataFrame, max_number_of_business_days_to_go_back: int, exclusions_function: callable, on_or_after: str = 'after'):
    '''Iterate backwards on `date` until the data after `date` is non-empty. Go back a maximum of 
    `max_number_of_business_days_to_go_back` days. If `exclusions_function` is not `None`, assumes 
    that the function returns values, where the first item is the data after exclusions, and the 
    second item is the data before exclusions.'''
    valid_values_for_on_or_after = ('on', 'after')
    assert on_or_after in valid_values_for_on_or_after, f'Invalid value for `on_or_after`: {on_or_after}. Must be one of {valid_values_for_on_or_after}'

    def get_data_on_or_after_date(date_of_interest: str) -> pd.DataFrame:
        if on_or_after == 'after':
            data_for_date = data[data.trade_date > date_of_interest]
            data_for_date_earliest_date = data_for_date.trade_date.min()
            data_for_date = data_for_date[data_for_date.trade_date == data_for_date_earliest_date]    # restrict `data_for_date` to have only one day of trades
        else:
            data_for_date = data[data.trade_date == date_of_interest]
        return data_for_date
    
    previous_date = date
    data_for_date = get_data_on_or_after_date(previous_date)
    if exclusions_function is not None: data_for_date, _ = exclusions_function(data_for_date)
    business_days_gone_back = 0
    while len(data_for_date) < MIN_TRADES_NEEDED_TO_BE_CONSIDERED_BUSINESS_DAY and business_days_gone_back < max_number_of_business_days_to_go_back:
        business_days_gone_back += 1
        previous_date = decrement_business_days(date, business_days_gone_back)
        data_for_date = get_data_on_or_after_date(previous_date)
        if exclusions_function is not None: data_for_date, _ = exclusions_function(data_for_date)
    if business_days_gone_back == max_number_of_business_days_to_go_back:
        print(f'Went back {business_days_gone_back} and could not find any data; not going back any further, so returning the original `date`')
        return date
    return previous_date


def get_trade_date_where_data_exists_after_this_date(date, data: pd.DataFrame, max_number_of_business_days_to_go_back: int = 10, exclusions_function: callable = None):
    if not TESTING: return date
    return _get_trade_date_where_data_exists(date, data, max_number_of_business_days_to_go_back, exclusions_function, 'after')


def get_trade_date_where_data_exists_on_this_date(date, data: pd.DataFrame, max_number_of_business_days_to_go_back: int = 10, exclusions_function: callable = None):
    return _get_trade_date_where_data_exists(date, data, max_number_of_business_days_to_go_back, exclusions_function, 'on')


@function_timer
def create_input(data: pd.DataFrame, encoders: dict, model: str):
    assert model in ('yield_spread', 'dollar_price'), f'Invalid value for model: {model}'
    datalist = []
    trade_history_feature_name = 'trade_history' if model == 'yield_spread' else 'trade_history_dollar_price'
    datalist.append(np.stack(data[trade_history_feature_name].to_numpy()))
    datalist.append(np.stack(data['target_attention_features'].to_numpy()))

    categorical_features = CATEGORICAL_FEATURES if model == 'yield_spread' else CATEGORICAL_FEATURES_DOLLAR_PRICE
    non_cat_features = NON_CAT_FEATURES if model == 'yield_spread' else NON_CAT_FEATURES_DOLLAR_PRICE
    binary = BINARY if model == 'yield_spread' else BINARY_DOLLAR_PRICE

    noncat_and_binary = []
    for feature in non_cat_features + binary:
        noncat_and_binary.append(np.expand_dims(data[feature].to_numpy().astype('float32'), axis=1))
    datalist.append(np.concatenate(noncat_and_binary, axis=-1))
    
    for feature in categorical_features:
        encoded = encoders[feature].transform(data[feature])
        datalist.append(encoded.astype('float32'))

    label_name = 'new_ys' if model == 'yield_spread' else 'dollar_price'
    return datalist, data[label_name]


def get_data_and_last_trade_datetime(bucket_name: str, file_name: str):
    '''Get the dataframe from `bucket_name/file_name` and the most recent trade datetime from this dataframe.'''
    data = download_data(STORAGE_CLIENT, bucket_name, file_name)
    if data is None: return None, EARLIST_TRADE_DATETIME, EARLIST_TRADE_DATETIME[:10]    # get trades starting from `EARLIEST_TRADE_DATETIME` if we do not have these trades already in a pickle file; string representation of datetime has the date as the first 10 characters (YYYY-MM-DD is 10 characters)
    last_trade_datetime = data.trade_datetime.max().strftime(YEAR_MONTH_DAY + 'T' + HOUR_MIN_SEC)
    last_trade_date = data.trade_date.max().date().strftime(YEAR_MONTH_DAY)
    return data, last_trade_datetime, last_trade_date


def get_data_query(last_trade_datetime, features: list, conditions: list) -> str:
    features_as_string = ', '.join(features)
    conditions = conditions + [f'trade_datetime > "{last_trade_datetime}"']
    conditions_as_string = ' AND '.join(conditions)
    return f'''SELECT {features_as_string}
               FROM `eng-reactor-287421.auxiliary_views.materialized_trade_history`
               WHERE {conditions_as_string}
               ORDER BY trade_datetime DESC'''


def update_data(model: str):
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    filename = CUMULATIVE_DATA_PICKLE_FILENAME_YIELD_SPREAD if model == 'yield_spread' else CUMULATIVE_DATA_PICKLE_FILENAME_DOLLAR_PRICE
    optional_arguments_for_process_data = OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA_YIELD_SPREAD if model == 'yield_spread' else OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA_DOLLAR_PRICE
    use_treasury_spread = optional_arguments_for_process_data.get('use_treasury_spread', False)
    data_before_last_trade_datetime, data_from_last_trade_datetime, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath = get_new_data(filename, 
                                                                                                                                                              model, 
                                                                                                                                                              use_treasury_spread=use_treasury_spread, 
                                                                                                                                                              optional_arguments_for_process_data=optional_arguments_for_process_data)
    data = combine_new_data_with_old_data(data_before_last_trade_datetime, data_from_last_trade_datetime, model)
    data = add_trade_history_derived_features(data, model, use_treasury_spread)

    predictors = PREDICTORS if model == 'yield_spread' else PREDICTORS_DOLLAR_PRICE
    data = drop_features_with_null_value(data, predictors)
    if SAVE_MODEL_AND_DATA: save_data(data, filename)
    return data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath


def save_update_data_results_to_pickle_files(model: str):
    '''The function specified in `update_data` is called, and the 3 return values are stored as pickle files. If 
    testing, then first check whether the pickle files exist, before calling `update_data`. `suffix` is appended 
    to the end of the filename for each pickle file.'''
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    os.mkdir(f'{WORKING_DIRECTORY}/files')
    data_pickle_filepath = f'{WORKING_DIRECTORY}/files/data_from_update_data_{model}.pkl'
    last_trade_data_from_update_data_pickle_filepath = f'{WORKING_DIRECTORY}/files/last_trade_data_from_update_data_{model}.pkl'
    num_features_for_each_trade_in_history_pickle_filepath = f'{WORKING_DIRECTORY}/files/num_features_for_each_trade_in_history_{model}.pkl'
    
    # if not os.path.isdir(f'{WORKING_DIRECTORY}/files'): os.mkdir(f'{WORKING_DIRECTORY}/files')

    # if USE_PICKLED_DATA and os.path.isfile(data_pickle_filepath):
    #     print(f'Found a data file in {data_pickle_filepath}, so no need to run update_data(...)')
    #     raw_data_filepath = None
    #     data = pd.read_pickle(data_pickle_filepath)
    #     with open(last_trade_data_from_update_data_pickle_filepath, 'rb') as file: last_trade_date = pickle.load(file)
    #     with open(num_features_for_each_trade_in_history_pickle_filepath, 'rb') as file: num_features_for_each_trade_in_history = pickle.load(file)
    # else:
    #     data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath = update_data(model)
    #     data.to_pickle(data_pickle_filepath)
    #     with open(last_trade_data_from_update_data_pickle_filepath, 'wb') as file: pickle.dump(last_trade_date, file)
    #     with open(num_features_for_each_trade_in_history_pickle_filepath, 'wb') as file: pickle.dump(num_features_for_each_trade_in_history, file)


    data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath = update_data(model)
    data.to_pickle(data_pickle_filepath)
    with open(last_trade_data_from_update_data_pickle_filepath, 'wb') as file: pickle.dump(last_trade_date, file)
    with open(num_features_for_each_trade_in_history_pickle_filepath, 'wb') as file: pickle.dump(num_features_for_each_trade_in_history, file)
    return data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath


def fit_encoders(data: pd.DataFrame, categorical_features: list, model: str):
    '''Fits label encoders to categorical features in the data. For a few of the categorical features, the values 
    don't change for these features we use the pre-defined set of values specified in `CATEGORICAL_FEATURES_VALUES`. 
    Outputs a tuple of dictionaries where the first item is the encoders and the second item is the maximum value 
    for each class.'''
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    encoders = {}
    fmax = {}
    for feature in categorical_features:
        if feature in CATEGORICAL_FEATURES_VALUES:
            fprep = preprocessing.LabelEncoder().fit(CATEGORICAL_FEATURES_VALUES[feature])
        else:
            fprep = preprocessing.LabelEncoder().fit(data[feature].drop_duplicates())
        fmax[feature] = np.max(fprep.transform(fprep.classes_))
        encoders[feature] = fprep
    
    filename = 'encoders.pkl' if model == 'yield_spread' else 'encoders_dollar_price.pkl'
    with open(f'{WORKING_DIRECTORY}/{filename}', 'wb') as file:
        pickle.dump(encoders, file)
    return encoders, fmax


def _trade_history_derived_features(row, model: str, use_treasury_spread: bool = False) -> list:
    assert model in ('yield_spread', 'dollar_price'), f'Invalid value for model: {model}'
    if model == 'yield_spread':
        variants = YS_VARIANTS
        trade_history_features = get_ys_trade_history_features(use_treasury_spread)
    else:
        variants = DP_VARIANTS
        trade_history_features = get_dp_trade_history_features()

    ys_or_dp_idx = trade_history_features.index(model)
    par_traded_idx = trade_history_features.index('par_traded')
    trade_type1_idx = trade_history_features.index('trade_type1')
    trade_type2_idx = trade_history_features.index('trade_type2')
    seconds_ago_idx = trade_history_features.index('seconds_ago')


    def extract_feature_from_trade(row, name, trade):    # `name` is used solely for debugging
        ys_or_dp = trade[ys_or_dp_idx]
        ttypes = TTYPE_DICT[(trade[trade_type1_idx], trade[trade_type2_idx])] + row.trade_type
        seconds_ago = trade[seconds_ago_idx]
        quantity_diff = np.log10(1 + np.abs(10**trade[par_traded_idx] - 10**row.quantity))
        return [ys_or_dp, ttypes, seconds_ago, quantity_diff]


    global D_prev
    global S_prev
    global P_prev
    
    trade_history_feature_name = 'trade_history' if model == 'yield_spread' else 'trade_history_dollar_price'
    trade_history = row[trade_history_feature_name]
    most_recent_trade = trade_history[0]
    
    D_min_ago_t = D_prev.get(row.cusip, most_recent_trade)
    D_min_ago = LONG_TIME_AGO_IN_NUM_SECONDS        

    P_min_ago_t = P_prev.get(row.cusip, most_recent_trade)
    P_min_ago = LONG_TIME_AGO_IN_NUM_SECONDS
    
    S_min_ago_t = S_prev.get(row.cusip, most_recent_trade)
    S_min_ago = LONG_TIME_AGO_IN_NUM_SECONDS
    
    max_ys_or_dp_t = most_recent_trade
    max_ys_or_dp = most_recent_trade[ys_or_dp_idx]
    min_ys_or_dp_t = most_recent_trade
    min_ys_or_dp = most_recent_trade[ys_or_dp_idx]
    max_qty_t = most_recent_trade
    max_qty = most_recent_trade[par_traded_idx]
    min_ago_t = most_recent_trade
    min_ago = most_recent_trade[seconds_ago_idx]
    
    for trade in trade_history:
        seconds_ago = trade[seconds_ago_idx]
        # Checking if the first trade in the history is from the same block; TODO: shouldn't this be checked for every trade?
        if seconds_ago == 0: continue

        ys_or_dp = trade[ys_or_dp_idx]
        if ys_or_dp > max_ys_or_dp: 
            max_ys_or_dp_t = trade
            max_ys_or_dp =ys_or_dp
        elif ys_or_dp < min_ys_or_dp: 
            min_ys_or_dp_t = trade
            min_ys_or_dp = ys_or_dp

        par_traded = trade[par_traded_idx]
        if par_traded > max_qty: 
            max_qty_t = trade 
            max_qty = par_traded

        if seconds_ago < min_ago:    # TODO: isn't this just the most recent trade not in the same block, and isn't this initialized above already?
            min_ago_t = trade
            min_ago = seconds_ago
            
        side = TTYPE_DICT[(trade[trade_type1_idx], trade[trade_type2_idx])]
        if side == 'D':
            if seconds_ago < D_min_ago: 
                D_min_ago_t = trade
                D_min_ago = seconds_ago
                D_prev[row.cusip] = trade
        elif side == 'P':
            if seconds_ago < P_min_ago: 
                P_min_ago_t = trade
                P_min_ago = seconds_ago
                P_prev[row.cusip] = trade
        elif side == 'S':
            if seconds_ago < S_min_ago: 
                S_min_ago_t = trade
                S_min_ago = seconds_ago
                S_prev[row.cusip] = trade
        else: 
            print('invalid side', trade)
    
    variant_trade_dict = dict(zip(variants, [max_ys_or_dp_t, min_ys_or_dp_t, max_qty_t, min_ago_t, D_min_ago_t, P_min_ago_t, S_min_ago_t]))
    variant_trade_list = []
    for variant_name, variant_trade in variant_trade_dict.items():
        feature_list = extract_feature_from_trade(row, variant_name, variant_trade)
        variant_trade_list += feature_list
    return variant_trade_list


def trade_history_derived_features_yield_spread(use_treasury_spread) -> callable:
    return lambda row: _trade_history_derived_features(row, 'yield_spread', use_treasury_spread)
def trade_history_derived_features_dollar_price(row) -> list:
    return _trade_history_derived_features(row, 'dollar_price')

def apply_exclusions(data):
        data_before_exclusions = data[:]
        data = data[(data.days_to_call == 0) | (data.days_to_call > np.log10(400))]
        data = data[(data.days_to_refund == 0) | (data.days_to_refund > np.log10(400))]
        data = data[(data.days_to_maturity == 0) | (data.days_to_maturity > np.log10(400))]
        data = data[data.days_to_maturity < np.log10(30000)]
        data = data[~data.last_calc_date.isna()]
        return data, data_before_exclusions

@function_timer
def get_model_results(data: pd.DataFrame, trade_date: str, model: str, loaded_model, encoders: dict, exclusions_function: callable = None) -> pd.DataFrame:
    '''NOTE: `model` is a string that denotes whether we are working with the yield spread model or 
    the dollar price model, and `loaded_model` is an actual keras model. This may cause confusion.
    If `exclusions_function` is not `None`, assumes that the function returns values, where the first 
    item is the data after exclusions, and the second item is the data before exclusions.'''
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    data_on_trade_date = data[data.trade_date == trade_date]
    if exclusions_function is not None: data_on_trade_date, _ = exclusions_function(data_on_trade_date)
    inputs, labels = create_input(data_on_trade_date, encoders, model)
    return create_summary_of_results(loaded_model, data_on_trade_date, inputs, labels)


def not_enough_trades_in_test_data(test_data, min_trades_to_use_test_data):
    '''Return `None` for the number of arguments that `train_model(...)` returns.'''
    print(f'No model is trained since there are only {len(test_data)} trades in `test_data`, which is less than our chosen threshold of {min_trades_to_use_test_data}; `train_model(...)` is terminated')
    return None, None, None, None, None, None, None, ''

def create_summary_of_results(model, data: pd.DataFrame, inputs: list, labels: list):
    '''Creates a dataframe that can be sent as a table over email for the performance of `model` on 
    `inputs` validated on `labels`. `inputs` and `labels` are transformed using `create_input(...)` 
    from `data` to be able to be used by the `model` for inference.'''
    try:
        predictions = model.predict(inputs, batch_size=BATCH_SIZE).flatten()
        delta = np.abs(predictions - labels)
        result_df = segment_results(data, delta)
    except Exception as e:
        print(f'Unable to create results dataframe with `segment_results(...)`. {type(e)}:', e)
        result_df = pd.DataFrame()

    try:
        print(result_df.to_markdown())
    except Exception as e:
        print(f'Unable to display results dataframe with .to_markdown(). Need to run `pip install tabulate` on this machine in order to display the dataframe in an easy to read way. {type(e)}:', e)
    return result_df
    
def load_model_from_date(date: str, folder: str, bucket: str):
    '''Taken almost directly from `point_in_time_pricing_timestamp.py`.
    When using the `cache_output` decorator, we should not have any optional arguments as this may interfere with 
    how the cache lookup is done (optional arguments may not be put into the args set).'''
    assert folder in MODEL_FOLDERS
    # removing the following line of code to align with production; models are now labelled using YYYY-MM-DD not MM-DD
    # if len(date) == 10: date = date[5:]    # remove the year and the hypen from `date`, i.e., remove 'YYYY-' from `date`, if the date is 10 characters which we assume to mean that it is in YYYY-MM-DD format 
    model_prefix = '' if folder == 'yield_spread_model' else 'dollar-'
    bucket_folder_model_path = os.path.join(os.path.join(bucket, folder), f'{model_prefix}model-{date}')    # create path of the form: <bucket>/<folder>/<model>
    base_model_path = os.path.join(bucket, f'{model_prefix}model-{date}')    # create path of the form: <bucket>/<model>
    for model_path in (bucket_folder_model_path, base_model_path):    # iterate over possible paths and try to load the model
        print(f'Attempting to load model from {model_path}')
        try:
            model = keras.models.load_model(model_path)
            print(f'Model loaded from {model_path}')
            return model
        except Exception as e:
            print(f'Model failed to load from {model_path} with exception: {e}')


@function_timer
def load_model(date_of_interest: str, folder: str, max_num_business_days_in_the_past_to_check: int = MAX_NUM_BUSINESS_DAYS_IN_THE_PAST_TO_CHECK, bucket: str = 'gs://'+BUCKET_NAME):
    '''Taken almost directly from `point_in_time_pricing_timestamp.py`.
    This function finds the appropriate model, either in the automated_training directory, or in a special directory. 
    TODO: clean up the way we store models on cloud storage by unifying the folders and naming convention and adding the 
    year to the name.'''
    for num_business_days_in_the_past in range(max_num_business_days_in_the_past_to_check):
        model_date_string = decrement_business_days(date_of_interest, num_business_days_in_the_past)
        model = load_model_from_date(model_date_string, folder, bucket)
        if model is not None: return model, model_date_string
    raise FileNotFoundError(f'No model for {folder} was found from {date_of_interest} to {model_date_string}')
    

@function_timer
def train_model(data: pd.DataFrame, last_trade_date: str, model: str, num_features_for_each_trade_in_history: int, date_for_previous_model: str, exclusions_function: callable = None):
    '''The final return value is a string that should be in the beginning of the email that is sent out 
    which provides transparency on the training procedure.'''
    
    print(f'TESTING = {TESTING}')
    print(f'SAVE_MODEL_AND_DATA = {SAVE_MODEL_AND_DATA}')

    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    train_model_text_list = []
    data = remove_old_trades(data, 240, last_trade_date, dataset_name='training/testing dataset')    # 240 = 8 * 30, so we are using approximately 8 months of data for training
    categorical_features = CATEGORICAL_FEATURES if model == 'yield_spread' else CATEGORICAL_FEATURES_DOLLAR_PRICE
    encoders, fmax = fit_encoders(data, categorical_features, model)
    
    if TESTING: 
        last_trade_date = get_trade_date_where_data_exists_after_this_date(last_trade_date, data, exclusions_function=exclusions_function)
        print(last_trade_date)
    test_data = data[data.trade_date > last_trade_date]
    test_data_date = test_data.trade_date.min()
    test_data = test_data[test_data.trade_date == test_data_date]    # restrict `test_data` to have only one day of trades
    if len(test_data) < MIN_TRADES_NEEDED_TO_BE_CONSIDERED_BUSINESS_DAY: return not_enough_trades_in_test_data(test_data, MIN_TRADES_NEEDED_TO_BE_CONSIDERED_BUSINESS_DAY)
    test_data_date = test_data_date.strftime(YEAR_MONTH_DAY)
    if exclusions_function is not None:
        test_data, test_data_before_exclusions = exclusions_function(test_data, 'test_data')
    else:
        test_data_before_exclusions = test_data
    
    train_data = data[data.trade_date <= last_trade_date]
    training_set_info = f'Training set contains {len(train_data)} trades ranging from trade datetimes of {train_data.trade_datetime.min()} to {train_data.trade_datetime.max()}'
    test_set_info = f'Test set contains {len(test_data)} trades ranging from trade datetimes of {test_data.trade_datetime.min()} to {test_data.trade_datetime.max()}'
    print(training_set_info)
    print(test_set_info)
    train_model_text_list.extend([training_set_info, test_set_info])

    non_cat_features = NON_CAT_FEATURES if model == 'yield_spread' else NON_CAT_FEATURES_DOLLAR_PRICE
    binary = BINARY if model == 'yield_spread' else BINARY_DOLLAR_PRICE

    x_train, y_train = create_input(train_data, encoders, model)
    x_test, y_test = create_input(test_data, encoders, model)

    yield_spread_model_or_dollar_price_model = yield_spread_model if model == 'yield_spread' else dollar_price_model
    num_trades_in_history = NUM_TRADES_IN_HISTORY_YIELD_SPREAD_MODEL if model == 'yield_spread' else NUM_TRADES_IN_HISTORY_DOLLAR_PRICE_MODEL
    untrained_model = yield_spread_model_or_dollar_price_model(x_train, 
                                                               num_trades_in_history, 
                                                               num_features_for_each_trade_in_history, 
                                                               categorical_features, 
                                                               non_cat_features, 
                                                               binary, 
                                                               fmax)
    trained_model, mae, history = train_and_evaluate_model(untrained_model, x_train, y_train, x_test, y_test)

    create_summary_of_results_for_test_data = lambda model: create_summary_of_results(model, test_data, x_test, y_test)
    result_df = create_summary_of_results_for_test_data(trained_model)
    try:
        models_folder = 'yield_spread_model' if model == 'yield_spread' else 'dollar_price_models'
        previous_business_date_model, previous_business_date_model_date = load_model(date_for_previous_model, models_folder)
        result_df_using_previous_day_model = create_summary_of_results_for_test_data(previous_business_date_model)
    except Exception as e:
        print(f'Unable to create the dataframe for the model evaluation email due to {type(e)}: {e}')
        print('Stack trace:')
        print(traceback.format_exc())
        result_df_using_previous_day_model = None
        if previous_business_date_model not in locals(): previous_business_date_model, previous_business_date_model_date = None, None
    
    # uploading predictions to bigquery (only for yield spread model)
    if SAVE_MODEL_AND_DATA and model == 'yield_spread':
        try:
            test_data_before_exclusions_x_test, _ = create_input(test_data_before_exclusions, encoders, 'yield_spread')
            test_data_before_exclusions['new_ys_prediction'] = trained_model.predict(test_data_before_exclusions_x_test, batch_size=BATCH_SIZE)
            test_data_before_exclusions = test_data_before_exclusions[['rtrs_control_number', 'cusip', 'trade_date', 'dollar_price', 'yield', 'new_ficc_ycl', 'new_ys', 'new_ys_prediction']]
            test_data_before_exclusions['prediction_datetime'] = pd.to_datetime(datetime.now().replace(microsecond=0))
            test_data_before_exclusions['trade_date'] = pd.to_datetime(test_data_before_exclusions['trade_date']).dt.date
            upload_predictions(test_data_before_exclusions, model)
                
        except Exception as e:
            print(f'Failed to upload predictions to BigQuery. {type(e)}:', e)
    return trained_model, test_data_date, previous_business_date_model, previous_business_date_model_date, encoders, mae, (result_df, result_df_using_previous_day_model), '<br>'.join(train_model_text_list)    # use '<br>' for the separator since this will create a new line in the HTML body that will be sent out by email

def get_table_schema_for_predictions():
    '''Returns the schema required for the bigquery table storing the predictions.'''
    schema = [
        bigquery.SchemaField('rtrs_control_number', 'INTEGER', 'REQUIRED'),
        bigquery.SchemaField('cusip', 'STRING', 'REQUIRED'),
        bigquery.SchemaField('trade_date', 'DATE', 'REQUIRED'),
        bigquery.SchemaField('dollar_price', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('yield','FLOAT', 'REQUIRED'),
        bigquery.SchemaField('new_ficc_ycl', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('new_ys', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('new_ys_prediction', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('prediction_datetime', 'DATETIME', 'REQUIRED')
    ] 
    return schema

def upload_predictions(data: pd.DataFrame, model: str) -> str:
    '''Upload the coefficient and scalar dataframe to BigQuery. Returns the table name.'''
    assert model in HISTORICAL_PREDICTION_TABLE, f'Trying to upload predictions for {model}, but can only upload predictions for the following models: {list(HISTORICAL_PREDICTION_TABLE.keys())}'
    table_name = HISTORICAL_PREDICTION_TABLE[model]
    job_config = bigquery.LoadJobConfig(schema=get_table_schema_for_predictions(), write_disposition='WRITE_APPEND')
    job = BQ_CLIENT.load_table_from_dataframe(data, table_name, job_config=job_config)
    try:
        job.result()
        print(f'Upload successful to {table_name}')
    except Exception as e:
        print(f'Failed to upload to {table_name} with {type(e)}: {e}')
        # raise e

def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    fit_callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=20,
                                                   verbose=0,
                                                   mode='auto',
                                                   restore_best_weights=True)]

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=keras.losses.MeanAbsoluteError(),
                  metrics=[keras.metrics.MeanAbsoluteError()])

    history = model.fit(x_train, 
                        y_train, 
                        epochs=NUM_EPOCHS, 
                        batch_size=BATCH_SIZE,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=fit_callbacks,
                        use_multiprocessing=True,
                        workers=8) 

    _, mae = model.evaluate(x_test, 
                            y_test, 
                            verbose=1, 
                            batch_size=BATCH_SIZE)
    return model, mae, history


def segment_results(data: pd.DataFrame, delta: np.array) -> pd.DataFrame:
    def get_mae_and_count(condition=None):
        if condition is not None:
            delta_cond, data_cond = delta[condition], data[condition]
        else:
            delta_cond, data_cond = delta, data
        return np.round(np.mean(delta_cond), 3), data_cond.shape[0]    # round mae to 3 digits after the decimal point to reduce noise

    total_mae, total_count = get_mae_and_count()
    inter_dealer = data.trade_type == 'D'
    dd_mae, dd_count = get_mae_and_count(inter_dealer)
    dealer_purchase = data.trade_type == 'P'
    dp_mae, dp_count = get_mae_and_count(dealer_purchase)
    dealer_sell = data.trade_type == 'S'
    ds_mae, ds_count = get_mae_and_count(dealer_sell)
    aaa = data.rating == 'AAA'
    aaa_mae, aaa_count = get_mae_and_count(aaa)
    investment_grade_ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
    investment_grade = data.rating.isin(investment_grade_ratings)
    investment_grade_mae, investment_grade_count = get_mae_and_count(investment_grade)
    par_traded_greater_than_or_equal_to_100k = data.par_traded >= 1e5
    hundred_k_mae, hundred_k_count = get_mae_and_count(par_traded_greater_than_or_equal_to_100k)

    result_df = pd.DataFrame(data=[[total_mae, total_count],
                                   [dd_mae, dd_count],
                                   [dp_mae, dp_count],
                                   [ds_mae, ds_count], 
                                   [aaa_mae, aaa_count], 
                                   [investment_grade_mae, investment_grade_count],
                                   [hundred_k_mae, hundred_k_count]],
                             columns=['Mean Absolute Error', 'Trade Count'],
                             index=['Entire set', 'Dealer-Dealer', 'Dealer-Purchase', 'Dealer-Sell', 'AAA', 'Investment Grade', 'Trade size >= 100k'])
    return result_df


@function_timer
def save_model(trained_model, encoders, model: str):
    '''NOTE: `model` is a string that denotes whether we are working with the yield spread model or 
    the dollar price model, and `trained_model` is an actual keras model. This may cause confusion.'''
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    if trained_model is None:
        print('trained_model is `None` and so not saving it to storage')
        return None
    suffix = '_dollar_price' if model == 'dollar_price' else ''
    suffix_wo_underscore = 'dollar_price' if model == 'dolar_price' else ''    # need this variable as well since past implementations of this function have model naming missing an underscore

    file_timestamp = datetime.now(EASTERN).strftime(YEAR_MONTH_DAY + '-%H-%M')
    print(f'file time stamp: {file_timestamp}')

    print('Saving encoders and uploading encoders')
    encoders_filename = f'encoders{suffix}.pkl'
    encoders_filepath = f'{WORKING_DIRECTORY}/files/{encoders_filename}'
    if not os.path.isdir(f'{WORKING_DIRECTORY}/files'): os.mkdir(f'{WORKING_DIRECTORY}/files')
    with open(encoders_filepath, 'wb') as file:
        pickle.dump(encoders, file)    
    upload_data(STORAGE_CLIENT, BUCKET_NAME, encoders_filename, encoders_filepath)

    print('Saving and uploading model')
    folder = 'dollar_price_models' if model == 'dollar_price' else 'yield_spread_models'
    if not os.path.isdir(f'{HOME_DIRECTORY}/trained_models'): os.mkdir(f'{HOME_DIRECTORY}/trained_models')
    model_filename = f'{HOME_DIRECTORY}/trained_models/{folder}/saved_models/saved_model_{suffix_wo_underscore}{file_timestamp}'
    trained_model.save(model_filename)
    
    model_zip_filename = f'model{suffix}'
    model_zip_filepath = f'{HOME_DIRECTORY}/trained_models/{model_zip_filename}'
    shutil.make_archive(model_zip_filepath, 'zip', model_filename)
    # shutil.make_archive(f'saved_model_{file_timestamp}', 'zip', f'saved_model_{file_timestamp}')
    
    upload_data(STORAGE_CLIENT, BUCKET_NAME, f'{model_zip_filename}.zip', f'{model_zip_filepath}.zip')
    # upload_data(STORAGE_CLIENT, f'{BUCKET_NAME}/{folder}', f'saved_model_{file_timestamp}.zip')
    os.system(f'rm -r {model_filename}')


def send_email(sender_email: str, message: str, recipients: list) -> None:
    '''Send email with `message` to `recipients` from `sender_email`.'''
    smtp_server = 'smtp.gmail.com'
    port = 587

    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()
            sender_password = 'ztwbwrzdqsucetbg'
            server.login(sender_email, sender_password)
            for receiver in recipients:
                server.sendmail(sender_email, receiver, message.as_string())
        except Exception as e:
            print(e)
        finally:
            server.quit()


def _get_email_subject(model_train_date: str, model: str) -> str:
    testing_addendum = '' if not TESTING else 'TESTING: '
    return f'{testing_addendum} Pipeline MAE for {model} model trained on {model_train_date}'


def send_results_email(mae, model_train_date: str, recipients: list, model: str) -> None:
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    print(f'Sending email to {recipients}')
    
    msg = MIMEMultipart()
    msg['Subject'] = _get_email_subject(model_train_date, model)
    msg['From'] = SENDER_EMAIL

    message = MIMEText(f'The MAE for the model on trades that occurred on {model_train_date} is {np.round(mae, 3)}.', 'plain')
    msg.attach(message)
    send_email(SENDER_EMAIL, msg, recipients)


# def send_results_email_table(data_metadata, model_metadata, result_df, result_df_before_exclusions, last_trade_date, recipients:list):
#         print(f'Sending email to {recipients}')
#         sender_email = 'notifications@ficc.ai'
        
#         msg = MIMEMultipart()
#         if TESTING:
#             msg['Subject'] = f'(TESTING)(MODEL-TRAINING-COMPONENT) MAE for model trained till {last_trade_date}'
#         else:
#              msg['Subject'] = f'(MODEL-TRAINING-COMPONENT) MAE for model trained till {last_trade_date}'
#         msg['From'] = sender_email

#         if TESTING:
#             #if debugging, send metadata for dataset and model components 
#             metadata_html = '<h2>Data Processing Component Metadata:</h2>'
#             for key, value in data_metadata.items():
#                 metadata_html += f'<p>{key}: {value}</p>'
#             metadata_body = MIMEText(metadata_html, 'html')
#             msg.attach(metadata_body)

#             metadata_html = '<h2>Model Training Component Metadata:</h2>'
#             for key, value in model_metadata.items():
#                 metadata_html += f'<p>{key}: {value}</p>'
#             metadata_body = MIMEText(metadata_html, 'html')
#             msg.attach(metadata_body)

#         html_table_before_exclusions = '<caption>Results Before Exclusions</caption>' + result_df_before_exclusions.to_html(index=True)
#         html_table_after_exclusions = '<caption>Results After Exclusions</caption>' + result_df.to_html(index=True)
#         html_combined = f'<table>{html_table_before_exclusions}</table><br><table>{html_table_after_exclusions}</table>'
#         body = MIMEText(html_combined, 'html')
#         msg.attach(body)
#         send_email(sender_email, msg, recipients)



def send_results_email_multiple_tables(df_list: list, text_list: list, model_train_date: str, recipients: list, model: str, intro_text: str = ''):
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    print(f'Sending email to {recipients}')
    msg = MIMEMultipart()
    msg['Subject'] = _get_email_subject(model_train_date, model)
    msg['From'] = SENDER_EMAIL

    def html_text_for_single_df(text, df):
        return f'''{text}<br>{df.to_html(index=True)}'''

    if intro_text != '': intro_text = intro_text + '<hr>'
    html_text = f'''
    <html>
    <body>
    {intro_text}

    Model ran with the following parameters:<br>
    - BATCH_SIZE = {BATCH_SIZE}<br>
    - EPOCHS = {NUM_EPOCHS}<br>
    - DROPOUT = {DROPOUT}<br>
    <hr>
    {'<hr>'.join([html_text_for_single_df(text, df) for text, df in zip(text_list, df_list)])}
    </body>
    </html>
    '''
    body = MIMEText(html_text, 'html')
    msg.attach(body)
    send_email(SENDER_EMAIL, msg, recipients)


def send_no_new_model_email(last_trade_date, recipients: list, model: str) -> None:
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    print(f'Sending email to {recipients}')
    msg = MIMEMultipart()
    msg['Subject'] = f'Not enough new data was found on {last_trade_date}, so no new {model} model was trained; need at least {MIN_TRADES_NEEDED_TO_BE_CONSIDERED_BUSINESS_DAY} new trades to train a new model'
    msg['From'] = SENDER_EMAIL
    send_email(SENDER_EMAIL, msg, recipients)