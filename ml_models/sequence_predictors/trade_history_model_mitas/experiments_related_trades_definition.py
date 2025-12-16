from collections import defaultdict
import os
import gc
import time
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

import datetime

from pytorch_lightning import seed_everything

from ficc.utils.auxiliary_variables import VERY_LARGE_NUMBER, \
                                           IDENTIFIERS, \
                                           CATEGORICAL_FEATURES, \
                                           NON_CAT_FEATURES, \
                                           BINARY, \
                                           TRADE_HISTORY, \
                                           NUM_OF_DAYS_IN_YEAR
from ficc.utils.auxiliary_functions import flatten
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.trade_dict_to_list import TRADE_TYPE_MAPPING, \
                                          FEATURES_IN_HISTORY, \
                                          FEATURES_TO_INDEX_IN_HISTORY, \
                                          CATEGORICAL_FEATURES_IN_HISTORY, \
                                          quantity_diff
from ficc.utils.trade_dict_to_list_mappings import TRADE_TYPE_MAPPING, \
                                                   TRADE_TYPE_CROSS_PRODUCT_MAPPING, \
                                                   RATING_TO_INT_MAPPING
from ficc.utils.related_trade import append_recent_trade_data, get_appended_feature_name

import sys
sys.path.insert(0,'../')

from trade_history_model_mitas.data_prep import get_past_trade_columns, \
                                                convert_trade_type_encoding_to_actual

from yield_spread_model_mitas.data_prep import FEATURES_AND_NAN_REPLACEMENT_VALUES, \
                                               ADDITIONAL_CATEGORICAL_FEATURES, \
                                               get_datestring_from_filename, \
                                               remove_rows_with_feature_value, \
                                               replace_rating_with_standalone_rating, \
                                               add_past_trades_info, \
                                               check_additional_features, \
                                               replace_nan_for_features, \
                                               encode_and_get_encoders, \
                                               encode_with_encoders
from yield_spread_model_mitas.train import get_train_test_data_trade_datetime, \
                                           is_gpu_available, \
                                           is_mps_available
from yield_spread_model_mitas.tree_models import train_lightgbm_model

from rating_model_mitas.data_prep import read_processed_file_pickle, \
                                         remove_fields_with_single_unique_value, \
                                         remove_rows_with_nan_value


# if True: data files are created and no LightGBM experiments are run; if False: LightGBM experiments are run, assuming the data files already exist
JUST_GENERATING_DATA = False


# default value of 0 is chosen for settlement_date_to_calc_date because we exclude bonds that have a calc date that is fewer than 400 days into the future, and so a true value of settlement date to calc date will never be close to 0
DEFAULT_VALUES_NONZERO_PADDING = {'quantity_diff': np.log10(VERY_LARGE_NUMBER),    # model should learn that a quantity diff that is very large, i.e., current quantity is much smaller than quantity being compared to, means that the trade is less meaningful to use for pricing since the trades are very different. Could also be -np.log10(VERY_LARGE_NUMBER) for the same reason
                                  'seconds_ago': np.log10(VERY_LARGE_NUMBER)}    # model should learn that if the trade being compared to is far back in the past, then it is less meaningful to pricing the current trade
DEFAULT_VALUES_NONZERO_PADDING = defaultdict(int, DEFAULT_VALUES_NONZERO_PADDING)    # constructing a defaultdict from a dict: https://stackoverflow.com/questions/7539115/how-to-construct-a-defaultdict-from-a-dictionary

DEFAULT_VALUES_ZERO_PADDING = defaultdict(int)

DEFAULT_VALUES = DEFAULT_VALUES_ZERO_PADDING


NUM_TRADES_IN_TRADE_HISTORY = 1    # maximum number of past trades in the history


TARGET = ['yield_spread']


DATA_PROCESSING_FEATURES = ['trade_datetime',    # used to split the data into training and test sets
                            'settlement_date',    # used (in conjunction with calc_date) to create the settlement_date_to_calc_date feature in past trades
                            'calc_date',    # used (in conjunction with settlement_date) to create the settlement_date_to_calc_date feature in past trades
                            'calc_day_cat',    # added in the past trades
                            # 'coupon_type'    # used to group related trades; currently commented out since there is only a single value of 8 present in the data
                           ]


processed_file_pickle = '../../../../ficc/ml_models/sequence_predictors/data/processed_data_ficc_ycl_long_history_2022-10-08-00-00.pkl'
processed_file_pickle_datestring = get_datestring_from_filename(processed_file_pickle)
trade_data = read_processed_file_pickle(processed_file_pickle)


trade_data = trade_data[trade_data.trade_datetime < datetime.datetime(2022, 10, 8)]    # keep all trades before October 1, to standardize with Charles


trade_data = trade_data[(trade_data.days_to_call == 0) | (trade_data.days_to_call > np.log10(400))]
trade_data = trade_data[(trade_data.days_to_refund == 0) | (trade_data.days_to_refund > np.log10(400))]
trade_data = trade_data[trade_data.days_to_maturity < np.log10(30000)]
trade_data = trade_data[trade_data.sinking == False]
trade_data = trade_data[trade_data.incorporated_state_code != 'VI']
trade_data = trade_data[trade_data.incorporated_state_code != 'GU']
# trade_data = trade_data[(trade_data.coupon_type == 8)]
# trade_data = trade_data[trade_data.is_called == False]

# restructured bonds and high chance of default bonds are removed
trade_data = remove_rows_with_feature_value(trade_data, 'purpose_sub_class', [6, 20, 21, 22, 44, 57, 90, 106])
# pre-refunded bonds and partially refunded bonds are removed
trade_data = remove_rows_with_feature_value(trade_data, 'called_redemption_type', [18, 19])


trade_data = replace_rating_with_standalone_rating(trade_data)


NON_CAT_FEATURES.append('ficc_treasury_spread')


ADDITIONAL_CATEGORICAL_FEATURES = check_additional_features(trade_data, ADDITIONAL_CATEGORICAL_FEATURES)

trade_data, _ = replace_nan_for_features(trade_data, FEATURES_AND_NAN_REPLACEMENT_VALUES, verbose=True)
trade_data = remove_fields_with_single_unique_value(trade_data, BINARY + CATEGORICAL_FEATURES + ADDITIONAL_CATEGORICAL_FEATURES + NON_CAT_FEATURES)

all_features_set = set(trade_data.columns)
BINARY = list(set(BINARY) & all_features_set)
CATEGORICAL_FEATURES = list((set(CATEGORICAL_FEATURES) | set(ADDITIONAL_CATEGORICAL_FEATURES)) & all_features_set)
NON_CAT_FEATURES = list(set(NON_CAT_FEATURES) & all_features_set)
PREDICTORS = BINARY + CATEGORICAL_FEATURES + NON_CAT_FEATURES

trade_data = trade_data[IDENTIFIERS + 
                        PREDICTORS + 
                        DATA_PROCESSING_FEATURES + 
                        TRADE_HISTORY + 
                        TARGET]

trade_data = remove_rows_with_nan_value(trade_data)


print(f'Identifiers: {sorted(IDENTIFIERS)}')
print(f'Predictors: {sorted(PREDICTORS)}')
print(f'Binary features: {sorted(BINARY)}')
print(f'Categorical features: {sorted(CATEGORICAL_FEATURES)}')
print(f'Numerical features: {sorted(NON_CAT_FEATURES)}')


PREDICTORS_WITHOUT_LAST_TRADE_FEATURES = [predictor for predictor in PREDICTORS if not predictor.startswith('last')]
print(f'The following features are in PREDICTORS but not in PREDICTORS_WITHOUT_LAST_TRADE_FEATURES: {set(PREDICTORS) - set(PREDICTORS_WITHOUT_LAST_TRADE_FEATURES)}')


columns_with_dtype_object = [column for column in CATEGORICAL_FEATURES if trade_data[column].dtype == 'object']
print(f'Converting the following features to dtype category: {columns_with_dtype_object}')
if columns_with_dtype_object: trade_data[columns_with_dtype_object] = trade_data[columns_with_dtype_object].astype('category')    # converting multiple columns to `category` dtype in one line: https://stackoverflow.com/questions/28910851/python-pandas-changing-some-column-types-to-categories


# sort by trade_datetime since order can be changed when reading pickle file into m1 since it loads by chunks
trade_data = trade_data.sort_values(by='trade_datetime', ascending=False)


oldest_trade_datetime = trade_data['trade_datetime'].iloc[-1]
newest_trade_datetime = trade_data['trade_datetime'].iloc[0]

print(f'Oldest trade datetime: {oldest_trade_datetime}.\
    Newest trade datetime: {newest_trade_datetime}.\
    Gap: {newest_trade_datetime - oldest_trade_datetime}')
print(f'Total number of trades: {len(trade_data)}')


DATE_TO_SPLIT = datetime.datetime(2022, 9, 15)    # September 15 2022


train_data, test_data = get_train_test_data_trade_datetime(trade_data, DATE_TO_SPLIT)
print(f'Number of trades for training: {len(train_data)}.\
    Number of trades for testing: {len(test_data)}')
assert len(train_data) != 0 and len(test_data) != 0, 'Either train or test data is empty. Consider checking how the train test split is being performed.'


trade_data_flattened_trade_history, \
    additional_binary_features_from_past_trades, \
    additional_noncat_features_from_past_trades, \
    past_trade_feature_groups = add_past_trades_info(trade_data, NUM_TRADES_IN_TRADE_HISTORY - 1, FEATURES_TO_INDEX_IN_HISTORY)
past_trade_feature_groups_flattened = flatten(past_trade_feature_groups)
print(f'Each of the past trades are in the following feature groups: {past_trade_feature_groups}')


TRADE_TYPE_NEW_COLUMN = 'trade_type'


SAME_CUSIP_PREFIX = 'last_'


make_data_filename = lambda name: f'data/{name}.pkl'    # used to create a filename to save the PyTorch model parameters
if not os.path.isdir('data/'):
    os.mkdir('data/')


label_encoders = encode_and_get_encoders(train_data, BINARY, CATEGORICAL_FEATURES)[1]    # index 1 corresponds to the label encoders, whereas index 0 (ignored in this expression) corresponds to the encoded dataframe

label_encoders_filepath = make_data_filename('label_encoders')
with open(label_encoders_filepath, 'wb') as pickle_handle: pickle.dump(label_encoders, pickle_handle, protocol=4)    # protocol 4 allows for use in the VM; use template from https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
encode_with_label_encoders = lambda df, features_to_exclude=[]: encode_with_encoders(df, label_encoders, features_to_exclude)


make_filename = lambda name: f'pt/{name}.pt'    # used to create a filename to save the PyTorch model parameters
if not os.path.isdir('pt/'):
    os.mkdir('pt/')


BATCH_SIZE = 1000
NUM_WORKERS = 8 if is_gpu_available() or is_mps_available() else 0
NUM_EPOCHS = 100

SEED = 1
seed_everything(SEED, workers=True)


NUM_HIDDEN_LAYERS = 3
NUM_NODES_HIDDEN_LAYER = 600
EMBEDDINGS_POWER = 0.5


nn_name = f'embeddings_power={EMBEDDINGS_POWER}_{NUM_HIDDEN_LAYERS}_hidden_layers_{NUM_NODES_HIDDEN_LAYER}_nodes_per_layer_{NUM_EPOCHS}_epochs'


NUM_TRADES_IN_RELATED_TRADE_HISTORY = 1


REFERENCE_FEATURES_TO_ADD = ['rating', 'incorporated_state_code', 'purpose_sub_class']    # choosing a few features from the most important features for the LightGBM model on just reference data
REFERENCE_FEATURES_TO_ADD = list(set(REFERENCE_FEATURES_TO_ADD) & set(trade_data.columns))    # make sure that all REFERNCE_FEATURES_TO_ADD are in the trade data as columns
print(f'Including the following reference features for each related trade: {REFERENCE_FEATURES_TO_ADD}')


ENCODE_REFERENCE_FEATURES = False    # boolean variable that determines whether trade history will contain categorical features that must be encoded before adding these features to the trade history


for feature in REFERENCE_FEATURES_TO_ADD:
    if feature not in FEATURES_TO_INDEX_IN_HISTORY: FEATURES_TO_INDEX_IN_HISTORY[feature] = len(FEATURES_TO_INDEX_IN_HISTORY)
    ENCODE_REFERENCE_FEATURES = True


appended_feature_prefix = 'related_last_'
get_neighbor_feature = lambda feature: lambda curr, neighbor: neighbor[feature]
APPENDED_FEATURES_FUNCTIONS = {'yield_spread': get_neighbor_feature('yield_spread'), 
                               'treasury_spread': get_neighbor_feature('ficc_treasury_spread'), 
                               'quantity': get_neighbor_feature('quantity'), 
                               'quantity_diff': lambda curr, neighbor: quantity_diff(10 ** neighbor['quantity'] - 10 ** curr['quantity']), 
                               'trade_type1': lambda curr, neighbor: TRADE_TYPE_MAPPING[neighbor['trade_type']][0], 
                               'trade_type2': lambda curr, neighbor: TRADE_TYPE_MAPPING[neighbor['trade_type']][1], 
                               'seconds_ago': lambda curr, neighbor: np.log10(1 + (curr['trade_datetime'] - neighbor['trade_datetime']).total_seconds()), 
                               'settlement_date_to_calc_date': lambda curr, neighbor: np.log10(1 + diff_in_days_two_dates(neighbor['calc_date'], neighbor['settlement_date'])), 
                               'calc_day_cat': get_neighbor_feature('calc_day_cat'), 
                               'trade_type_past_latest': lambda curr, neighbor: TRADE_TYPE_CROSS_PRODUCT_MAPPING[neighbor['trade_type'] + curr['trade_type']], 
                            #    'rating_diff': lambda curr, neighbor: RATING_TO_INT_MAPPING[curr['rating']] - RATING_TO_INT_MAPPING[neighbor['rating']]
                              }

appended_features_wo_reference_features_related_trades_groups = [[get_appended_feature_name(idx, feature, appended_feature_prefix) for feature in APPENDED_FEATURES_FUNCTIONS] 
                                                                 for idx in range(NUM_TRADES_IN_RELATED_TRADE_HISTORY)]    # insertion order of the dictionary is preserved for Python v3.7+
appended_features_wo_reference_features_related_trades = flatten(appended_features_wo_reference_features_related_trades_groups)

data_processing_functions = {'same_day': lambda curr, neighbor: int(neighbor['trade_datetime'].date() == curr['trade_datetime'].date())}    # used to track additional information about the related trades; compare date only instead of entire datetime: https://stackoverflow.com/questions/3743222/how-do-i-convert-a-datetime-to-date
reference_features_to_add_functions = {feature: get_neighbor_feature(feature) for feature in REFERENCE_FEATURES_TO_ADD}
APPENDED_FEATURES_FUNCTIONS = APPENDED_FEATURES_FUNCTIONS | data_processing_functions | reference_features_to_add_functions    # combine two dictionaries together for Python v3.9+: https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression

print(f'Related trades will have the following features: {APPENDED_FEATURES_FUNCTIONS.keys()}')

appended_features_related_trades_groups = [[get_appended_feature_name(idx, feature, appended_feature_prefix) for feature in APPENDED_FEATURES_FUNCTIONS] 
                                           for idx in range(NUM_TRADES_IN_RELATED_TRADE_HISTORY)]    # insertion order of the dictionary is preserved for Python v3.7+
appended_features_related_trades = flatten(appended_features_related_trades_groups)


# format for this dictionary: key is the name of the feature; value is a tuple where the first item is a function of the current trade and related trade, and the second item is the default value to be filled in if that value does not exist
APPENDED_FEATURES_FUNCTIONS_AND_DEFAULT_VALUES = {key: (function, DEFAULT_VALUES[key]) for key, function in APPENDED_FEATURES_FUNCTIONS.items()}


assert FEATURES_IN_HISTORY == [key for key in APPENDED_FEATURES_FUNCTIONS if key not in data_processing_functions and key not in reference_features_to_add_functions]    # insertion order of the dictionary is preserved for Python v3.7+ so this will check if the ordering of the keys are the same

epsilon = 1 / VERY_LARGE_NUMBER

RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED = 'rating_without_+-_b_nr_combined'
trade_data_flattened_trade_history[RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED] = trade_data_flattened_trade_history['rating'].transform(lambda rating: str.rstrip(rating, '+-'))    # remove + and - from right side of string
# group BBB, BB, B, and NR together since each have a very small number of trades
b_ratings = trade_data_flattened_trade_history[RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED].isin(['B', 'BB', 'BBB', 'NR'])
trade_data_flattened_trade_history.loc[b_ratings, RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED] = 'B'
print(f'Created {RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED} feature')

DAYS_TO_MATURITY_CATEGORICAL = 'days_to_maturity_categorical'
num_of_days_bins_maturity = [np.log10(days) for days in [epsilon, NUM_OF_DAYS_IN_YEAR * 2, NUM_OF_DAYS_IN_YEAR * 5, NUM_OF_DAYS_IN_YEAR * 10, VERY_LARGE_NUMBER]]    # 2 years, 5 years, 10 years; arbitrarily chosen
trade_data_flattened_trade_history[DAYS_TO_MATURITY_CATEGORICAL] = pd.cut(trade_data_flattened_trade_history['days_to_maturity'], num_of_days_bins_maturity).astype('string')
print(f'Created {DAYS_TO_MATURITY_CATEGORICAL} feature')

DAYS_TO_CALL_CATEGORICAL = 'days_to_call_categorical'
num_of_days_bins_call = [np.log10(days) for days in [epsilon, NUM_OF_DAYS_IN_YEAR * 2, NUM_OF_DAYS_IN_YEAR * 5, VERY_LARGE_NUMBER]]    # 2 years, 5 years; arbitrarily chosen
trade_data_flattened_trade_history[DAYS_TO_CALL_CATEGORICAL] = pd.cut(trade_data_flattened_trade_history['days_to_call'], num_of_days_bins_call).astype('string')
print(f'Created {DAYS_TO_CALL_CATEGORICAL} feature')

COUPON_CATEGORICAL = 'coupon_categorical'
coupon_bins = [0, 3, 4, 4.5, 5.0 + epsilon, VERY_LARGE_NUMBER]   # 0 - 2.99, 3 - 3.99, 4 - 4.49, 4.5 - 5; from discussion with a team member
trade_data_flattened_trade_history[COUPON_CATEGORICAL] = pd.cut(trade_data_flattened_trade_history['coupon'], coupon_bins, right=False).astype('string')
print(f'Created {COUPON_CATEGORICAL} feature')

COUPON_CATEGORICAL_SUDHAR = 'coupon_categorical_sudhar'
coupon_bins = [0, 3, 4, 4.5, 5, 5.25, 5.5, 6, VERY_LARGE_NUMBER]    # from Sudhar's paper: Kolm, Purushothaman. 2021. Systematic Pricing and Trading of Municipal Bonds
trade_data_flattened_trade_history[COUPON_CATEGORICAL_SUDHAR] = pd.cut(trade_data_flattened_trade_history['coupon'], coupon_bins, right=False).astype('string')
print(f'Created {COUPON_CATEGORICAL_SUDHAR} feature')

# COUPON_TOP_VALUES = 'coupon_top_values'
# trade_data_flattened_trade_history[COUPON_TOP_VALUES] = trade_data_flattened_trade_history['coupon']
# top4_coupon_values = trade_data_flattened_trade_history['coupon'].value_counts().head(4).index.tolist()    # select the top 4 coupon values based on frequency in the data, which are: 5.0, 4.0, 3.0, 2.0 comprising about 90% of the data
# trade_data_flattened_trade_history.loc[~trade_data_flattened_trade_history['coupon'].isin(top4_coupon_values), COUPON_TOP_VALUES] = -1    # arbitrary numerical value that is invalid as a coupon value
# print(f'Created {COUPON_TOP_VALUES} feature')

PURPOSE_CLASS_TOP_VALUES = 'purpose_class_top_values'
trade_data_flattened_trade_history[PURPOSE_CLASS_TOP_VALUES] = trade_data_flattened_trade_history['purpose_class']
top6_purpose_class_values = trade_data_flattened_trade_history['purpose_class'].value_counts().head(6).index.tolist()    # select the top 6 coupon values based on frequency in the data, which are: 37 (school district), 51 (various purpose), 50 (utility), 46 (tax revenue), 9 (education), 48 (transportation) comprising about 80% of the data
trade_data_flattened_trade_history.loc[~trade_data_flattened_trade_history['purpose_class'].isin(top6_purpose_class_values), PURPOSE_CLASS_TOP_VALUES] = -1    # arbitrary numerical value that is invalid as a purpose_class value
print(f'Created {PURPOSE_CLASS_TOP_VALUES} feature')

MUNI_SECURITY_TYPE_TOP_VALUES = 'muni_security_type_top_values'
trade_data_flattened_trade_history[MUNI_SECURITY_TYPE_TOP_VALUES] = trade_data_flattened_trade_history['muni_security_type']
top6_muni_security_type_values = trade_data_flattened_trade_history['muni_security_type'].value_counts().head(2).index.tolist()    # select the top 2 coupon values based on frequency in the data, which are: 8 (revenue), 5 (unlimited g.o.) comprising about 80% of the data
trade_data_flattened_trade_history.loc[~trade_data_flattened_trade_history['muni_security_type'].isin(top6_muni_security_type_values), MUNI_SECURITY_TYPE_TOP_VALUES] = -1    # arbitrary numerical value that is invalid as a purpose_class value
print(f'Created {MUNI_SECURITY_TYPE_TOP_VALUES} feature')

TRADE_DATETIME_DAY = 'trade_datetime_day'
trade_data_flattened_trade_history[TRADE_DATETIME_DAY] = trade_data_flattened_trade_history['trade_datetime'].transform(lambda datetime: datetime.date()).astype('string')    # remove timestamp from datetime
print(f'Created {TRADE_DATETIME_DAY} feature')

QUANTITY_CATEGORICAL = 'quantity_categorical'
quantity_bins = [0, 5, 6, 7, VERY_LARGE_NUMBER]    # 0 - 100k, 100k - 1m, 1m - 10m, 10m+
trade_data_flattened_trade_history[QUANTITY_CATEGORICAL] = pd.cut(trade_data_flattened_trade_history['quantity'], quantity_bins).astype('string')
print(f'Created {QUANTITY_CATEGORICAL} feature')


quantized_features = [RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, 
                      DAYS_TO_MATURITY_CATEGORICAL, 
                      DAYS_TO_CALL_CATEGORICAL, 
                      COUPON_CATEGORICAL, 
                      COUPON_CATEGORICAL_SUDHAR, 
                      PURPOSE_CLASS_TOP_VALUES, 
                      MUNI_SECURITY_TYPE_TOP_VALUES, 
                    #   COUPON_TOP_VALUES, 
                      TRADE_DATETIME_DAY, 
                      QUANTITY_CATEGORICAL]


quantity_greater_than_100k = lambda row: row['quantity'] >= np.log10(1e5)
quantity_greater_than_1m = lambda row: row['quantity'] >= np.log10(1e6)
trade_type_is_interdealer = lambda row: row['trade_type'] == 'D'


# key: name of criteria, value: (categories to match, filtering conditions)
related_trades_criterion = {# 'sudhar1': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL], []), 
                            # 'sudhar1_100k': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL], [quantity_greater_than_100k]), 
                            # 'sudhar1_1m': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL], [quantity_greater_than_1m]), 
                            # 'sudhar2': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, 'trade_type'], []), 
                            # 'sudhar2_100k': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, 'trade_type'], [quantity_greater_than_100k]), 
                            # 'sudhar2_1m': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, 'trade_type'], [quantity_greater_than_1m]), 
                            # 'sudhar3': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL], [trade_type_is_interdealer]), 
                            # 'sudhar3_100k': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL], [quantity_greater_than_100k, trade_type_is_interdealer]), 
                            # 'sudhar3_1m': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL], [quantity_greater_than_1m, trade_type_is_interdealer]), 
                            # 'sudhar4': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, MUNI_SECURITY_TYPE_TOP_VALUES], []), 
                            # 'sudhar4_100k': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, MUNI_SECURITY_TYPE_TOP_VALUES], [quantity_greater_than_100k]), 
                            # 'sudhar4_1m': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, MUNI_SECURITY_TYPE_TOP_VALUES], [quantity_greater_than_1m]), 
                            # 'sudhar5': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, TRADE_DATETIME_DAY], []), 
                            # 'sudhar5_100k': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, TRADE_DATETIME_DAY], [quantity_greater_than_100k]), 
                            # 'sudhar5_1m': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, TRADE_DATETIME_DAY], [quantity_greater_than_1m]), 
                            # 'sudhar6': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, COUPON_CATEGORICAL, TRADE_DATETIME_DAY], []), 
                            # 'sudhar6_100k': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, COUPON_CATEGORICAL, TRADE_DATETIME_DAY], [quantity_greater_than_100k]), 
                            # 'sudhar6_1m': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, COUPON_CATEGORICAL, TRADE_DATETIME_DAY], [quantity_greater_than_1m]), 
                            # 'mitas1': ([TRADE_DATETIME_DAY, 'trade_type'], []),
                            # 'mitas1_100k': ([TRADE_DATETIME_DAY, 'trade_type'], [quantity_greater_than_100k]), 
                            # 'mitas1_1m': ([TRADE_DATETIME_DAY, 'trade_type'], [quantity_greater_than_1m]), 
                            # 'desmond': (['incorporated_state_code', RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, DAYS_TO_CALL_CATEGORICAL, COUPON_CATEGORICAL, PURPOSE_CLASS_TOP_VALUES], []), 
                            # 'desmond_100k': (['incorporated_state_code', RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, DAYS_TO_CALL_CATEGORICAL, COUPON_CATEGORICAL, PURPOSE_CLASS_TOP_VALUES], [quantity_greater_than_100k]), 
                            # 'desmond_1m': (['incorporated_state_code', RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, DAYS_TO_CALL_CATEGORICAL, COUPON_CATEGORICAL, PURPOSE_CLASS_TOP_VALUES], [quantity_greater_than_1m]), 
                            # 'yellow': (['trade_type', DAYS_TO_MATURITY_CATEGORICAL, QUANTITY_CATEGORICAL, DAYS_TO_CALL_CATEGORICAL], []), 
                            # 'yellow_100k': (['trade_type', DAYS_TO_MATURITY_CATEGORICAL, QUANTITY_CATEGORICAL, DAYS_TO_CALL_CATEGORICAL], [quantity_greater_than_100k]), 
                            # 'yellow_1m': (['trade_type', DAYS_TO_MATURITY_CATEGORICAL, QUANTITY_CATEGORICAL, DAYS_TO_CALL_CATEGORICAL], [quantity_greater_than_1m]), 
                            # 'yellow_lite': (['trade_type', DAYS_TO_MATURITY_CATEGORICAL], []), 
                            # 'yellow_lite_100k': (['trade_type', DAYS_TO_MATURITY_CATEGORICAL], [quantity_greater_than_100k]), 
                            # 'yellow_lite_1m': (['trade_type', DAYS_TO_MATURITY_CATEGORICAL], [quantity_greater_than_1m]), 
                            }


# combine two dictionaries together for Python v3.9+: https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression
related_trades_criterion = related_trades_criterion | \
                           {'NONE': ([], []), 
                            # 'trade_type': (['trade_type'], []), 
                            # 'incorporated_state_code': (['incorporated_state_code'], []), 
                            # 'days_to_maturity_categorical': ([DAYS_TO_MATURITY_CATEGORICAL], []), 
                            # 'quantity_categorical': ([QUANTITY_CATEGORICAL], []), 
                            # 'coupon_categorical': ([COUPON_CATEGORICAL], []), 
                            # 'trade_datetime_day': ([TRADE_DATETIME_DAY], []), 
                            # 'rating_without_plus_minus_B_NR_combined': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED], []), 
                            # 'days_to_call': ([DAYS_TO_CALL_CATEGORICAL], []), 
                            # 'purpose_class_top_values': ([PURPOSE_CLASS_TOP_VALUES], []), 
                            # 'muni_security_type_top_values': ([MUNI_SECURITY_TYPE_TOP_VALUES], []), 
                            # '100k': ([], [quantity_greater_than_100k]), 
                            # '1m': ([], [quantity_greater_than_1m]), 
                            # 'dd': ([], [trade_type_is_interdealer]), 
                            # 'rating': (['rating'], []), 
                            # 'purpose_class': (['purpose_class'], []), 
                            # 'coupon_categorical_sudhar': ([COUPON_CATEGORICAL_SUDHAR], [])
                            }


related_trades_criterion_losses = dict()
related_trades_criterion_losses_filepath = make_data_filename('related_trades_criterion_losses')
if os.path.exists(related_trades_criterion_losses_filepath):
    print(f'Loading losses from {related_trades_criterion_losses_filepath}')
    with open(related_trades_criterion_losses_filepath, 'rb') as pickle_handle: related_trades_criterion_losses = pickle.load(pickle_handle)    # use template from https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
    print(f'Already have loss results for: {list(related_trades_criterion_losses.keys())}')


FEATURE_TO_DETECT_NO_PAST_TRADES = 'seconds_ago'    # arbitrarily chosen, but should choose a feature that does not have


NUM_TRADES_IN_TRADE_HISTORY_OPT = min(NUM_TRADES_IN_TRADE_HISTORY, 16)
print(f'NUM_TRADES_IN_TRADE_HISTORY_OPT: {NUM_TRADES_IN_TRADE_HISTORY_OPT}')


NUM_TRADES_IN_RELATED_TRADE_HISTORY_OPT = min(NUM_TRADES_IN_RELATED_TRADE_HISTORY, 32)
print(f'NUM_TRADES_IN_RELATED_TRADE_HISTORY_OPT: {NUM_TRADES_IN_RELATED_TRADE_HISTORY_OPT}')


features_in_history_without_reference_features = [feature for feature in FEATURES_TO_INDEX_IN_HISTORY if feature not in REFERENCE_FEATURES_TO_ADD]    # can be made faster if REFERENCE_FEATURES_TO_ADD were a set, but since we assume it is small, the speedup is trivial


past_trades_columns_opt, all_categorical_features_in_trade_history = get_past_trade_columns(NUM_TRADES_IN_TRADE_HISTORY_OPT, 
                                                                                            features_in_history_without_reference_features, 
                                                                                            SAME_CUSIP_PREFIX, 
                                                                                            trade_type_actual=True, 
                                                                                            trade_type_column=TRADE_TYPE_NEW_COLUMN, 
                                                                                            categorical_features_per_trade=CATEGORICAL_FEATURES_IN_HISTORY)
past_related_trades_columns_opt, all_categorical_features_in_trade_history_related = get_past_trade_columns(NUM_TRADES_IN_RELATED_TRADE_HISTORY_OPT, 
                                                                                                            FEATURES_IN_HISTORY, 
                                                                                                            appended_feature_prefix, 
                                                                                                            trade_type_actual=True, 
                                                                                                            trade_type_column=TRADE_TYPE_NEW_COLUMN, 
                                                                                                            categorical_features_per_trade=CATEGORICAL_FEATURES_IN_HISTORY + REFERENCE_FEATURES_TO_ADD)


target_trade_features = list(set(PREDICTORS_WITHOUT_LAST_TRADE_FEATURES + REFERENCE_FEATURES_TO_ADD)) + TARGET

columns_to_select_to_create_dataframe = target_trade_features + past_trade_feature_groups_flattened + appended_features_related_trades + DATA_PROCESSING_FEATURES
assert len(columns_to_select_to_create_dataframe) == len(set(columns_to_select_to_create_dataframe))    # checks that there is no intersection between the groups of features
columns_to_select_for_lightgbm_model = target_trade_features # + past_trades_columns_opt # + past_related_trades_columns_opt
assert len(columns_to_select_for_lightgbm_model) == len(set(columns_to_select_for_lightgbm_model))    # checks that there is no intersection between the groups of features

target_trade_categorical_features = list(set(CATEGORICAL_FEATURES + REFERENCE_FEATURES_TO_ADD))

categorical_features_for_lightgbm_model = target_trade_categorical_features # + all_categorical_features_in_trade_history # + all_categorical_features_in_trade_history_related
assert len(categorical_features_for_lightgbm_model) == len(set(categorical_features_for_lightgbm_model))    # checks that there is no intersection between the groups of features

print(f'Features used for LightGBM model: {columns_to_select_for_lightgbm_model}')
print(f'Categorical features used for LightGBM model: {categorical_features_for_lightgbm_model}')


df_encoded = encode_with_label_encoders(trade_data_flattened_trade_history, features_to_exclude=['trade_type']) if ENCODE_REFERENCE_FEATURES else trade_data_flattened_trade_history


if not JUST_GENERATING_DATA:   # when running LightGBM experiments, `trade_data_flattened_trade_history` is no longer needed
    del trade_data_flattened_trade_history
    gc.collect()


for name, (categories_to_match, filtering_conditions) in tqdm(related_trades_criterion.items()):
    if name not in related_trades_criterion_losses:

        print(f'{name}')
        filename = f'trade_data_flattened_trade_history_and_related_trades_{name}'
        filepath = make_data_filename(filename)
        if os.path.exists(filepath):    # check if a file exists https://www.pythontutorial.net/python-basics/python-check-if-file-exists/
            print(f'Loading dataset for {name} from pickle file {filepath}')
            trade_data_flattened_trade_history_and_related_trades = pd.read_pickle(filepath)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html

        if JUST_GENERATING_DATA:

            print(f'Creating dataset for {name} and saving it to {filepath}')
            start_time_when_creating_dataset = time.time()    # used to measure elapsed time of below procedure: https://stackoverflow.com/questions/7370801/how-do-i-measure-elapsed-time-in-python
            trade_data_flattened_trade_history_and_related_trades = append_recent_trade_data(trade_data_flattened_trade_history, 
                                                                                             NUM_TRADES_IN_RELATED_TRADE_HISTORY, 
                                                                                             APPENDED_FEATURES_FUNCTIONS_AND_DEFAULT_VALUES, 
                                                                                             feature_prefix=appended_feature_prefix, 
                                                                                             categories=categories_to_match, 
                                                                                             filtering_conditions=filtering_conditions, 
                                                                                             return_df=True, 
                                                                                             multiprocessing=True, 
                                                                                             df_for_related_trades=df_encoded).drop(columns=quantized_features)    # drop the quantized features from the final dataframe
            print(f'Took {datetime.timedelta(seconds=time.time() - start_time_when_creating_dataset)} to create dataset for {name}')    # use `timedelta` to convert seconds to minutes: https://stackoverflow.com/questions/775049/how-do-i-convert-seconds-to-hours-minutes-and-seconds
            start_time_when_saving_dataset = time.time()    # used to measure elapsed time of below procedure: https://stackoverflow.com/questions/7370801/how-do-i-measure-elapsed-time-in-python
            trade_data_flattened_trade_history_and_related_trades.to_pickle(filepath, protocol=4)    # protocol 4 allows for use in the VM: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
            print(f'Took {datetime.timedelta(seconds=time.time() - start_time_when_saving_dataset)} to save dataset for {name} to {filepath}')    # use `timedelta` to convert seconds to minutes: https://stackoverflow.com/questions/775049/how-do-i-convert-seconds-to-hours-minutes-and-seconds

            for past_trade_idx in (0, 1, 15, 31):    # range(2):
                if past_trade_idx < NUM_TRADES_IN_RELATED_TRADE_HISTORY:
                    feature_name = get_appended_feature_name(past_trade_idx, FEATURE_TO_DETECT_NO_PAST_TRADES, appended_feature_prefix)
                    num_trades = (trade_data_flattened_trade_history_and_related_trades[feature_name] == DEFAULT_VALUES[FEATURE_TO_DETECT_NO_PAST_TRADES]).sum()
                    print(f'Number of trades with fewer than {past_trade_idx + 1} past related trades: {num_trades}. Percentage of total trades: {round(num_trades / len(trade_data) * 100, 3)} %')

        else:

            print('Decoding the reference features.')
            for feature in REFERENCE_FEATURES_TO_ADD:
                encoder = label_encoders[feature]
                for past_trade_idx in range(NUM_TRADES_IN_RELATED_TRADE_HISTORY):
                    feature_name = get_appended_feature_name(past_trade_idx, feature, appended_feature_prefix)
                    trade_data_flattened_trade_history_and_related_trades[feature_name] = encoder.inverse_transform(trade_data_flattened_trade_history_and_related_trades[feature_name].to_numpy(dtype=int))    # inverse transform the encoded categorical feature column; must set to dtype=int since label encoder encodes to integers
                    trade_data_flattened_trade_history_and_related_trades[feature_name] = trade_data_flattened_trade_history_and_related_trades[feature_name].astype('category')    # change dtype to `categorical` to use in LightGBM model
                if trade_data_flattened_trade_history_and_related_trades[feature].dtype.name != 'category': trade_data_flattened_trade_history_and_related_trades[feature] = trade_data_flattened_trade_history_and_related_trades[feature].astype('category')    # check dtype of a column: https://stackoverflow.com/questions/26924904/check-if-dataframe-column-is-categorical
            
            # convert trade_type1 and trade_type2 to trade_type with S, P, D for same CUSIP trades
            trade_data_predictors_history_related_trades_actual_trade_type, old_trade_type_columns, _ = convert_trade_type_encoding_to_actual(trade_data_flattened_trade_history_and_related_trades[columns_to_select_to_create_dataframe], 
                                                                                                                                              NUM_TRADES_IN_TRADE_HISTORY, 
                                                                                                                                              TRADE_TYPE_NEW_COLUMN, 
                                                                                                                                              SAME_CUSIP_PREFIX)
            # convert trade_type1 and trade_type2 to trade_type with S, P, D for related trades
            trade_data_predictors_history_related_trades_actual_trade_type, old_trade_type_columns_related, _ = convert_trade_type_encoding_to_actual(trade_data_predictors_history_related_trades_actual_trade_type, 
                                                                                                                                                      NUM_TRADES_IN_RELATED_TRADE_HISTORY, 
                                                                                                                                                      TRADE_TYPE_NEW_COLUMN, 
                                                                                                                                                      appended_feature_prefix)

            train_data_predictors_history_related_trades_actual_trade_type, \
                test_data_predictors_history_related_trades_actual_trade_type = get_train_test_data_trade_datetime(trade_data_predictors_history_related_trades_actual_trade_type, DATE_TO_SPLIT)
            del trade_data_predictors_history_related_trades_actual_trade_type
            gc.collect()
            assert len(train_data_predictors_history_related_trades_actual_trade_type) != 0 and len(test_data_predictors_history_related_trades_actual_trade_type) != 0, 'Either train or test data is empty. Consider checking how the train test split is being performed.'

            print(f'Training the LightGBM model for {name}')
            _, lgb_losses = train_lightgbm_model(train_data_predictors_history_related_trades_actual_trade_type[columns_to_select_for_lightgbm_model], 
                                                 test_data_predictors_history_related_trades_actual_trade_type[columns_to_select_for_lightgbm_model], 
                                                 categorical_features_for_lightgbm_model, 
                                                 wandb_project='mitas_trade_history')
            related_trades_criterion_losses[name] = lgb_losses['Train'][0], lgb_losses['Test'][0]
            with open(related_trades_criterion_losses_filepath, 'wb') as pickle_handle: pickle.dump(related_trades_criterion_losses, pickle_handle, protocol=4)    # use template from https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object


if not JUST_GENERATING_DATA:
    for name, (train_loss, test_loss) in related_trades_criterion_losses.items():
        print(f'{name}\t\tTrain error: {train_loss}\tTest error: {test_loss}')
    related_trades_criterion_ascending_order_of_test_loss = sorted(related_trades_criterion_losses, key=lambda name: related_trades_criterion_losses.get(name)[1])    # sort by minimum test error (which is represented by index 1)
    related_trades_criterion_opt = related_trades_criterion_ascending_order_of_test_loss[0]    # optimal name is the one with the minimum test error
