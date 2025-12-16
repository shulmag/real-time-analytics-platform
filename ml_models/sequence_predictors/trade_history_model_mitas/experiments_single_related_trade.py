from collections import defaultdict
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from datetime import datetime

# import lightgbm as lgb

from pytorch_lightning import seed_everything

from ficc.utils.auxiliary_variables import VERY_LARGE_NUMBER, \
                                           IDENTIFIERS, \
                                           CATEGORICAL_FEATURES, \
                                           NON_CAT_FEATURES, \
                                           BINARY, \
                                           TRADE_HISTORY, \
                                           NUM_OF_DAYS_IN_YEAR
from ficc.utils.auxiliary_functions import flatten, \
                                           list_to_index_dict
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.trade_dict_to_list import TRADE_TYPE_MAPPING, \
                                          FEATURES_TO_INDEX_IN_HISTORY, \
                                          quantity_diff, \
                                          TRADE_TYPE_CROSS_PRODUCT_MAPPING
from ficc.utils.related_trade import append_recent_trade_data, get_appended_feature_name

import sys
sys.path.insert(0,'../')

from trade_history_model_mitas.data_prep import get_past_trade_columns, \
                                                feature_group_as_single_feature, \
                                                limit_history_to_k_trades, \
                                                combine_two_histories_sorted_by_seconds_ago, \
                                                remove_feature_from_trade_history, \
                                                convert_trade_type_encoding_to_actual, \
                                                add_reference_data_to_trade_history, \
                                                embed_with_arrays, \
                                                add_single_trade_from_history_as_reference_features
from trade_history_model_mitas.models import MultipleRecurrentL1Loss, \
                                             NNL1LossEmbeddingsWithMultipleRecurrence

from yield_spread_model_mitas.data_prep import FEATURES_AND_NAN_REPLACEMENT_VALUES, \
                                               ADDITIONAL_CATEGORICAL_FEATURES, \
                                               get_datestring_from_filename, \
                                               remove_rows_with_feature_value, \
                                               replace_rating_with_standalone_rating, \
                                               add_past_trades_info, \
                                               reverse_order_of_trade_history, \
                                               check_additional_features, \
                                               replace_nan_for_features, \
                                               encode_and_get_encoders, \
                                               encode_with_encoders
from yield_spread_model_mitas.models import single_feature_model, \
                                            RecurrentL1Loss, \
                                            NNL1LossEmbeddings, \
                                            NNL1LossEmbeddingsWithRecurrence
from yield_spread_model_mitas.train import get_train_test_data_trade_datetime, \
                                           is_gpu_available, \
                                           is_mps_available, \
                                           train, \
                                           get_all_losses
from yield_spread_model_mitas.tree_models import train_lightgbm_model, \
                                                 get_all_losses_for_single_dataset

from rating_model_mitas.data_prep import read_processed_file_pickle, \
                                         remove_fields_with_single_unique_value, \
                                         remove_rows_with_nan_value
from rating_model_mitas.train import load_model


# default value of 0 is chosen for settlement_date_to_calc_date because we exclude bonds that have a calc date that is fewer than 400 days into the future, and so a true value of settlement date to calc date will never be close to 0
DEFAULT_VALUES_NONZERO_PADDING = {'quantity_diff': np.log10(VERY_LARGE_NUMBER),    # model should learn that a quantity diff that is very large, i.e., current quantity is much smaller than quantity being compared to, means that the trade is less meaningful to use for pricing since the trades are very different. Could also be -np.log10(VERY_LARGE_NUMBER) for the same reason
                                  'seconds_ago': np.log10(VERY_LARGE_NUMBER)}    # model should learn that if the trade being compared to is far back in the past, then it is less meaningful to pricing the current trade
DEFAULT_VALUES_NONZERO_PADDING = defaultdict(int, DEFAULT_VALUES_NONZERO_PADDING)    # constructing a defaultdict from a dict: https://stackoverflow.com/questions/7539115/how-to-construct-a-defaultdict-from-a-dictionary

DEFAULT_VALUES_ZERO_PADDING = defaultdict(int)

DEFAULT_VALUES = DEFAULT_VALUES_ZERO_PADDING


NUM_TRADES_IN_TRADE_HISTORY = 32    # maximum number of past trades in the history


TARGET = ['yield_spread']


DATA_PROCESSING_FEATURES = ['trade_datetime',    # used to split the data into training and test sets
                            'settlement_date',    # used (in conjunction with calc_date) to create the settlement_date_to_calc_date feature in past trades
                            'calc_date',    # used (in conjunction with settlement_date) to create the settlement_date_to_calc_date feature in past trades
                            'calc_day_cat',    # added in the past trades
                            # 'coupon_type'    # used to group related trades; currently commented out since there is only a single value of 8 present in the data
                           ]


processed_file_pickle = '../../../../ficc/ml_models/sequence_predictors/data/processed_data_ficc_ycl_long_history_zero_padding_2021-12-31-23-59.pkl'
processed_file_pickle_datestring = get_datestring_from_filename(processed_file_pickle)
trade_data = read_processed_file_pickle(processed_file_pickle)


trade_data = trade_data[(trade_data.days_to_call == 0) | (trade_data.days_to_call > np.log10(400))]
trade_data = trade_data[(trade_data.days_to_refund == 0) | (trade_data.days_to_refund > np.log10(400))]
trade_data = trade_data[trade_data.days_to_maturity < np.log10(30000)]
trade_data = trade_data[trade_data.sinking == False]
trade_data = trade_data[trade_data.incorporated_state_code != 'VI']
trade_data = trade_data[trade_data.incorporated_state_code != 'GU']
trade_data = trade_data[(trade_data.coupon_type == 8)]
trade_data = trade_data[trade_data.is_called == False]

# restructured bonds and high chance of default bonds are removed
trade_data = remove_rows_with_feature_value(trade_data, 'purpose_sub_class', [6, 20, 21, 22, 44, 57, 90, 106])
# pre-refunded bonds and partially refunded bonds are removed
trade_data = remove_rows_with_feature_value(trade_data, 'called_redemption_type', [18, 19])


trade_data = replace_rating_with_standalone_rating(trade_data)


ADDITIONAL_CATEGORICAL_FEATURES = check_additional_features(trade_data, ADDITIONAL_CATEGORICAL_FEATURES)

all_features_set = set(trade_data.columns)
BINARY = list(set(BINARY).intersection(all_features_set))
CATEGORICAL_FEATURES = list((set(CATEGORICAL_FEATURES) | set(ADDITIONAL_CATEGORICAL_FEATURES)).intersection(all_features_set))
NON_CAT_FEATURES = list(set(NON_CAT_FEATURES).intersection(all_features_set))

trade_data = trade_data[IDENTIFIERS + 
                        BINARY + 
                        CATEGORICAL_FEATURES + 
                        NON_CAT_FEATURES + 
                        DATA_PROCESSING_FEATURES + 
                        TRADE_HISTORY + 
                        TARGET]

trade_data, _ = replace_nan_for_features(trade_data, FEATURES_AND_NAN_REPLACEMENT_VALUES, verbose=True)
trade_data = remove_fields_with_single_unique_value(trade_data)

all_features_set = set(trade_data.columns)
BINARY = list(set(BINARY).intersection(all_features_set))
CATEGORICAL_FEATURES = list(set(CATEGORICAL_FEATURES).intersection(all_features_set))
NON_CAT_FEATURES = list(set(NON_CAT_FEATURES).intersection(all_features_set))
PREDICTORS = BINARY + CATEGORICAL_FEATURES + NON_CAT_FEATURES

trade_data = remove_rows_with_nan_value(trade_data)


# sort by trade_datetime since order can be changed when reading pickle file into m1 since it loads by chunks
trade_data = trade_data.sort_values(by='trade_datetime', ascending=False)


oldest_trade_datetime = trade_data['trade_datetime'].iloc[-1]
newest_trade_datetime = trade_data['trade_datetime'].iloc[0]

print(f'Oldest trade datetime: {oldest_trade_datetime}.\
    Newest trade datetime: {newest_trade_datetime}.\
    Gap: {newest_trade_datetime - oldest_trade_datetime}')
print(f'Total number of trades: {len(trade_data)}')


DATE_TO_SPLIT = datetime(2021, 12, 1)    # December 1 2021


train_data, test_data = get_train_test_data_trade_datetime(trade_data, DATE_TO_SPLIT)
print(f'Number of trades for training: {len(train_data)}.\
    Number of trades for testing: {len(test_data)}')
assert len(train_data) != 0 and len(test_data) != 0, 'Either train or test data is empty. Consider checking how the train test split is being performed.'
train_data_with_trade_history = train_data.drop(columns=DATA_PROCESSING_FEATURES + IDENTIFIERS)
test_data_with_trade_history = test_data.drop(columns=DATA_PROCESSING_FEATURES + IDENTIFIERS)
train_data_only_reference = train_data.drop(columns=DATA_PROCESSING_FEATURES + TRADE_HISTORY + IDENTIFIERS)
test_data_only_reference = test_data.drop(columns=DATA_PROCESSING_FEATURES + TRADE_HISTORY + IDENTIFIERS)


# TODO: remove this when we use the updated data file
FEATURES_TO_INDEX_IN_HISTORY = {'yield_spread': 0,
                                'quantity': 1,
                                'quantity_diff': 2,
                                'trade_type1': 3,
                                'trade_type2': 4,
                                'seconds_ago': 5,
                                'settlement_date_to_calc_date': 6}


trade_data_flattened_trade_history, \
    additional_binary_features_from_past_trades, \
    additional_noncat_features_from_past_trades, \
    past_trade_feature_groups = add_past_trades_info(trade_data, NUM_TRADES_IN_TRADE_HISTORY - 1, FEATURES_TO_INDEX_IN_HISTORY)
past_trade_feature_groups_flattened = flatten(past_trade_feature_groups)
print(f'Each of the past trades are in the following feature groups: {past_trade_feature_groups}')


trade_data_only_history = trade_data_flattened_trade_history[past_trade_feature_groups_flattened + DATA_PROCESSING_FEATURES + TARGET]
train_data_only_history, test_data_only_history = get_train_test_data_trade_datetime(trade_data_only_history, DATE_TO_SPLIT)
assert len(train_data_only_history) != 0 and len(test_data_only_history) != 0, 'Either train or test data is empty. Consider checking how the train test split is being performed.'
train_data_only_history = train_data_only_history.drop(columns=DATA_PROCESSING_FEATURES)
test_data_only_history = test_data_only_history.drop(columns=DATA_PROCESSING_FEATURES)
trade_data_only_history = trade_data_only_history.drop(columns=DATA_PROCESSING_FEATURES)


# TRADE_TYPE_NEW_COLUMN = 'trade_type'


# trade_data_only_history_actual_trade_type = trade_data_flattened_trade_history[past_trade_feature_groups_flattened + 
#                                                                                DATA_PROCESSING_FEATURES + 
#                                                                                TARGET]
# trade_data_only_history_actual_trade_type, old_trade_type_columns, _ = convert_trade_type_encoding_to_actual(trade_data_only_history_actual_trade_type, 
#                                                                                                              NUM_TRADES_IN_TRADE_HISTORY, 
#                                                                                                              TRADE_TYPE_NEW_COLUMN, 
#                                                                                                              'last_')

# train_data_only_history_actual_trade_type, \
#     test_data_only_history_actual_trade_type = get_train_test_data_trade_datetime(trade_data_only_history_actual_trade_type, DATE_TO_SPLIT)
# assert len(train_data_only_history_actual_trade_type) != 0 and len(test_data_only_history_actual_trade_type) != 0, 'Either train or test data is empty. Consider checking how the train test split is being performed.'
# columns_to_remove = DATA_PROCESSING_FEATURES + old_trade_type_columns
# train_data_only_history_actual_trade_type = train_data_only_history_actual_trade_type.drop(columns=columns_to_remove)
# test_data_only_history_actual_trade_type = test_data_only_history_actual_trade_type.drop(columns=columns_to_remove)
# trade_data_only_history_actual_trade_type = trade_data_only_history_actual_trade_type.drop(columns=columns_to_remove)


make_data_filename = lambda name: f'data/{name}.pkl'    # used to create a filename to save the PyTorch model parameters
if not os.path.isdir('data/'):
    os.mkdir('data/')


train_data_only_reference_encoded, label_encoders = encode_and_get_encoders(train_data_only_reference, BINARY, CATEGORICAL_FEATURES)
label_encoders_filepath = make_data_filename('label_encoders')
with open(label_encoders_filepath, 'wb') as pickle_handle: pickle.dump(label_encoders, pickle_handle, protocol=4)    # protocol 4 allows for use in the VM; use template from https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
encode_with_label_encoders = lambda df, features_to_exclude=[]: encode_with_encoders(df, label_encoders, features_to_exclude)
test_data_only_reference_encoded = encode_with_label_encoders(test_data_only_reference)
train_data_with_trade_history_encoded = encode_with_encoders(train_data_with_trade_history, label_encoders)
test_data_with_trade_history_encoded = encode_with_encoders(test_data_with_trade_history, label_encoders)


make_filename = lambda name: f'pt/{name}.pt'    # used to create a filename to save the PyTorch model parameters
if not os.path.isdir('pt/'):
    os.mkdir('pt/')


results = dict()    # store the results of this experiment
results_filepath = make_data_filename('results')
if os.path.exists(results_filepath):    # check if a file exists https://www.pythontutorial.net/python-basics/python-check-if-file-exists/
    print(f'Loading results so far from pickle file {results_filepath}')
    with open(results_filepath, 'rb') as pickle_handle: results = pickle.load(pickle_handle)    # use template from https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object


BATCH_SIZE = 1000
NUM_WORKERS = 0    # 8 if is_gpu_available() or is_mps_available() else 0
NUM_EPOCHS = 100

SEED = 1
seed_everything(SEED, workers=True)


NUM_HIDDEN_LAYERS = 3
NUM_NODES_HIDDEN_LAYER = 600
EMBEDDINGS_POWER = 0.5


nn_name = f'embeddings_power={EMBEDDINGS_POWER}_{NUM_HIDDEN_LAYERS}_hidden_layers_{NUM_NODES_HIDDEN_LAYER}_nodes_per_layer_{NUM_EPOCHS}_epochs'


def train_and_store_model_loss(model, experiment_name):
    model_name = experiment_name + '_' + nn_name
    _, test_loss = train(model, 
                         NUM_EPOCHS, 
                         model_filename=make_filename(model_name), 
                         save=True, 
                         print_losses_before_training=False,    # setting this to True may cause the kernel to crash
                         print_losses_after_training=False,    # setting this to True may cause the kernel to crash
                         wandb_logging_name=model_name)
    results[experiment_name] = test_loss
    with open(results_filepath, 'wb') as pickle_handle: pickle.dump(results, pickle_handle, protocol=4)    # protocol 4 allows for use in the VM; use template from https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object


experiment_name = 'reference_data'
if experiment_name not in results:
    model = NNL1LossEmbeddings(BATCH_SIZE, 
                               NUM_WORKERS, 
                               train_data_only_reference_encoded, 
                               test_data_only_reference_encoded, 
                               label_encoders, 
                               CATEGORICAL_FEATURES, 
                               NUM_NODES_HIDDEN_LAYER, 
                               NUM_HIDDEN_LAYERS, 
                               power=EMBEDDINGS_POWER)
    train_and_store_model_loss(model, experiment_name)


RECURRENT_ARCHITECTURE = 'lstm'
NUM_RECURRENT_LAYERS = 3
RECURRENT_HIDDEN_SIZE = 64


experiment_name = 'reference_data_same_cusip_rnn'
if experiment_name not in results:
    model = NNL1LossEmbeddingsWithRecurrence(BATCH_SIZE, 
                                             NUM_WORKERS, 
                                             train_data_with_trade_history_encoded, 
                                             test_data_with_trade_history_encoded, 
                                             label_encoders, 
                                             CATEGORICAL_FEATURES, 
                                             NUM_NODES_HIDDEN_LAYER, 
                                             NUM_HIDDEN_LAYERS, 
                                             NUM_RECURRENT_LAYERS, 
                                             RECURRENT_HIDDEN_SIZE, 
                                             recurrent_architecture=RECURRENT_ARCHITECTURE, 
                                             power=EMBEDDINGS_POWER)
    train_and_store_model_loss(model, experiment_name)


NUM_TRADES_IN_RELATED_TRADE_HISTORY = 64


REFERENCE_FEATURES_TO_ADD = ['rating', 'incorporated_state_code', 'purpose_sub_class']    # choosing a few features from the most important features for the LightGBM model on just reference data


ENCODE_REFERENCE_FEATURES = False    # boolean variable that determines whether trade history will contain categorical features that must be encoded before adding these features to the trade history


for feature in REFERENCE_FEATURES_TO_ADD:
    if feature not in FEATURES_TO_INDEX_IN_HISTORY: FEATURES_TO_INDEX_IN_HISTORY[feature] = len(FEATURES_TO_INDEX_IN_HISTORY)
    ENCODE_REFERENCE_FEATURES = True


appended_feature_prefix = 'related_last_'
get_neighbor_feature = lambda feature: lambda curr, neighbor: neighbor[feature]
APPENDED_FEATURES_FUNCTIONS = {'yield_spread': get_neighbor_feature('yield_spread'), 
                               'quantity': get_neighbor_feature('quantity'), 
                               'quantity_diff': lambda curr, neighbor: quantity_diff(10 ** neighbor['quantity'] - 10 ** curr['quantity']), 
                               'trade_type1': lambda curr, neighbor: TRADE_TYPE_MAPPING[neighbor['trade_type']][0], 
                               'trade_type2': lambda curr, neighbor: TRADE_TYPE_MAPPING[neighbor['trade_type']][1], 
                               'seconds_ago': lambda curr, neighbor: np.log10(1 + (curr['trade_datetime'] - neighbor['trade_datetime']).total_seconds()), 
                               'settlement_date_to_calc_date': lambda curr, neighbor: np.log10(1 + diff_in_days_two_dates(neighbor['calc_date'], neighbor['settlement_date'])), 
                            #    'calc_day_cat': get_neighbor_feature('calc_day_cat'), 
                            #    'trade_type_past_latest': lambda curr, neighbor: TRADE_TYPE_CROSS_PRODUCT_MAPPING[neighbor['trade_type'] + curr['trade_type']]
                              }

appended_features_wo_reference_features_related_trades_groups = [[get_appended_feature_name(idx, feature, appended_feature_prefix) for feature in APPENDED_FEATURES_FUNCTIONS] 
                                                                 for idx in range(NUM_TRADES_IN_RELATED_TRADE_HISTORY)]    # insertion order of the dictionary is preserved for Python v3.7+
appended_features_wo_reference_features_related_trades = flatten(appended_features_wo_reference_features_related_trades_groups)

data_processing_functions = {'same_day': lambda curr, neighbor: int(neighbor['trade_datetime'].date() == curr['trade_datetime'].date())}    # used to track additional information about the related trades; compare date only instead of entire datetime: https://stackoverflow.com/questions/3743222/how-do-i-convert-a-datetime-to-date
reference_features_to_add_functions = {feature: get_neighbor_feature(feature) for feature in REFERENCE_FEATURES_TO_ADD}
APPENDED_FEATURES_FUNCTIONS = APPENDED_FEATURES_FUNCTIONS | data_processing_functions | reference_features_to_add_functions    # combine two dictionaries together for Python v3.9+: https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression

appended_features_related_trades_groups = [[get_appended_feature_name(idx, feature, appended_feature_prefix) for feature in APPENDED_FEATURES_FUNCTIONS] 
                                           for idx in range(NUM_TRADES_IN_RELATED_TRADE_HISTORY)]    # insertion order of the dictionary is preserved for Python v3.7+
appended_features_related_trades = flatten(appended_features_related_trades_groups)


# format for this dictionary: key is the name of the feature; value is a tuple where the first item is a function of the current trade and related trade, and the second item is the default value to be filled in if that value does not exist
APPENDED_FEATURES_FUNCTIONS_AND_DEFAULT_VALUES = {key: (function, DEFAULT_VALUES[key]) for key, function in APPENDED_FEATURES_FUNCTIONS.items()}


appended_features_functions_keys_wo_data_processing_features = [key for key in list(APPENDED_FEATURES_FUNCTIONS.keys()) if key not in data_processing_functions]
assert list(FEATURES_TO_INDEX_IN_HISTORY.keys()) == appended_features_functions_keys_wo_data_processing_features    # insertion order of the dictionary is preserved for Python v3.7+ so this will check if the ordering of the keys are the same


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


related_trades_criterion = {'trade_type': (['trade_type'], [])}


trade_data_flattened_trade_history_and_related_trades = dict()


df_encoded = encode_with_label_encoders(trade_data_flattened_trade_history, features_to_exclude=['trade_type']) if ENCODE_REFERENCE_FEATURES else trade_data_flattened_trade_history
for name, (categories_to_match, filtering_conditions) in tqdm(related_trades_criterion.items()):
    filename = f'trade_data_flattened_trade_history_and_related_trades_{name}'
    filepath = make_data_filename(filename)
    if os.path.exists(filepath):    # check if a file exists https://www.pythontutorial.net/python-basics/python-check-if-file-exists/
        print(f'Loading dataset for {name} from pickle file {filepath}')
        trade_data_flattened_trade_history_and_related_trades[name] = pd.read_pickle(filepath)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
    elif name not in trade_data_flattened_trade_history_and_related_trades:
        print(f'Creating dataset for {name} and saving it to {filepath}')
        trade_data_flattened_trade_history_and_related_trades[name] = append_recent_trade_data(trade_data_flattened_trade_history, 
                                                                                               NUM_TRADES_IN_RELATED_TRADE_HISTORY, 
                                                                                               APPENDED_FEATURES_FUNCTIONS_AND_DEFAULT_VALUES, 
                                                                                               feature_prefix=appended_feature_prefix, 
                                                                                               categories=categories_to_match, 
                                                                                               filtering_conditions=filtering_conditions, 
                                                                                               return_df=True, 
                                                                                               multiprocessing=True, 
                                                                                               df_for_related_trades=df_encoded).drop(columns=quantized_features)    # drop the quantized features from the final dataframe
        trade_data_flattened_trade_history_and_related_trades[name].to_pickle(filepath, protocol=4)    # protocol 4 allows for use in the VM: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
trade_data_flattened_trade_history = trade_data_flattened_trade_history.drop(columns=quantized_features)    # drop the quantized features from the final dataframe


FEATURE_TO_DETECT_NO_PAST_TRADES = 'seconds_ago'    # arbitrarily chosen, but should choose a feature that does not have 
for name, df in trade_data_flattened_trade_history_and_related_trades.items():
    print(f'{name}')
    for past_trade_idx in (0, 1, 15, 31):    # range(2):
        feature_name = get_appended_feature_name(past_trade_idx, FEATURE_TO_DETECT_NO_PAST_TRADES, appended_feature_prefix)
        num_trades = (df[feature_name] == DEFAULT_VALUES[FEATURE_TO_DETECT_NO_PAST_TRADES]).sum()
        print(f'Number of trades with fewer than {past_trade_idx + 1} past related trades: {num_trades}. Percentage of total trades: {round(num_trades / len(trade_data) * 100, 3)} %')


if ENCODE_REFERENCE_FEATURES:
    print('Decoding the reference features.')
    for df in tqdm(trade_data_flattened_trade_history_and_related_trades.values()):
        for feature in REFERENCE_FEATURES_TO_ADD:
            encoder = label_encoders[feature]
            for past_trade_idx in range(NUM_TRADES_IN_RELATED_TRADE_HISTORY):
                feature_name = get_appended_feature_name(past_trade_idx, feature, appended_feature_prefix)
                df[feature_name] = encoder.inverse_transform(df[feature_name].to_numpy(dtype=int))    # inverse transform the encoded categorical feature column; must set to dtype=int since label encoder encodes to integers
                df[feature_name] = df[feature_name].astype('category')    # change dtype to `categorical` to use in LightGBM model


for df in tqdm(trade_data_flattened_trade_history_and_related_trades.values()):
    for feature in REFERENCE_FEATURES_TO_ADD:
        if df[feature].dtype.name != 'category': df[feature] = df[feature].astype('category')    # check dtype of a column: https://stackoverflow.com/questions/26924904/check-if-dataframe-column-is-categorical


NUM_TRADES_IN_TRADE_HISTORY_OPT = 16


NUM_TRADES_IN_RELATED_TRADE_HISTORY_OPT = 32


only_history_and_related_trades = dict()    # key: name, value: (train data, test data)
# only_history_and_related_trades_actual_trade_type = dict()    # key: name, value: (train data, test data)


columns_to_select = REFERENCE_FEATURES_TO_ADD + past_trade_feature_groups_flattened + appended_features_related_trades + DATA_PROCESSING_FEATURES + TARGET
for name, df in tqdm(trade_data_flattened_trade_history_and_related_trades.items()):
    if name not in only_history_and_related_trades:    # and name not in only_history_and_related_trades_actual_trade_type:
        trade_data_only_history_and_related_trades = df[columns_to_select]
        train_data_only_history_and_related_trades, \
            test_data_only_history_and_related_trades = get_train_test_data_trade_datetime(trade_data_only_history_and_related_trades, DATE_TO_SPLIT)
        assert len(train_data_only_history_and_related_trades) != 0 and len(test_data_only_history_and_related_trades) != 0, 'Either train or test data is empty. Consider checking how the train test split is being performed.'
        train_data_only_history_and_related_trades = train_data_only_history_and_related_trades.drop(columns=DATA_PROCESSING_FEATURES)
        test_data_only_history_and_related_trades = test_data_only_history_and_related_trades.drop(columns=DATA_PROCESSING_FEATURES)
        only_history_and_related_trades[name] = train_data_only_history_and_related_trades, test_data_only_history_and_related_trades

        # trade_data_only_history_and_related_trades_actual_trade_type = df[columns_to_select]
        # trade_data_only_history_and_related_trades_actual_trade_type, old_trade_type_columns, _ = convert_trade_type_encoding_to_actual(trade_data_only_history_and_related_trades_actual_trade_type, 
        #                                                                                                                                 NUM_TRADES_IN_TRADE_HISTORY, 
        #                                                                                                                                 TRADE_TYPE_NEW_COLUMN, 
        #                                                                                                                                 'last_')
        # trade_data_only_history_and_related_trades_actual_trade_type, old_trade_type_columns_related, _ = convert_trade_type_encoding_to_actual(trade_data_only_history_and_related_trades_actual_trade_type, 
        #                                                                                                                                         NUM_TRADES_IN_RELATED_TRADE_HISTORY, 
        #                                                                                                                                         TRADE_TYPE_NEW_COLUMN, 
        #                                                                                                                                         appended_feature_prefix)

        # train_data_only_history_and_related_trades_actual_trade_type, \
        #     test_data_only_history_and_related_trades_actual_trade_type = get_train_test_data_trade_datetime(trade_data_only_history_and_related_trades_actual_trade_type, DATE_TO_SPLIT)
        # assert len(train_data_only_history_and_related_trades_actual_trade_type) != 0 and len(test_data_only_history_and_related_trades_actual_trade_type) != 0, 'Either train or test data is empty. Consider checking how the train test split is being performed.'
        # columns_to_remove = DATA_PROCESSING_FEATURES + old_trade_type_columns + old_trade_type_columns_related
        # train_data_only_history_and_related_trades_actual_trade_type = train_data_only_history_and_related_trades_actual_trade_type.drop(columns=columns_to_remove)
        # test_data_only_history_and_related_trades_actual_trade_type = test_data_only_history_and_related_trades_actual_trade_type.drop(columns=columns_to_remove)
        # only_history_and_related_trades_actual_trade_type[name] = (train_data_only_history_and_related_trades_actual_trade_type, \
        #                                                            test_data_only_history_and_related_trades_actual_trade_type)


related_trades_criterion_opt = list(related_trades_criterion.keys())[0]


trade_data_flattened_trade_history_and_related_trades = trade_data_flattened_trade_history_and_related_trades[related_trades_criterion_opt]
train_data_only_history_and_related_trades, test_data_only_history_and_related_trades = only_history_and_related_trades[related_trades_criterion_opt]
# train_data_only_history_and_related_trades_actual_trade_type, \
#     test_data_only_history_and_related_trades_actual_trade_type = only_history_and_related_trades_actual_trade_type[related_trades_criterion_opt]


embeddings_name = f'embeddings_{NUM_EPOCHS}_epochs'


embeddings_arrays = dict()


embeddings_arrays_filepath = make_data_filename('embeddings_arrays')
if os.path.exists(embeddings_arrays_filepath):    # check if a file exists https://www.pythontutorial.net/python-basics/python-check-if-file-exists/
    print(f'Loading embeddings_arrays from {embeddings_arrays_filepath}')
    with open(embeddings_arrays_filepath, 'rb') as pickle_handle: embeddings_arrays = pickle.load(pickle_handle)    # use template from https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
else:
    for feature in REFERENCE_FEATURES_TO_ADD:
        if feature not in embeddings_arrays:
            print(f'Creating embeddings for feature: {feature}')
            model, _ = train(NNL1LossEmbeddings(BATCH_SIZE, 
                                                NUM_WORKERS, 
                                                train_data_only_reference_encoded[[feature] + TARGET],    # just rating and labels
                                                test_data_only_reference_encoded[[feature] + TARGET],    # just rating and labels
                                                label_encoders, 
                                                [feature], 
                                                power=EMBEDDINGS_POWER), 
                             NUM_EPOCHS, 
                             model_filename=make_filename(f'{embeddings_name}_{feature}'), 
                             save=False, 
                             print_losses_before_training=False,    # setting this to True may cause the kernel to crash
                             print_losses_after_training=False,    # setting this to True may cause the kernel to crash
                             wandb_logging_name=embeddings_name, 
                             wandb_project='mitas_trade_history')
            embedding = list(model.embeddings)[0]    # get the embedding from the model; since there is only one feature, we select it
            embeddings_arrays[feature] = embedding.weight.detach().numpy()    # embedding is a matrix where each row corresponds to a different possible value; convert the tensor to numpy: https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array
    with open(embeddings_arrays_filepath, 'wb') as pickle_handle: pickle.dump(embeddings_arrays, pickle_handle, protocol=4)    # protocol 4 allows for use in the VM; use template from https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
print(f'Keys in embeddings_arrays: {list(embeddings_arrays.keys())}')


TRADE_HISTORY_RELATED = ['trade_history_related']


COMBINED_TRADE_HISTORY = ['combined_trade_history']


combine_two_histories_sorted_by_seconds_ago_caller = lambda data: combine_two_histories_sorted_by_seconds_ago(data, TRADE_HISTORY + TRADE_HISTORY_RELATED, COMBINED_TRADE_HISTORY[0], FEATURES_TO_INDEX_IN_HISTORY)


no_reference_data_filepath = make_data_filename('no_reference_data')
if os.path.exists(no_reference_data_filepath):    # check if a file exists https://www.pythontutorial.net/python-basics/python-check-if-file-exists/
    print(f'Loading dataset from pickle file {no_reference_data_filepath}')
    no_reference_data = pd.read_pickle(no_reference_data_filepath)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
else:
    no_reference_data = pd.DataFrame(index=trade_data.index)    # preserving the original index https://stackoverflow.com/questions/18176933/create-an-empty-data-frame-with-index-from-another-data-frame
    no_reference_data[TRADE_HISTORY[0]] = trade_data[TRADE_HISTORY[0]]
    no_reference_data[TRADE_HISTORY_RELATED[0]] = feature_group_as_single_feature(trade_data_flattened_trade_history_and_related_trades, appended_features_wo_reference_features_related_trades, NUM_TRADES_IN_RELATED_TRADE_HISTORY)
no_reference_data.to_pickle(no_reference_data_filepath, protocol=4)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html


encoded_reference_data_filepath = make_data_filename('encoded_reference_data')
trade_data_encoded_filepath = make_data_filename('trade_data_encoded')
trade_data_flattened_trade_history_and_related_trades_encoded_filepath = make_data_filename('trade_data_flattened_trade_history_and_related_trades_encoded')
if os.path.exists(encoded_reference_data_filepath):    # check if a file exists https://www.pythontutorial.net/python-basics/python-check-if-file-exists/
    print(f'Loading dataset from pickle file {encoded_reference_data_filepath}')
    encoded_reference_data = pd.read_pickle(encoded_reference_data_filepath)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
    trade_data_encoded = pd.read_pickle(trade_data_encoded_filepath)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
    trade_data_flattened_trade_history_and_related_trades_encoded = pd.read_pickle(trade_data_flattened_trade_history_and_related_trades_encoded_filepath)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
else:
    trade_data_encoded = encode_with_label_encoders(trade_data)

    encoded_reference_data = pd.DataFrame(index=trade_data.index)    # preserving the original index https://stackoverflow.com/questions/18176933/create-an-empty-data-frame-with-index-from-another-data-frame
    encoded_reference_data[TRADE_HISTORY[0]] = add_reference_data_to_trade_history(trade_data_encoded, REFERENCE_FEATURES_TO_ADD, TRADE_HISTORY)

    print('Encoding the reference features')
    trade_data_flattened_trade_history_and_related_trades_encoded = trade_data_flattened_trade_history_and_related_trades.copy()
    for feature in REFERENCE_FEATURES_TO_ADD:
        encoder = label_encoders[feature]
        for past_trade_idx in range(NUM_TRADES_IN_RELATED_TRADE_HISTORY):
            feature_name = get_appended_feature_name(past_trade_idx, feature, appended_feature_prefix)
            trade_data_flattened_trade_history_and_related_trades_encoded[feature_name] = encoder.transform(trade_data_flattened_trade_history_and_related_trades[feature_name])    # transform the categorical feature column to its encoding

    encoded_reference_data[TRADE_HISTORY_RELATED[0]] = feature_group_as_single_feature(trade_data_flattened_trade_history_and_related_trades_encoded, appended_features_related_trades, NUM_TRADES_IN_RELATED_TRADE_HISTORY)
    
trade_data_encoded.to_pickle(trade_data_encoded_filepath, protocol=4)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
trade_data_flattened_trade_history_and_related_trades_encoded.to_pickle(trade_data_flattened_trade_history_and_related_trades_encoded_filepath, protocol=4)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
encoded_reference_data.to_pickle(encoded_reference_data_filepath, protocol=4)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html


trade_history_data = {'no_reference_data': no_reference_data, 
                      'encoded_reference_data': encoded_reference_data}


reference_data_representation_opt = 'encoded_reference_data'


trade_data[TRADE_HISTORY + TRADE_HISTORY_RELATED] = trade_history_data[reference_data_representation_opt]


train_data_reference_both_histories_single_related_trade_encoded_filepath = make_data_filename('train_data_reference_both_histories_single_related_trade_encoded')
test_data_reference_both_histories_single_related_trade_encoded_filepath = make_data_filename('test_data_reference_both_histories_single_related_trade_encoded')
train_data_reference_same_cusip_history_single_related_trade_encoded_filepath = make_data_filename('train_data_reference_same_cusip_history_single_related_trade_encoded')
test_data_reference_same_cusip_history_single_related_trade_encoded_filepath = make_data_filename('test_data_reference_same_cusip_history_single_related_trade_encoded')
if os.path.exists(test_data_reference_same_cusip_history_single_related_trade_encoded_filepath):
    print(f'Loading dataset from pickle file {test_data_reference_same_cusip_history_single_related_trade_encoded_filepath}')
    train_data_reference_both_histories_single_related_trade_encoded = pd.read_pickle(train_data_reference_both_histories_single_related_trade_encoded_filepath)
    test_data_reference_both_histories_single_related_trade_encoded = pd.read_pickle(test_data_reference_both_histories_single_related_trade_encoded_filepath)
    train_data_reference_same_cusip_history_single_related_trade_encoded = pd.read_pickle(train_data_reference_same_cusip_history_single_related_trade_encoded_filepath)
    test_data_reference_same_cusip_history_single_related_trade_encoded = pd.read_pickle(test_data_reference_same_cusip_history_single_related_trade_encoded_filepath)
else:
    for trade_history_column in TRADE_HISTORY + TRADE_HISTORY_RELATED:
        trade_history_column_dtype = np.stack(trade_data[trade_history_column].to_numpy()).dtype    # `np.stack(...)` converts the numpy array from a numpy array of numpy arrays to a single 3d numpy array
        assert np.issubdtype(trade_history_column_dtype, np.number), f'trade history column dtype: {trade_history_column_dtype}'    # asserts that the dtype of the trade history array is a numerical type: https://stackoverflow.com/questions/29518923/numpy-asarray-how-to-check-up-that-its-result-dtype-is-numeric


    train_data_reference_and_both_histories, test_data_reference_and_both_histories = get_train_test_data_trade_datetime(trade_data, DATE_TO_SPLIT)
    assert len(train_data_reference_and_both_histories) != 0 and len(test_data_reference_and_both_histories) != 0, 'Either train or test data is empty. Consider checking how the train test split is being performed.'
    train_data_reference_and_both_histories = reverse_order_of_trade_history(train_data_reference_and_both_histories, TRADE_HISTORY + TRADE_HISTORY_RELATED)
    test_data_reference_and_both_histories = reverse_order_of_trade_history(test_data_reference_and_both_histories, TRADE_HISTORY + TRADE_HISTORY_RELATED)
    train_data_reference_and_both_histories = train_data_reference_and_both_histories.drop(columns=DATA_PROCESSING_FEATURES + IDENTIFIERS)
    test_data_reference_and_both_histories = test_data_reference_and_both_histories.drop(columns=DATA_PROCESSING_FEATURES + IDENTIFIERS)
    train_data_both_histories = train_data_reference_and_both_histories[TRADE_HISTORY + TRADE_HISTORY_RELATED + TARGET]
    test_data_both_histories = test_data_reference_and_both_histories[TRADE_HISTORY + TRADE_HISTORY_RELATED + TARGET]


    limit_history_to_opt_trades_caller = lambda data: limit_history_to_k_trades(data, {TRADE_HISTORY[0]: NUM_TRADES_IN_TRADE_HISTORY_OPT, TRADE_HISTORY_RELATED[0]: NUM_TRADES_IN_RELATED_TRADE_HISTORY})


    train_data_reference_and_both_histories = limit_history_to_opt_trades_caller(train_data_reference_and_both_histories)
    test_data_reference_and_both_histories = limit_history_to_opt_trades_caller(test_data_reference_and_both_histories)


    train_data_reference_and_both_histories_encoded = encode_with_label_encoders(train_data_reference_and_both_histories)
    test_data_reference_and_both_histories_encoded = encode_with_label_encoders(test_data_reference_and_both_histories)


    train_data_reference_both_histories_single_related_trade_encoded = add_single_trade_from_history_as_reference_features(train_data_reference_and_both_histories_encoded, 
                                                                                                                           TRADE_HISTORY_RELATED, 
                                                                                                                           FEATURES_TO_INDEX_IN_HISTORY.keys(), 
                                                                                                                           prefix=appended_feature_prefix, 
                                                                                                                           datetime_ascending=True)
    train_data_reference_same_cusip_history_single_related_trade_encoded = train_data_reference_both_histories_single_related_trade_encoded.drop(columns=TRADE_HISTORY_RELATED[0])
    test_data_reference_both_histories_single_related_trade_encoded = add_single_trade_from_history_as_reference_features(test_data_reference_and_both_histories_encoded, 
                                                                                                                          TRADE_HISTORY_RELATED, 
                                                                                                                          FEATURES_TO_INDEX_IN_HISTORY.keys(), 
                                                                                                                          prefix=appended_feature_prefix, 
                                                                                                                          datetime_ascending=True)
    test_data_reference_same_cusip_history_single_related_trade_encoded = test_data_reference_both_histories_single_related_trade_encoded.drop(columns=TRADE_HISTORY_RELATED[0])
train_data_reference_both_histories_single_related_trade_encoded.to_pickle(train_data_reference_both_histories_single_related_trade_encoded_filepath, protocol=4)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
test_data_reference_both_histories_single_related_trade_encoded.to_pickle(test_data_reference_both_histories_single_related_trade_encoded_filepath, protocol=4)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
train_data_reference_same_cusip_history_single_related_trade_encoded.to_pickle(train_data_reference_same_cusip_history_single_related_trade_encoded_filepath, protocol=4)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
test_data_reference_same_cusip_history_single_related_trade_encoded.to_pickle(test_data_reference_same_cusip_history_single_related_trade_encoded_filepath, protocol=4)    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html


label_encoders_for_single_related_trade = dict()
categorical_features_for_single_related_trade = []
for new_categorical_feature, original_categorical_feature in zip(get_past_trade_columns(1, REFERENCE_FEATURES_TO_ADD, appended_feature_prefix)[0], REFERENCE_FEATURES_TO_ADD):    # select index 0 to get just the column names
    label_encoders_for_single_related_trade[new_categorical_feature] = label_encoders[original_categorical_feature]
    categorical_features_for_single_related_trade.append(new_categorical_feature)
label_encoders_and_label_encoders_for_single_related_trade = label_encoders | label_encoders_for_single_related_trade    # combine two dictionaries together for Python v3.9+: https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression


experiment_name = 'reference_data_same_cusip_rnn_single_related_trade'
if experiment_name not in results:
    model = NNL1LossEmbeddingsWithRecurrence(BATCH_SIZE, 
                                             NUM_WORKERS, 
                                             train_data_reference_same_cusip_history_single_related_trade_encoded, 
                                             test_data_reference_same_cusip_history_single_related_trade_encoded, 
                                             label_encoders_and_label_encoders_for_single_related_trade, 
                                             CATEGORICAL_FEATURES + categorical_features_for_single_related_trade, 
                                             NUM_NODES_HIDDEN_LAYER, 
                                             NUM_HIDDEN_LAYERS, 
                                             NUM_RECURRENT_LAYERS, 
                                             RECURRENT_HIDDEN_SIZE, 
                                             recurrent_architecture=RECURRENT_ARCHITECTURE, 
                                             power=EMBEDDINGS_POWER)
    train_and_store_model_loss(model, experiment_name)


experiment_name = 'reference_data_single_related_trade_both_histories_interleaved_rnn'
if experiment_name not in results:
    model = NNL1LossEmbeddingsWithRecurrence(BATCH_SIZE, 
                                             NUM_WORKERS, 
                                             combine_two_histories_sorted_by_seconds_ago_caller(train_data_reference_both_histories_single_related_trade_encoded).drop(columns=TRADE_HISTORY + TRADE_HISTORY_RELATED), 
                                             combine_two_histories_sorted_by_seconds_ago_caller(test_data_reference_both_histories_single_related_trade_encoded).drop(columns=TRADE_HISTORY + TRADE_HISTORY_RELATED), 
                                             label_encoders_and_label_encoders_for_single_related_trade, 
                                             CATEGORICAL_FEATURES + categorical_features_for_single_related_trade, 
                                             NUM_NODES_HIDDEN_LAYER, 
                                             NUM_HIDDEN_LAYERS, 
                                             NUM_RECURRENT_LAYERS, 
                                             RECURRENT_HIDDEN_SIZE, 
                                             recurrent_architecture=RECURRENT_ARCHITECTURE, 
                                             trade_history_column=COMBINED_TRADE_HISTORY, 
                                             power=EMBEDDINGS_POWER)
    train_and_store_model_loss(model, experiment_name)


experiment_name = 'reference_data_single_related_trade_both_histories_different_rnn'
if experiment_name not in results:
    model = NNL1LossEmbeddingsWithMultipleRecurrence(BATCH_SIZE, 
                                                     NUM_WORKERS, 
                                                     train_data_reference_both_histories_single_related_trade_encoded, 
                                                     test_data_reference_both_histories_single_related_trade_encoded, 
                                                     label_encoders_and_label_encoders_for_single_related_trade, 
                                                     CATEGORICAL_FEATURES + categorical_features_for_single_related_trade, 
                                                     NUM_NODES_HIDDEN_LAYER, 
                                                     NUM_HIDDEN_LAYERS, 
                                                     NUM_RECURRENT_LAYERS, 
                                                     RECURRENT_HIDDEN_SIZE, 
                                                     recurrent_architecture=RECURRENT_ARCHITECTURE, 
                                                     trade_history_columns=TRADE_HISTORY + TRADE_HISTORY_RELATED, 
                                                     power=EMBEDDINGS_POWER)
    train_and_store_model_loss(model, experiment_name)


for name, test_loss in results.items():
    print(f'{name}\t\tTest error: {test_loss}')
results_ascending_order = sorted(results, key=lambda name: results.get(name))    # sort by minimum test error
opt = results_ascending_order[0]    # optimal name is the one with the minimum test error
