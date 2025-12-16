import pandas as pd
import numpy as np
import scipy.sparse as sp
from google.cloud import bigquery
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import pickle
from itertools import chain, combinations

import os

import wandb
from wandb.keras import WandbCallback

from ficc.models import *
from ficc.data.process_data import process_data
from ficc.utils.graph import *
from ficc.utils.auxiliary_variables import PREDICTORS, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES, IDENTIFIERS
from ficc.utils.eval import eval_model
from ficc.utils.auxiliary_variables import VERY_LARGE_NUMBER

from similarity_functions import *

SEED = 10

TRAIN_TEST_SPLIT = 0.85
LEARNING_RATE = 0.0001
BATCH_SIZE = 1000
NUM_EPOCHS = 100

DROPOUT = 0.01
SEQUENCE_LENGTH = 5
NUM_FEATURES = 5

DATA_QUERY = """ SELECT
*
FROM
`eng-reactor-287421.primary_views.speedy_trade_history` 
WHERE
yield IS NOT NULL
AND yield > 0 
AND yield <= 3 
AND par_traded IS NOT NULL
AND sp_long IS NOT NULL
AND trade_date >= '2021-07-01' 
AND trade_date <= '2021-08-01'
AND msrb_valid_to_date > current_date -- condition to remove cancelled trades
ORDER BY
trade_datetime DESC
"""


def create_input(encoders, df, non_cat_features):
    datalist = []
    datalist.append(np.stack(df['trade_history'].to_numpy()))

    noncat_and_binary = []
    for f in non_cat_features + BINARY:
        noncat_and_binary.append(np.expand_dims(
            df[f].to_numpy().astype('float32'), axis=1))

    datalist.append(np.concatenate(noncat_and_binary, axis=-1))

    for f in CATEGORICAL_FEATURES:
        encoded = encoders[f].transform(df[f])
        datalist.append(encoded.astype('float32'))
    return datalist


def prep_dataframe(df, num_recent_trades, appended_features, appended_features_names_and_functions, filtering_categories, similarity_function=None, shortened_dates_set=set()):
    df.reset_index(drop=True, inplace=True)
    df.index = list(df.index)

    # If using approximated dates, then adds these columns to the dataframe and updates the filtering_categories used when grouping them
    if shortened_dates_set:
        add_shortened_date_columns(df, shortened_dates_set)
    for category_idx, category in enumerate(filtering_categories):
        if category in shortened_dates_set:
            filtering_categories[category_idx] = f'{category}_shortened'

    # Appends the nearest trades to the dataframe
    append_recent_trade_data(df, 
                             num_recent_trades, 
                             appended_features_names_and_functions, 
                             categories=filtering_categories, 
                             is_similar=similarity_function)

    # If using approximated dates, then removes these columns from the dataframe
    if shortened_dates_set:
        drop_shortened_date_columns(df, shortened_dates_set)

    df = df[IDENTIFIERS + PREDICTORS + appended_features]
    df['A/E'] = df.accrued_days / (360 / df.days_in_interest_payment)
    # df['R/M'] = df.coupon / df.days_in_interest_payment    # this feature does not improve accuracy

    return df


def process_data_filenames(raw_data_pickle_filename, processed_data_pickle_filename):
    bq_client = bigquery.Client()
    
    if not os.path.isfile(processed_data_pickle_filename):
        data = process_data(DATA_QUERY,
                            bq_client,
                            SEQUENCE_LENGTH,
                            NUM_FEATURES,
                            raw_data_pickle_filename)
        data.to_pickle(processed_data_pickle_filename)
    else:
        print('START: Reading from processed file')
        # data = pd.read_pickle(processed_data_pickle_filename)
        with open(processed_data_pickle_filename, 'rb') as f:
            data = pickle.load(f)
        print('END: Reading from processed file')

    data.purpose_class.fillna(1, inplace=True)
    return data


'''
Adds the appended features to be used in the model training
'''
def create_appended_features(num_recent_trades, appended_features_names_and_functions):
    appended_features = []

    for appended_feature_name in appended_features_names_and_functions:
        appended_features = appended_features + \
            [f'{appended_feature_name}_{i}' for i in range(num_recent_trades)]
    
    return appended_features


'''
Called when we are using approximated versions of dates: adds a column to the dataframe with the approximated date
'''
def add_shortened_date_columns(df, date_column_names_to_shorten, granularity='year'):
    assert granularity == 'month' or granularity == 'year', f"`granularity` must be 'month' or 'year', but was passed in as {granularity}"

    def shorten_date(date):
        month = date.month
        if granularity == 'year':
            month = 1
        return pd.Timestamp(date.year, month, 1)

    columns_set = set(df.columns.to_list())
    for date_column_name in date_column_names_to_shorten:
        shortened_column_name = f'{date_column_name}_shortened'
        assert shortened_column_name not in columns_set
        df[shortened_column_name] = [shorten_date(date) for date in df[date_column_name]]


'''
Called when we are using approximated versions of dates: removes the column to the dataframe with the approximated date
'''
def drop_shortened_date_columns(df, date_column_names_to_shorten):
    columns_set = set(df.columns.to_list())
    date_column_names_shortened = [f'{date_column_name}_shortened' for date_column_name in date_column_names_to_shorten]
    for date_column_name in date_column_names_shortened:
        assert date_column_name in columns_set
    df.drop(columns=date_column_names_shortened, inplace=True)

'''
num_recent_trades: number of recent trades to be appended to the data
appended_features_similarity_categories: list of categories which are used to filter the data, and thus, used to consider similar functions
appended_features_names_and_functions: see description of APPENDED_FEATURES_NAMES_AND_FUNCTIONS in the runner() method
similarity_function: similarity function that should be defined in `similarity_functions.py`
testing_categories: list categories on which we can see model performance, e.g., if we pass in ['incorporated_state_code'] then we can see the MAE for each state code
data: cached data to speed up procedure
shortened_dates_set: see description of shortened_dates_set in the runner() method
wandb_name: name used for this run for wandb logging
'''
def run(num_recent_trades, 
        appended_features_similarity_categories, 
        appended_features_names_and_functions, 
        similarity_function=None, 
        testing_categories=None, 
        data=None, 
        shortened_dates_set=set(), 
        wandb_name=''):
    try:
        tf.keras.utils.set_random_seed(SEED)
    except:
        import random
        import numpy as np
        import tensorflow as tf
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
    layer_initializer = initializers.RandomNormal(mean=0.0, stddev=0.1, seed=SEED)
    
    if data is None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "eng-reactor-287421-112eb767e1b3.json"
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        data = process_data_filenames('raw_data.pkl', 'processed_data.pkl')

    test_index = int(len(data) * TRAIN_TEST_SPLIT)
    test_dataframe = data[test_index:]
    train_dataframe = data[:test_index]

    APPENDED_FEATURES = create_appended_features(num_recent_trades, appended_features_names_and_functions)

    NON_CAT_FEATURES_APPENDED = NON_CAT_FEATURES + ['A/E'] + APPENDED_FEATURES

    encoders = {}
    fmax = {}
    for f in CATEGORICAL_FEATURES:
        fprep = preprocessing.LabelEncoder().fit(data[f].drop_duplicates())
        fmax[f] = np.max(fprep.transform(fprep.classes_))
        encoders[f] = fprep

    prep_dataframe_caller = lambda df: prep_dataframe(df, 
                                                      num_recent_trades, 
                                                      APPENDED_FEATURES, 
                                                      appended_features_names_and_functions, 
                                                      appended_features_similarity_categories, 
                                                      similarity_function, 
                                                      shortened_dates_set)

    train_dataframe = prep_dataframe_caller(train_dataframe)
    x_train = create_input(encoders, train_dataframe, NON_CAT_FEATURES_APPENDED)
    y_train = train_dataframe.yield_spread.to_numpy().astype('float32')

    test_dataframe = prep_dataframe_caller(test_dataframe)

    trade_history_normalizer = Normalization()
    trade_history_normalizer.adapt(x_train[0])

    noncat_binary_normalizer = Normalization()
    noncat_binary_normalizer.adapt(x_train[1])

    model = get_model_instance(
        "lstm_yield_spread_model",
        learning_rate=LEARNING_RATE,
        SEQUENCE_LENGTH=SEQUENCE_LENGTH,
        NUM_FEATURES=NUM_FEATURES,
        NON_CAT_FEATURES=NON_CAT_FEATURES_APPENDED,
        BINARY=BINARY,
        CATEGORICAL_FEATURES=CATEGORICAL_FEATURES,
        noncat_binary_normalizer=noncat_binary_normalizer,
        trade_history_normalizer=trade_history_normalizer,
        fmax=fmax)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])

    wandb_logger = wandb.init(project="KNN", entity="ficc-ai", name=f"{wandb_name}.{num_recent_trades}NN")

    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE, 
        callbacks=[
            WandbCallback(),
            EarlyStopping(
                monitor="val_loss", 
                patience=25,
                verbose=0,
                mode="auto",
                restore_best_weights=True
            )]
    )

    mae = eval_model(model, 
                     test_dataframe, 
                     lambda df: create_input(encoders, df, NON_CAT_FEATURES_APPENDED),
                     lambda x: x.yield_spread, 
                     wandb=wandb_logger, 
                     categories=testing_categories)
    wandb_logger.finish()

    return mae


if __name__ == "__main__":
    def powerset(lst):
        # this function creates a powerset from a provided list and returns it as a list
        # https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
        return [list(combination) for combination in list(chain.from_iterable(combinations(lst, r) for r in range(len(lst) + 1)))]

    filters = ['incorporated_state_code', 'rating', 'maturity_date', 'trade_type']    #, 'is_general_obligation', 'state_tax_status', 'federal_tax_status']

    # TODO: consider yield_spread_recent as a gap instead of the raw value
    # TODO: consider keeping track of the quantity

    # dictionary where the key is the name of the data_column of the appended feature, and the value is a pair where the first 
    # item is a function to fill in the respective data_column, and the second item is the initialization value for this data_column 
    APPENDED_FEATURES_NAMES_AND_FUNCTIONS = {'related_last_yield_spread': (lambda curr, neighbor: neighbor['yield_spread'], 0), 
                                             'related_last_seconds_ago': (lambda curr, neighbor: np.log10((curr['trade_datetime'] - neighbor['trade_datetime']).total_seconds()), np.log10(VERY_LARGE_NUMBER)),
                                             'related_last_quantity': (lambda curr, neighbor: neighbor['quantity'], 0)}
    NUM_RECENT_TRADES = 5

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "eng-reactor-287421-112eb767e1b3.json"
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    orig_data = process_data_filenames('raw_data.pkl', 'processed_data.pkl')    # this ensures that `process_data_filenames` is called only once for multiple runs with the same data

    # this set contains all categories that we will be using an approximated version of for the purposes of grouping nearby trades
    # in order to speed up the grouping process
    shortened_dates_set = set(['maturity_date'])

    def all_shortened_dates_in_filters(filter_set, needed_filters):
        '''Makes sure that all items in `shortened_date_set` are (1) not in the `filter_set`, or (2) if in `filter_set`, then 
        also in `needed_filters`. For example, this ensures that no run is called with a filter of say `maturity_date`, but 
        the similarity function does not use `maturity_date`, since this would cause extremely poor runtime.'''
        needed_filters = set(needed_filters)
        return all([shortened_date not in filter_set or shortened_date in needed_filters for shortened_date in shortened_dates_set])

    # list of triple where the item 1 is a similarity function, item 2 is a list of needed filters, and item 3 is a string representing the similarity function to be sued for logging
    similarity_functions_with_needed_filters = [# (all_same, [], 'all_same'), 
                                                # (rating_pm1, ['rating'], 'rating_pm1'), 
                                                # (maturity_date_pm6months_shortened, ['maturity_date'], 'maturity_date_pm6months_shortened'), 
                                                (maturity_date_pm1year_shortened, ['maturity_date'], 'maturity_date_pm1year_shortened'), 
                                                (maturity_date_pm2years_shortened, ['maturity_date'], 'maturity_date_pm2years_shortened'), 
                                                # (rating_pm1_maturity_date_pm6months_shortened, ['rating', 'maturity_date'], 'rating_pm1_maturity_date_pm6months_shortened'), 
                                                (rating_pm1_maturity_date_pm1year_shortened, ['rating', 'maturity_date'], 'rating_pm1_maturity_date_pm1year_shortened'), 
                                                (rating_pm1_maturity_date_pm2years_shortened, ['rating', 'maturity_date'], 'rating_pm1_maturity_date_pm2years_shortened')
                                                ]
    
    runs_dict = dict()
    for similarity_function, needed_filters, similarity_function_name in similarity_functions_with_needed_filters:
        for filter_list in powerset(filters)[1:]:
            filter_list_as_set = set(filter_list)
            if all([needed_filter in filter_list for needed_filter in needed_filters]):
                if all_shortened_dates_in_filters(filter_list_as_set, needed_filters):
                    run_name = '-'.join(filter_list) + '.' + similarity_function_name    # this creates the name of the run to be used for logging
                    
                    runs_dict[run_name] = run(NUM_RECENT_TRADES, 
                                              filter_list, 
                                              APPENDED_FEATURES_NAMES_AND_FUNCTIONS, 
                                              similarity_function, 
                                              data=orig_data.copy(),    
                                              shortened_dates_set=shortened_dates_set.intersection(filter_list_as_set), 
                                              wandb_name=run_name)
    print(runs_dict)
