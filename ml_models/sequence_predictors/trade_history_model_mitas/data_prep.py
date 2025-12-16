import warnings
import multiprocess as mp    # using `multiprocess` instead of `multiprocessing` because function to be called in `map` is in the same file as the function which is calling it: https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror

import numpy as np
import pandas as pd

from ficc.utils.related_trade import get_appended_feature_name
from ficc.utils.auxiliary_functions import list_to_index_dict
from ficc.utils.trade_dict_to_list_mappings import TRADE_TYPE_MAPPING

import sys
sys.path.insert(0,'../')

from trade_history_model_mitas.auxiliary_functions import numpy_array_to_list_of_arrays


def get_past_trade_columns(num_past_trades, feature_names_per_trade, prefix='', trade_type_actual=False, trade_type_column='trade_type', categorical_features_per_trade=[]):
    '''Gets the column names corresponding to the past trades in the trade history for 
    `num_past_trades` past trades. Also returns the new categorical features as a result 
    of making this change. `trade_type_actual` is a boolean that determines whether we 
    use the actual trade type (e.g., 'S', 'P', 'D') in the data or the encoded trade type. 
    `categorical_features_per_trade` is a list of column names that are categorical for 
    each trade.'''
    if trade_type_actual:
        feature_names_per_trade = [feature for feature in feature_names_per_trade if not feature.endswith('trade_type1') and not feature.endswith('trade_type2')]    # all features besides ...trade_type1 and ...trade_type2
        feature_names_per_trade.append(trade_type_column)

    all_past_trade_columns = []
    for trade_idx in range(num_past_trades):
        all_past_trade_columns.extend([get_appended_feature_name(trade_idx, feature_name, prefix) for feature_name in feature_names_per_trade])
    
    if trade_type_actual and trade_type_column not in categorical_features_per_trade: categorical_features_per_trade.append(trade_type_column)    # add in the actual trade type columns
    all_categorical_features = []
    for trade_idx in range(num_past_trades):
        all_categorical_features.extend([get_appended_feature_name(trade_idx, categorical_feature_name, prefix) for categorical_feature_name in categorical_features_per_trade])

    return all_past_trade_columns, all_categorical_features


def feature_group_as_single_feature(df, group_features_flattened, num_past_trades, flatten_each_row=False, multiprocessing=True):
    '''Isolates the features in `group_features_flattened` and reshapes this data into a 
    2D array for every row, where the first dimension of this 2D array is the `num_past_trades` 
    and the second is the number of features per group. Note that we infer the value of the 
    number of features per group since this can change due to the representation of the 
    features (e.g., if the features are encoded versus if the features are embedded). Returns 
    the feature group as a list so that it can be put into a column in the DataFrame.'''
    features_as_numpy_array = df[group_features_flattened].to_numpy()

    if flatten_each_row:
        if multiprocessing:
            print('Using multiprocessing to flatten each row')
            with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
                features_as_numpy_array = pool_object.map(np.hstack, features_as_numpy_array)    # .map(...) preserves the results order https://stackoverflow.com/questions/41273960/python-3-does-pool-keep-the-original-order-of-data-passed-to-map
            features_as_numpy_array = np.vstack(features_as_numpy_array)    # stack the list of arrays vertically https://stackoverflow.com/questions/27516849/how-to-convert-list-of-numpy-arrays-into-single-numpy-array
        else:
            features_as_numpy_array = np.apply_along_axis(np.hstack, 1, features_as_numpy_array)    # first argument: function, second argument: axis, third argument: array; `.apply_along_axis(...)` is very slow for large dataframes: https://stackoverflow.com/questions/45604688/apply-function-on-each-row-row-wise-of-a-numpy-array

    return numpy_array_to_list_of_arrays(features_as_numpy_array.reshape(len(df), num_past_trades, -1))    # need to return this as a list in order to put it into the DataFrame; -1 is an inferred dimension


def limit_history_to_k_trades(df, feature_to_limit_dict):
    '''Limits the number of trades in the history to `k` trades maximum. This means that 
    anything beyond that is truncated. This procedure assumes that the trades are in ascending 
    order by `trade_datetime`, so the most recent trade is at the end of the list.'''
    df = df.copy()
    for feature, limit in feature_to_limit_dict.items():
        history_as_numpy_array = df[feature].to_numpy()    # convert history to numpy array
        history_as_numpy_array = np.stack(history_as_numpy_array)    # converts the numpy array from a numpy array of numpy arrays to a single 3d numpy array
        df[feature] = history_as_numpy_array[:, -limit:, :].tolist()    # first index is trade index, second index is trade history index, third index is trade history feature; convert the numpy array to a list of numpy arrays to create a pd.Series object out of it when inserted into a dataframe: https://stackoverflow.com/questions/38840319/put-a-2d-array-into-a-pandas-series
    return df


def combine_two_histories_sorted_by_seconds_ago(df, history_column_names, combined_history_column_name, features_to_index_in_history):
    '''Combines two histories from `history_column_names` into a single history. 
    More specifically, assume that a history is represented as a series where each 
    item in the series is a list. The list contains the important features of a 
    trade, and each history series must have the same features in the same places.'''
    assert len(history_column_names) == 2, 'Current implementation can only have two features in `history_column_names`'
    assert np.all(np.array([len(df[column_name].iloc[0][0]) for column_name in history_column_names]) == len(df[history_column_names[0]].iloc[0][0]))    # checks that all columns are series with lists that have the same number of items
    df = df.copy()

    # add an additional feature to each of the past trades in the same CUSIP trade history and the related trade history denoting whether the trade comes from the same CUSIP (0) or from a related CUSIP (1)
    add_item_to_every_row_func = lambda item, num_rows: lambda row: np.concatenate((row, np.array([[item] * num_rows]).T), axis=1)    # add `item` to every row of a 2D numpy array; https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
    for idx, column in enumerate(history_column_names):
        df[column] = df[column].apply(add_item_to_every_row_func(idx, len(df[column].iloc[0])))    # len(df[column].iloc[0]) equals the number of trades in the history

    seconds_ago_index = features_to_index_in_history['seconds_ago']
    sort_history_by_seconds_ago = lambda history: np.array(sorted(history, key=lambda item: item[seconds_ago_index], reverse=True))    # wrap the result with np.array(...) since sorted(...) returns a list
    combined_history = np.concatenate([np.array(df[column].to_list()) for column in history_column_names], axis=1)        # cannot use .to_numpy() since this creates a numpy array where each item is a numpy array instead of a single 3-dimensional numpy array; so have to call .to_list() and then wrap in np.array(...)
    df[combined_history_column_name] = numpy_array_to_list_of_arrays(combined_history)    # only a list can be put into a DataFrame column
    df[combined_history_column_name] = df[combined_history_column_name].apply(sort_history_by_seconds_ago)
    return df


def remove_feature_from_trade_history(df, trade_history_columns, features_to_remove, features_to_index_in_history):
    '''Remove a feature from each row in the trade history.'''
    df = df.copy()
    indices_to_remove = [features_to_index_in_history[feature] for feature in features_to_remove]
    indices_to_remove = sorted(indices_to_remove, reverse=True)
    for trade_history_column in trade_history_columns:
        for index_to_remove in indices_to_remove:
            df[trade_history_column] = df[trade_history_column].apply(lambda trade_history: np.delete(trade_history, index_to_remove, axis=1))
    return df


def convert_trade_type_encoding_to_actual(df, num_trades_in_history, new_column_name='trade_type', prefix='last_'):
    '''Replace every encoded trade_type value in a flattened `df` (without reference data) 
    with the decoded actual trade_type. Using subtraction as the aggregation method.'''
    decoded_trade_type_map = {2 * value1 - value2 : trade_type for trade_type, (value1, value2) in TRADE_TYPE_MAPPING.items()}
    assert len(decoded_trade_type_map) == len(TRADE_TYPE_MAPPING), 'Aggregation procedure is not one-to-one'

    df = df.copy()    # prevents this function from mutating `df` when adding `trade_type_column_name` as a new feature
    df_columns_set = set(df.columns)
    old_columns, new_columns = [], []
    for trade_idx in range(num_trades_in_history):
        get_trade_type_column_name = lambda num: get_appended_feature_name(trade_idx, f'trade_type{num}', prefix)
        trade_type1_column_name, trade_type2_column_name = get_trade_type_column_name(1), get_trade_type_column_name(2)
        old_columns.extend([trade_type1_column_name, trade_type2_column_name])
        trade_type_column_name = get_appended_feature_name(trade_idx, new_column_name, prefix)
        assert trade_type_column_name not in df_columns_set, f'Need to use the `{trade_type_column_name}` column to store the results of the decoding'
        new_columns.append(trade_type_column_name)
        with warnings.catch_warnings():    # temporarily suppress warnings for a block of code: https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
            warnings.simplefilter('ignore')    # suppress this warning for the line below: `SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead`
            df.loc[:, trade_type_column_name] = ((2 * df[trade_type1_column_name] - df[trade_type2_column_name]).map(decoded_trade_type_map.get)).astype('category')    # change type to category so LightGBM can use this feature as a categorical feature
    return df, old_columns, new_columns


def add_reference_data_to_trade_history(df, reference_features, trade_history_column, flatten_each_row=False):
    '''Add reference data to each trade in the trade history. Since this only adds the 
    reference data of the target trade, it is only appropriate when the trade history 
    is of rom the same CUSIP, since all trades from the same CUSIP share the same 
    reference data. Returns the new trade history as a pandas Series.'''
    if type(trade_history_column) == list: 
        assert len(trade_history_column) == 1, 'Can only process a single trade history column'
        trade_history_column = trade_history_column[0]
    trade_history_as_numpy_array = df[trade_history_column].to_numpy()    # isolate the trade history features
    trade_history_as_numpy_array = np.stack(trade_history_as_numpy_array)    # converts the numpy array from a numpy array of numpy arrays to a single 3d numpy array
    num_trades, num_trades_in_trade_history, _ = trade_history_as_numpy_array.shape    # third entry in shape tuple is the number of features in the trade history
    reference_features_as_numpy_array = df[reference_features].to_numpy()    # isolate the reference features
    if flatten_each_row: reference_features_as_numpy_array = np.apply_along_axis(np.hstack, 1, reference_features_as_numpy_array)    # first argument: function, second argument: axis, third argument: array; `.apply_along_axis(...)` is very slow for large dataframes: https://stackoverflow.com/questions/45604688/apply-function-on-each-row-row-wise-of-a-numpy-array
    features_as_numpy_array = np.tile(reference_features_as_numpy_array, num_trades_in_trade_history)    # create `num_trades_in_trade_history` copies of the reference feature values to put into each row of the trade history
    features_as_numpy_array = np.reshape(features_as_numpy_array, (num_trades, num_trades_in_trade_history, -1))    # reshape feature values to be directly concatenated into the trade history array; third argument used to be `len(reference_features)` but changed to -1 due to case where the reference feature was a vector embedding that has no been flattened
    return np.concatenate((trade_history_as_numpy_array, features_as_numpy_array), axis=2).tolist()    # combine old trade history with new features; convert the numpy array to a list of numpy arrays to create a pd.Series object out of it when inserted into a dataframe: https://stackoverflow.com/questions/38840319/put-a-2d-array-into-a-pandas-series


def embed_with_arrays(df, arrays, features=None):
    '''Embed each feature in `df` by iterating through the dictionary `arrays`, where each 
    item has a key for the feature name and a value as the corresponding embedding numpy. 
    array. `features` selects which features are to be embedded, and if no argument is 
    passed in, then all possible features are embedded. Note that `df` must already be 
    encoded before calling this function.'''
    df = df.copy()
    if features == None: features = arrays.keys()
    for feature in features:
        embeddings_array = arrays[feature]
        df[feature] = df[feature].map(list_to_index_dict(embeddings_array))    # `.map(...)` is the fastest way to do this: https://stackoverflow.com/questions/20250771/remap-values-in-pandas-column-with-a-dict-preserve-nans
    return df


def add_single_trade_from_history_as_reference_features(df, trade_history_column, feature_names_per_trade, prefix='', datetime_ascending=False):
    '''Flattens the features from the most recent trade in the history, specified by 
    `trade_history_column`, and appends it to `df`. The names of these features are given 
    in `feature_names_per_trade`.`datetime_ascending` is a boolean that marks whether the 
    trades in `trade_history_column` are in ascending or descending order of trade_datetime.'''
    df = df.copy()
    if type(trade_history_column) == list:
        assert len(trade_history_column) == 1, f'Can only process a single trade history column, but was passed in: {trade_history_column}'
        trade_history_column = trade_history_column[0]
    trade_history_as_numpy_array = df[trade_history_column].to_numpy()    # isolate the trade history features
    trade_history_as_numpy_array = np.stack(trade_history_as_numpy_array)    # converts the numpy array from a numpy array of numpy arrays to a single 3d numpy array
    most_recent_trade_idx = -1 if datetime_ascending else 0
    most_recent_trade = trade_history_as_numpy_array[:, most_recent_trade_idx, :]
    for idx, new_feature in enumerate(get_past_trade_columns(1, feature_names_per_trade, prefix)[0]):    # select index 0 to get just the column names
        df[new_feature] = most_recent_trade[:, idx]
    return df


def is_sorted(arr, ascending=True):
    '''Returns `True` if `arr` is sorted; False otherwise.'''
    if isinstance(arr, pd.Series): arr = arr.to_numpy()
    assert isinstance(arr, np.ndarray), '`arr` must be an instance of a numpy array'
    return np.all(arr[:-1] <= arr[1:]) if ascending else np.all(arr[:-1] >= arr[1:])    # https://stackoverflow.com/questions/47004506/check-if-a-numpy-array-is-sorted