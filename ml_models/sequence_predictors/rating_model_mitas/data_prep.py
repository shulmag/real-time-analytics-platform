import os
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype

from sklearn.preprocessing import LabelEncoder

from ficc.utils.auxiliary_variables import VERY_LARGE_NUMBER
from ficc.utils.diff_in_days import diff_in_days_two_dates


'''Read from `filename`, which should be a pickle file with the trade data after 
running the query.'''
def read_processed_file_pickle(filename):
    if not os.path.isfile(filename):
        print(f'{filename} not found')
        return None
    print(f'START: Reading from processed file at {filename}')
    data = pd.read_pickle(filename)
    print(f'END: Reading from processed file at {filename}')
    return data


'''Adds most recent event info to `df`. Need a special function for this since 
the most recent event information is stored as a dictionary in each row. The 
`reference_datetime` is when the pickle file was created.'''
def add_most_recent_event(df, reference_datetime):
    assert 'most_recent_event_code' not in df.columns, '"most_recent_event_code" was already found in the the dataframe'
    assert 'most_recent_event_days_ago' not in df.columns, '"most_recent_event_days_ago" was already found in the the dataframe'
    
    df = df.copy()
    get_event_code = lambda row: row['most_recent_event']['code'] if row['most_recent_event'] != None else 0
    get_event_days_ago = lambda row: np.log10(1 + diff_in_days_two_dates(reference_datetime, row['most_recent_event']['date'])) if row['most_recent_event'] != None else np.log10(VERY_LARGE_NUMBER)

    df['most_recent_event_code'] = df.apply(get_event_code, axis=1)
    df['most_recent_event_days_ago'] = df.apply(get_event_days_ago, axis=1)
    new_columns = ('most_recent_event_code', 'most_recent_event_days_ago')
    print(f'The new columns from using the most recent event info is: {new_columns}')
    return df, new_columns


'''Checks whether the columns in `additional_features` are present in `df`, and
returns a list of all the features that are not found.'''
def check_additional_features(df, additional_features):
    features_not_found = []
    columns_set = set(df.columns)
    for feature in additional_features:
        if feature not in columns_set:
            print(f'{feature} not found')
            features_not_found.append(feature)
    if len(features_not_found) == 0:
        print('All additional features were found')
    return features_not_found


'''Removes all trades with a duplicate value for `feature` and keeps the first trade. 
This is a step in converting trade data to bond data.'''
def remove_duplicates_by_feature(df, *features):
    num_rows = len(df)
    df = df.drop_duplicates(subset=features, keep='first')
    print(f'{num_rows - len(df)} rows had the same {features} as a previous trade so it was removed and only the first occurence of that {features} was kept')
    return df


'''Replaces nan values in `df`. `features_to_replace_nan_value_dict` is a dictionary  
mapping of the feature name to its corresponding replacement value for nan. Note that 
this function is very similar to `replace_nan_for_features` in `../yield_spread_model_mitas/data_prep.py`, 
but this function does not handle adding a missingness flag.'''
def fill_nan(df, features_to_replace_nan_value_dict):
    df = df.copy()
    for feature, replacement_value in features_to_replace_nan_value_dict.items():
        if feature in df.columns:
            print(f'{len(df[pd.isna(df[feature])])} trades had a null value for {feature} and were replaced by the value {replacement_value}')
            if df[feature].dtype.name == 'category':    # checks whether the dtype of the column is `category` and if so, converts to `string` in order to fill in a string replacement value
                df[feature] = df[feature].astype('string')
            assert is_numeric_dtype(df[feature].dtype) if type(replacement_value) == int or type(replacement_value) == float else is_string_dtype(df[feature].dtype), f'The column for feature {feature} has type {df[feature].dtype}, but the replacement value has type {type(replacement_value)}'
            df[feature] = df[feature].fillna(replacement_value)
            #assert df[feature].dtype == float and type(replacement_value) == str, f"The column for feature {feature} has type {df[feature].dtype}, but the replacement value has type {type(replacement_value)}"
        else:
            print(f'{feature} was not found to fill nan value for')
    return df


'''Remove MSRB fields in order to only consider ICE reference data.'''
def remove_MSRB_fields(df):
    MSRB_fields = ('is_non_transaction_based_compensation', 'transaction_type', 'trade_type', 'quantity', 'days_to_maturity', 'days_to_call', 'last_seconds_ago', 'last_yield_spread', 'days_to_settle', 'days_to_par', 'accrued_days', 'yield_spread', 'trade_history', 'A/E')
    columns_set = set(df.columns)
    MSRB_fields_to_remove = [field for field in MSRB_fields if field in columns_set]
    return df.drop(columns=MSRB_fields_to_remove)


'''Remove features that have a single unique value since these features will not help with predictions.'''
def remove_fields_with_single_unique_value(df, columns_to_inspect=None):
    if columns_to_inspect is None: columns_to_inspect = df.columns
    else: columns_to_inspect = list(set(columns_to_inspect) & set(df.columns))
    print(f'Checking (and removing) the following features if they have a single unique value: {columns_to_inspect}')

    columns_to_remove = []
    for column in columns_to_inspect:
        try:
            unique_values = df[column].unique()
        except TypeError:    # handles the case where `trade_history` is a list and cannot be hashed to determine if it is unique among the other rows in the dataframe
            warnings.warn(f'Since `{column}` is an unhashable type, this feature could not be removed even if it has a single unique value.')
        else:
            if len(unique_values) == 1:
                print(f'{column} has a single unique value of {unique_values[0]}, and was thus removed')
                columns_to_remove.append(column)
    return df.drop(columns=columns_to_remove)


'''Remove all rows with at least one nan value.'''
def remove_rows_with_nan_value(df):
    num_trades_without_removing_null = len(df)
    df = df.dropna()
    print(f'{num_trades_without_removing_null - len(df)} rows had a null value for at least one of the features and were removed')
    return df


'''Remove all bonds with an NR rating.'''
def remove_NR_ratings(df):
    num_trades_with_NR_rating = len(df[df['rating'] == 'NR'])
    print(f'{num_trades_with_NR_rating} trades had a rating of NR and were removed from the data')
    return df.drop(df[df.rating == 'NR'].index)


'''This function manually encodes the rating so that we can choose the order as specified in `ratings_order`.'''
def encode_rating(df, ratings_order, column_name='rating'):
    return df[column_name].astype('str').apply(lambda rating: ratings_order.index(rating))    # need to change the type to `str` in order to appropriately handle the categories issue where NR is still found after removal


'''Encode all features in `features` with a `LabelEncoder` to convert the features into integers.'''
def encode(df, features, ratings_order=None):
    df_encoded = df.copy()
    encoders = dict()
    for feature in features:
        if feature in df_encoded:
            if feature == 'rating' and ratings_order is not None:
                df_encoded['rating'] = encode_rating(df_encoded, ratings_order)   # manually encode the ratings so that we can choose the ordering
            else:
                encoder = LabelEncoder()
                df_encoded[feature] = encoder.fit_transform(df_encoded[feature])
                encoders[feature] = encoder
    return df_encoded, encoders


'''Encode each feature in `df` by iterating through the dictionary `encoders`, where each 
item has a key for the feature name and a value as the corresponding LabelEncoder.'''
def encode_with_encoders(df, encoders, ratings_order=None):
    df_encoded = df.copy()
    for feature, encoder in encoders.items():
        try:
            df_encoded[feature] = encoder.transform(df_encoded[feature])
        except ValueError:    # output a more informative error message
            raise ValueError(f'{feature} has a value that was not found in its corresponding encoder.')
    if ratings_order is not None:
        df_encoded['rating'] = encode_rating(df_encoded, ratings_order)
    return df_encoded


'''Return subsets of the data. To make sure that the data in the training set is 
earlier in time than the validation set which is earlier in time to the test set, 
we choose the tail of `df` with the assumption that `df` is in descending order.'''
def get_subsets_of_data(df, subset_sizes):
    df_subsets = dict()
    for subset_size in subset_sizes:
        df_subsets[subset_size] = df.tail(subset_size)
    
    max_subset_size = max(subset_sizes)
    dataset_size = len(df)
    test_data = df.head(dataset_size - max_subset_size)
    print(f'largest Train / Validation set size: {max_subset_size}')
    test_set_size = len(test_data)
    print(f'Test set size: {test_set_size}')
    print(f'Train / Test split: {round(100 * max_subset_size / dataset_size, 3)} % / {round(100 * test_set_size / dataset_size, 3)} %')
    return df_subsets, test_data