import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ficc.utils.fill_missing_values import replace_nan_with_value

import sys
sys.path.insert(0,'../')

from yield_spread_model_mitas.data_exploration import filter_data_with_string_conditions


# The following list was formed by looking at all possible features and choosing some that may intuitively have predictive power, i.e., this list was not determined programmatically or through experiments
ADDITIONAL_CATEGORICAL_FEATURES = ['asset_claim_code', 
                                   'capital_type', 
                                   'muni_issue_type', 
                                   'muni_security_type', 
                                   'orig_instrument_enhancement_type', 
                                   'other_enhancement_type', 
                                   'purpose_sub_class', 
                                   'sale_type', 
                                   'sec_regulation', 
                                   'secured', 
                                   'state_tax_status', 
                                   'use_of_proceeds']


# Discussed with a team member to replace the nan values in the data with 0, except for `purpose_sub_class` which should be replaced by 1 since 1 indicates that the value is unknown
FEATURES_AND_NAN_REPLACEMENT_VALUES = {'purpose_class': 0,
                                       'asset_claim_code': 0,
                                       'muni_issue_type': 0,
                                       'orig_instrument_enhancement_type': 0,
                                       'other_enhancement_type': 0,
                                       'purpose_sub_class': 1,
                                       'sec_regulation': 0,
                                       'secured': 0}


NOT_SEEN_IN_DATA_STRING = ' not_seen_in_data'    # used as a placeholder text for feature values that were not previously seen by the corresponding LabelEncoder; extra space in the beginning allows for it to be first in sorted order and therefore encoded as 0


'''Assumes `filename` has the date at the very end starting with the year, 
e.g., `processed_data_2022-04-05-16-37`.'''
def get_datestring_from_filename(filename):
    year_position = filename.index('20')    # assumes that the year starts with `20`
    end_of_filename_position = filename.index('.pkl')   # assumes that the filename corresponds to a pickle file
    return filename[year_position:end_of_filename_position]


'''Remove a row in `df` if that row has the condition that `df[feature]` 
equals any value in `values`.'''
def remove_rows_with_feature_value(df, feature, values):
    num_rows_without_removal = len(df)
    conditions_as_strings = [f'df["{feature}"] != {value}' for value in values]
    df = filter_data_with_string_conditions(df, [conditions_as_strings])
    print(f'{num_rows_without_removal - len(df)} rows had {feature} in {values} and were removed')
    return df


'''Replaces nan values for each feature with its corresponding default value 
in `features_and_default_values` dict. Note that this function is very similar 
to `fill_nan` in `../rating_model_mitas/data_prep.py`, but handles adding a 
missingness flag.'''
def replace_nan_for_features(df, features_and_default_values, add_missingness_flag=False, verbose=False):
    print(f'Replacing nan values for the following features: {features_and_default_values.keys()}')
    df = df.copy()
    df_columns_set = set(df.columns)
    new_binary_features_from_missingness_flag = []
    for feature, default_value in features_and_default_values.items():
        if feature not in df_columns_set:
            print(f'{feature} not found in dataframe')
        elif df[feature].isnull().values.any():
            if add_missingness_flag:
                new_binary_feature_from_missingness_flag = feature + '_missing'
                df[new_binary_feature_from_missingness_flag] = df[feature].isnull()
                new_binary_features_from_missingness_flag.append(new_binary_feature_from_missingness_flag)
            if df[feature].dtype.name == 'category': df[feature] = df[feature].astype('string')    # checks whether the dtype of the column is `category` and if so, converts to `string` in order to fill in a string replacement value
            assert is_numeric_dtype(df[feature].dtype) if type(default_value) == int or type(default_value) == float else is_string_dtype(df[feature].dtype), f'The column for feature {feature} has type {df[feature].dtype}, but the replacement value has type {type(default_value)}'
            if verbose: print(f'{len(df[pd.isna(df[feature])])} trades had a null value for {feature} and were replaced by the value {default_value}')
            replace_nan_with_value(df, feature, default_value)
    return df, new_binary_features_from_missingness_flag


'''Checks whether `additional_features` are columns in `df`, and if not, returns 
the subset of `additional_features` that are present in `df`.'''
def check_additional_features(df, additional_features):
    print(f'Checking if the following additional features are in the dataframe: {additional_features}')
    df_columns_set = set(df.columns)
    additional_features_set = set(additional_features)
    features_not_found_in_df = additional_features_set - df_columns_set
    if len(features_not_found_in_df) != 0: print(f'The following features were not found in the dataframe: {features_not_found_in_df}')
    return list(additional_features_set - features_not_found_in_df)


'''Converts numerical features to categorical features based on the assumption 
that these numerical features have very few unique values. Returns a new 
dataframe that has renamed the old numerical features, and the names of these new 
categorical (or binary if the feature only has two unique values) features.'''
def convert_numerical_features_to_categorical(df, numerical_features_to_convert):
    columns_set = set(df.columns)
    numerical_features_to_convert_in_df = []
    for numerical_feature in numerical_features_to_convert:
        if numerical_feature not in columns_set:
            print(f'{numerical_feature} not found in dataframe')
        else:
            numerical_features_to_convert_in_df.append(numerical_feature)
    
    NUM_UNIQUE_VALUES_THRESHOLD = 10
    addon = '_cat'    # this string is added onto the end of the original feature name
    new_binary_features = []
    new_categorical_features = []
    for numerical_feature in numerical_features_to_convert_in_df:
        num_unique_values = len(df[numerical_feature].unique())
        if num_unique_values == 2:
            new_binary_features.append(numerical_feature + addon)
        elif num_unique_values <= NUM_UNIQUE_VALUES_THRESHOLD:
            new_categorical_features.append(numerical_feature + addon)
        else:
            raise ValueError(f'{numerical_feature} has {num_unique_values}, which is more than {NUM_UNIQUE_VALUES_THRESHOLD}, so it should not be converted to a categorical feature')
    
    df_with_renamed_columns = df.rename(columns={numerical_feature: numerical_feature + addon for numerical_feature in numerical_features_to_convert_in_df})
    return df_with_renamed_columns, new_binary_features, new_categorical_features


'''Converts numerical features to categorical features for the numerical features 
that have a number of unique values lesser than `value_count_threshold`. Returns 
a new dataframe that has renamed the old numerical features, and the names of 
these new categorical (or binary if the feature only has two unique values) 
features.'''
def convert_numerical_features_to_categorical_by_threshold(df, numerical_features_to_inspect, value_count_threshold):
    addon = '_cat'    # this string is added onto the end of the original feature name
    new_binary_features = []
    new_categorical_features = []
    numerical_features_converted = []
    for numerical_feature in numerical_features_to_inspect:
        warnings.warn(f'{numerical_feature} not found in the passed in dataframe')
        num_unique_values = len(df[numerical_feature].unique())
        if num_unique_values == 2:
            new_binary_features.append(numerical_feature + addon)
            numerical_features_converted.append(numerical_feature)
        elif num_unique_values <= value_count_threshold:
            new_categorical_features.append(numerical_feature + addon)
            numerical_features_converted.append(numerical_feature)
            
    df_with_renamed_columns = df.rename(columns={numerical_feature: numerical_feature + addon for numerical_feature in numerical_features_converted})
    return df_with_renamed_columns, new_binary_features, new_categorical_features


'''Replaces `rating` with the stand alone rating from `sp_stand_alone`.'''
def replace_rating_with_standalone_rating(df):
    assert 'sp_stand_alone' in df.columns and 'rating' in df.columns
    df = df.copy()
    df.loc[df.sp_stand_alone.isna(), 'sp_stand_alone'] = 'NR'
    df['rating'] = df['rating'].astype('string')    # changed from .astype(str) because it no longer works: https://stackoverflow.com/questions/22231592/pandas-change-data-type-of-series-to-string
    df['sp_stand_alone'] = df['sp_stand_alone'].astype('string')
    df.loc[(df.sp_stand_alone != 'NR'), 'rating'] = df[(df.sp_stand_alone != 'NR')]['sp_stand_alone'].loc[:]
    
    df['rating'] = df['rating'].astype('category')    # convert to categorical dtype in order to use this feature with LightGBM
    return df


'''Reverses the order of the trade history in `df` so that the most recent past trade 
is the last one in the trade_history array (assuming that the trade_history array 
is originally in ascending order of num_seconds_ago with the most recent trade first).'''
def reverse_order_of_trade_history(df, columns=['trade_history']):
    assert set(columns) <= set(df.columns)
    print(f'Reversing the order of the trade history for {columns}')
    df = df.copy()
    reverse = lambda lst: lst[::-1]
    for column in columns:
        df[column] = df[column].apply(reverse)
    return df


'''Adds the features of `quantity`, `last_yield_spread`, and `seconds_ago` of 
the `num_past_trades` previous trades beyond the last trade from the trade 
history. Returns the new dataframe and the new numerical features that have 
been added to the dataframe from the old `df`.'''
def _add_past_trades_info_prior_to_last_trade(df, num_past_trades, features_to_index_in_history):
    assert 'trade_history' in df.columns
    assert len(df.iloc[0]['trade_history']) > num_past_trades, f'{num_past_trades + 1} past trades are requested, but the trade history only contains {len(df.iloc[0]["trade_history"])} trades'
    df = df.copy()
    columns_set = set(df.columns)
    additional_binary_features = []
    additional_non_cat_features = []
    groups = []

    binary_feature_names = {'trade_type1', 'trade_type2'}
    
    df_as_dict = dict()    # create the df as a dict first and then convert it into a DataFrame in order to use .concat when joining to the original dataframe to avoid the following warning: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    for trade_index in range(1, num_past_trades + 1):
        group = []
        feature_prefix = f'{trade_index + 1}last_'
        for feature, feature_index in features_to_index_in_history.items():        # insertion order of the dictionary is preserved for Python v3.7+
            feature_with_prefix = feature_prefix + feature
            group.append(feature_with_prefix)
            assert feature_with_prefix not in columns_set, f'{feature_with_prefix} already exists in the dataframe'
            if feature in binary_feature_names:
                additional_binary_features.append(feature_with_prefix)
            else:
                additional_non_cat_features.append(feature_with_prefix)
            df_as_dict[feature_with_prefix] = df['trade_history'].apply(lambda past_trades: past_trades[trade_index][feature_index])
        groups.append(group)
    return pd.concat((df, pd.DataFrame(df_as_dict)), axis='columns'), additional_binary_features, additional_non_cat_features, groups    # first convert the df_as_dict to a pandas DataFrame and then join it to the original dataframe with .concat


'''Adds the quantity of the previous trade, as well as the features of 
`quantity`, `last_yield_spread`, and `seconds_ago` of the `num_past_trades` 
previous trades beyond the last trade from the trade history. Returns the 
new dataframe and the new numerical features that have been added to the 
dataframe from the old `df`. The boolean variable 
`contains_settlement_date_to_calc_date` handles the case when the feature 
`settlement_date_to_calc_date` is in the trade history.'''
def add_past_trades_info(df, num_past_trades_beyond_last_trade, features_to_index_in_history):
    features_set = set(features_to_index_in_history.keys())
    columns_set = set(df.columns)

    assert 'trade_history' in columns_set

    assert 'last_yield_spread' in columns_set
    assert 'last_seconds_ago' in columns_set

    assert 'last_quantity' not in columns_set
    assert 'last_trade_type1' not in columns_set
    assert 'last_trade_type2' not in columns_set

    new_non_cat_features = ['settlement_date_to_calc_date', 'quantity_diff', 'treasury_spread']
    new_categorical_features = ['calc_day_cat', 'trade_type_past_latest']

    for feature in new_non_cat_features + new_categorical_features:
        if feature in features_set: assert f'last_{feature}' not in columns_set
    
    df = df.copy()
    last_non_cat_features = []
    last_binary_features = []
    
    df['last_quantity'] = df['trade_history'].apply(lambda past_trades: past_trades[0][features_to_index_in_history['quantity']])
    last_non_cat_features.append('last_quantity')

    df['last_trade_type1'] = df['trade_history'].apply(lambda past_trades: past_trades[0][features_to_index_in_history['trade_type1']])
    last_binary_features.append('last_trade_type1')
    df['last_trade_type2'] = df['trade_history'].apply(lambda past_trades: past_trades[0][features_to_index_in_history['trade_type2']])
    last_binary_features.append('last_trade_type2')

    for feature in new_non_cat_features + new_categorical_features:
        if feature in features_set: df[f'last_{feature}'] = df['trade_history'].apply(lambda past_trades: past_trades[0][features_to_index_in_history[feature]])

    group_for_last_trade = ['last_yield_spread', 'last_quantity', 'last_trade_type1', 'last_trade_type2', 'last_seconds_ago']
    if 'settlement_date_to_calc_date' in features_set: group_for_last_trade.append('last_settlement_date_to_calc_date')
    if 'calc_day_cat' in features_set: group_for_last_trade.append('last_calc_day_cat')
    if 'trade_type_past_latest' in features_set: group_for_last_trade.append('last_trade_type_past_latest')
    if 'treasury_spread' in features_set: group_for_last_trade.insert(1, 'last_treasury_spread')
    if 'quantity_diff' in features_set: group_for_last_trade.insert(3, 'last_quantity_diff')

    df, additional_binary_features, additional_non_cat_features, groups = _add_past_trades_info_prior_to_last_trade(df, num_past_trades_beyond_last_trade, features_to_index_in_history)
    return df, \
           last_binary_features + additional_binary_features, \
           last_non_cat_features + additional_non_cat_features, \
           [group_for_last_trade] + groups


'''Encode all features in `features` with an encoder (specified by `encoder_type`) to 
convert the features into integers. Creates another option for unseen values only for 
`categorical_features`. If `label_or_one_hot` is 'label' '''
def encode_and_get_encoders(df, binary_features, categorical_features, encoder_type='label'):
    assert encoder_type == 'label' or encoder_type == 'one_hot'
    df_encoded = df.copy()
    encoders = dict()
    binary_features_set = set(binary_features)
    for feature in binary_features + categorical_features:
        if not is_string_dtype(df_encoded[feature].dtype):
            df_encoded[feature] = df_encoded[feature].astype('string')
        if feature in df_encoded:
            encoder = LabelEncoder() if encoder_type == 'label' or feature in binary_features_set else OneHotEncoder(sparse=False)    # sparse=False returns an array after transforming the values instead of a sparse matrix object
            series = df_encoded[feature]
            if encoder_type == 'one_hot': series = series.to_numpy().reshape(-1, 1)    # need to reshape to avoid this error: `ValueError: Expected 2D array, got 1D array instead...Reshape your data either using array.reshape(-1, 1) if your data has a single feature`
            if feature in binary_features_set:
                series_to_fit = series
            else:    # feature must be a categorical feature
                assert NOT_SEEN_IN_DATA_STRING not in series, f'{NOT_SEEN_IN_DATA_STRING} is one of the feature values of the feature: {feature}'
                series_to_fit = pd.concat([series, pd.Series([NOT_SEEN_IN_DATA_STRING])]) if encoder_type == 'label' else np.append(series, [[NOT_SEEN_IN_DATA_STRING]], axis=0)    # add  a tag for a value that has not been seen in the training data to the end of the series
            encoder.fit(series_to_fit)
            transformed_values = encoder.transform(series)
            df_encoded[feature] = transformed_values if encoder_type == 'label' else list(transformed_values)
            encoders[feature] = encoder
        else:
            warnings.warn(f'{feature} not found in the passed in dataframe')
    return df_encoded, encoders


'''Encode each feature in `df` by iterating through the dictionary `encoders`, where each 
item has a key for the feature name and a value as the corresponding encoder. 
`features_to_exclude` allows certain features to not be encoded.'''
def encode_with_encoders(df, encoders, features_to_exclude=[]):
    if type(features_to_exclude) == str: features_to_exclude = [features_to_exclude]    # if single feature passed in, then convert it to a one item list since `features_to_exclude` is handled as a list later on
    df_encoded = df.copy()
    for feature, encoder in encoders.items():
        if feature not in features_to_exclude:
            if type(encoder) == LabelEncoder: encoder_type = 'label'
            elif type(encoder) == OneHotEncoder: encoder_type = 'one_hot'
            else: raise TypeError(f'Encoder type must be either LabelEncoder or OneHotEncoder, but was instead: {type(encoder)}')

            if not is_string_dtype(df_encoded[feature]): df_encoded[feature] = df_encoded[feature].astype('string')
            unique_values = set(df_encoded[feature].unique())
            encoder_seen_values = encoder.classes_ if encoder_type == 'label' else encoder.categories_[0]
            for seen_value in encoder_seen_values:
                unique_values.discard(seen_value)
            unseen_values = list(unique_values)    # update the name to be more representative of the final list
            replaced_feature_values = df_encoded[feature].replace(unseen_values, NOT_SEEN_IN_DATA_STRING)
            if encoder_type == 'one_hot': replaced_feature_values.to_numpy().reshape(-1, 1)    # need to reshape to avoid this error: `ValueError: Expected 2D array, got 1D array instead...Reshape your data either using array.reshape(-1, 1) if your data has a single feature`
            try:
                transformed_values = encoder.transform(replaced_feature_values)
                df_encoded[feature] = transformed_values if encoder_type == 'label' else list(transformed_values)
            except ValueError:    # output a more informative error message
                raise ValueError(f'{feature} has a value that was not found in its corresponding encoder.')
    return df_encoded


'''Adds a column with name `column_name` and values `values` to `df`.'''
def add_column(df, column_name, values):
    assert column_name not in df.columns
    assert len(df) == len(values)
    df = df.copy()
    df[column_name] = values
    return df


'''Add each column from `columns` into `df`, where each column exists in `other_df`.'''
def add_columns_into_df_from_other_df(df, columns, other_df):
    for column_name in columns:
        assert column_name in other_df
        df = add_column(df, column_name, other_df[column_name])
    return df


'''Perform z score normalization for each feature in `numerical_features`. Use the 
means and standard deviations from `mean_std_dict` if `mean_std_dict` is not `None`.'''
def z_score_normalization(df, numerical_features, mean_std_dict=None):
    df = df.copy()
    if mean_std_dict is None:
        mean_std_dict = dict()
        for feature in numerical_features:
            mean, std = df[feature].mean(), df[feature].std()
            mean_std_dict[feature] = (mean, std)
    
    for feature in numerical_features:
        mean, std = mean_std_dict[feature]
        df[feature] = (df[feature] - mean) / std
    return df, mean_std_dict
