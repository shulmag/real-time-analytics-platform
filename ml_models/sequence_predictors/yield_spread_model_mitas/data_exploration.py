import warnings

import matplotlib.pyplot as plt
import pandas as pd

# importing from parent directory: https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import sys
sys.path.insert(0,'../')

from rating_model_mitas.data_exploration import _get_new_keys


'''Returns a frame where each row is a feature from `features`, the largest 
value in `df` for that feature, the smallest value for that feature, the 
dynamic range, the `identifier` (e.g., 'rtrs_control_number') for the 
largest value, and the `identifier` for the smallest value. Note that we 
assume `features` are all numerical where all the values are positive.'''
def compute_dynamic_ranges(df, features, identifier):
    columns_set = set(df.columns)
    assert identifier in columns_set
    
    features_in_result_df = []
    num_unique_values_in_result_df = []
    largest_values_in_result_df = []
    smallest_values_in_result_df = []
    dynamic_ranges_in_result_df = []
    largest_value_identifiers_in_result_df = []
    smallest_value_identifiers_in_result_df = []
    for feature in features:
        assert feature in columns_set, f'{feature} not in the dataframe'
        features_in_result_df.append(feature)
        values_with_identifier = df[[identifier, feature]]
        num_unique_values_in_result_df.append(values_with_identifier[feature].nunique())

        largest_value_with_identifier = values_with_identifier[values_with_identifier[feature] == max(values_with_identifier[feature])].iloc[0]
        largest_values_in_result_df.append(largest_value_with_identifier[feature])
        largest_value_identifiers_in_result_df.append(largest_value_with_identifier[identifier])
        
        smallest_value_with_identifier = values_with_identifier[values_with_identifier[feature] == min(values_with_identifier[feature])].iloc[0]
        smallest_value = smallest_value_with_identifier[feature]
        smallest_values_in_result_df.append(smallest_value)
        if smallest_value <= 0:
            warnings.warn(f'The smallest value for {feature} is {smallest_value} which is nonpositive.')
        smallest_value_identifiers_in_result_df.append(smallest_value_with_identifier[identifier])
        
        dynamic_range = largest_values_in_result_df[-1] / smallest_value if smallest_value != 0 else float('inf')    # removes the RuntimeError: Division by zero
        dynamic_ranges_in_result_df.append(dynamic_range)

    return pd.DataFrame({'feature': features_in_result_df, 
                         'num_unique_values': num_unique_values_in_result_df, 
                         'largest_value': largest_values_in_result_df,
                         'smallest_value': smallest_values_in_result_df, 
                         'dynamic_range': dynamic_ranges_in_result_df, 
                         f'largest_value_{identifier}': largest_value_identifiers_in_result_df,
                         f'smallest_value_{identifier}': smallest_value_identifiers_in_result_df})


'''Splits a string into a new line for every word. Helps '''
def _split_into_multiple_lines(string):
        string = string.replace(' ', '\n')    # replace all spaces with a new line
        return string.replace('\n&\n', ' &\n')    # replace all '\n&\n' with just ' &\n' to not have a line with just '&'


'''Plots feature distributions as histograms for numerical features, and bar plots 
for categorical features (for the `k` feature values with the largest counts).'''
def plot_feature_distributions(df, numerical_features, categorical_features, figsize, k=8):
    columns_set = set(df.columns)
    for feature in numerical_features:
        assert feature in columns_set, f'{feature} not in the dataframe'
        plt.figure(figsize=figsize)
        plt.hist(df[feature])
        plt.ylabel('counts')
        plt.title(f'{feature}: numerical')
        plt.show()

    for feature in categorical_features:
        assert feature in columns_set, f'{feature} not in the dataframe'
        keys = []
        counts = []
        count_for_other = 0
        for value, count in df[feature].value_counts().iteritems():    # assumes that the iteration occurs in descending order of count
            if len(keys) > k:
                count_for_other += count
            else:
                keys.append(value)
                counts.append(count)
        if count_for_other > 0:
            keys.append('other')
            counts.append(count_for_other)
        keys = _get_new_keys(feature, keys)
        keys = [_split_into_multiple_lines(key) for key in keys]
        plt.figure(figsize=figsize)
        plt.bar(keys, counts)
        plt.ylabel('counts')
        plt.title(f'{feature}: categorical')
        plt.show()


'''Returns a `middle_percent` percent of data for `feature` centered at the median of that 
`feature`'s values. E.g., if `middle_percent == 50` and `feature == 'yield_spread'`, then 
this function returns all rows in `df` such that the `yield_spread` value is between the 
25th and 75th percentile.'''
def get_middle_percent(df, middle_percent, feature):
    if middle_percent == 100:
        return df
    
    assert 0 < middle_percent < 100
    assert middle_percent // 2 * 2 == middle_percent, '`middle_percent` must be divisible by 2 in order for the mask to work properly due to the integer typecasting'
    lower = (50 - middle_percent / 2) / 100
    upper = (50 + middle_percent / 2) / 100
    info = df[feature].describe(percentiles=sorted({lower, upper, 0.25, 0.5, 0.75}))
    print(info)
    mask = (df[feature] >= info[f'{int(lower * 100)}%']) & (df[feature] <= info[f'{int(upper * 100)}%'])
    return df[mask]


'''This function plots pairs of numerical feature value and `select_numerical_feature` value as 
a scatterplot.'''
def _numerical_feature_against_select_numerical_feature_plotter(df, feature, select_numerical_feature, figsize):
    feature_values = df[feature]
    select_numerical_feature_values = df[select_numerical_feature]
    plt.figure(figsize=figsize)
    plt.plot(feature_values, select_numerical_feature_values, 'bo')
    plt.xlabel(feature)
    plt.ylabel(select_numerical_feature)
    plt.title(f'{feature}: numerical')
    plt.show()


'''The values of `select_numerical_feature` corresponding to the top `k` keys of `categorical_feature` 
are grouped together, and those values which are are not in the top `k` keys are grouped together under 
`overflow_key_name`.'''
def _get_top_k_categories_and_values(df, categorical_feature, select_numerical_feature, k, overflow_key_name):
    keys = df[categorical_feature].unique()

    with warnings.catch_warnings():    # this suppresses the warning that occurs when `keys` has numeric types, and numpy is unsure how to compare `other` to the numeric types
        warnings.simplefilter(action='ignore', category=FutureWarning)
        assert overflow_key_name not in keys, f'We assume the value `{overflow_key_name}` will not be a value of the feature: {categorical_feature}'

    more_than_k_values = False
    if len(keys) > k:
        more_than_k_values = True
        k_most_common_feature_values = list(df[categorical_feature].value_counts().index[:k])
        k_most_common_feature_values_set = set(k_most_common_feature_values)
    
    select_numerical_feature_values = []
    if more_than_k_values:
        keys = []
        select_numerical_feature_values_for_other = []
        for feature_value, group in df.groupby([categorical_feature]):
            if feature_value in k_most_common_feature_values_set:
                keys.append(feature_value)
                select_numerical_feature_values.append(group[select_numerical_feature].values)
            else:    # feature_value data should be put in the `other` category
                select_numerical_feature_values_for_other.extend(group[select_numerical_feature].values)
        keys.append('other')
        select_numerical_feature_values.append(select_numerical_feature_values_for_other)
    else:
        for key in keys:
            select_numerical_feature_values.append(df[df[categorical_feature] == key][select_numerical_feature].values)
    return _get_new_keys(categorical_feature, keys), select_numerical_feature_values


'''This function plots a categorical / binary feature as a histogram against `select_numerical_feature` 
for the `k` categories with the most values.'''
def _categorical_feature_against_select_numerical_feature_plotter(df, feature, select_numerical_feature, figsize, k=8):
    assert k < 10, 'A value of `k` larger than 10 causes colors to repeat on the plotter'
    OVERFLOW_KEY_NAME = 'other'
    df = df[[feature, select_numerical_feature]]
    keys, select_numerical_feature_values = _get_top_k_categories_and_values(df, feature, select_numerical_feature, k, OVERFLOW_KEY_NAME)

    plt.figure(figsize=figsize)
    plt.hist(select_numerical_feature_values, label=keys)
    plt.xlabel(select_numerical_feature)
    plt.ylabel('counts')
    plt.title(f'{feature}: categorical')
    plt.legend()
    plt.show()


'''Iterates through the columns of `df` and plots the columns according to whether the 
feature is categorical or numerical.'''
def plot_all_features_against_select_numerical_feature(df, binary_features, categorical_features, non_cat_features, select_numerical_feature='yield_spread', figsize=(20, 10)):
    binary_and_categorical_features_set = set(binary_features + categorical_features)
    non_cat_features_set = set(non_cat_features)
    columns_not_found = []
    for column in df.columns:
        if column != select_numerical_feature:
            if column in binary_and_categorical_features_set:
                _categorical_feature_against_select_numerical_feature_plotter(df, column, select_numerical_feature, figsize)
            elif column in non_cat_features_set:
                _numerical_feature_against_select_numerical_feature_plotter(df, column, select_numerical_feature, figsize)
            else:
                columns_not_found.append(column)
    if columns_not_found:
        print(f'No plotter found for {columns_not_found}, as the feature does not appear in BINARY, CATEGORICAL_FEATURES, or NON_CAT_FEATURES')


'''Returns a new dataframe which applys each condition in `condition_as_strings` to 
`df`. `groups_of_conditions_as_strings` should be a list where each sublist has items of the form: 
`df['quantity'] == 5`, where the condition is a string that would be evaluated as code. Afterwards, 
each group of conditions is concatenated together. This can be thought of as having OR statements 
between groups and having AND statements within each group.'''
def filter_data_with_string_conditions(df, groups_of_conditions_as_strings):
    if not groups_of_conditions_as_strings:
        return df
    frames = []
    for conditions_as_strings in groups_of_conditions_as_strings:
        conditions_as_strings = ['(' + condition + ')' for condition in conditions_as_strings]
        conditions_as_one_string = ' & '.join(conditions_as_strings)
        print(conditions_as_one_string)
        frames.append(df[eval(conditions_as_one_string)])
    return pd.concat(frames)