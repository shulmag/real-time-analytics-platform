from distutils.log import ERROR
import warnings

import matplotlib.pyplot as plt
import numpy as np

import torch

from ficc.utils.auxiliary_functions import calculate_dollar_error

import sys
sys.path.insert(0,'../')

from yield_spread_model_mitas.data_exploration import filter_data_with_string_conditions, \
                                                      _split_into_multiple_lines, \
                                                      _get_top_k_categories_and_values, \
                                                      _categorical_feature_against_select_numerical_feature_plotter, \
                                                      get_middle_percent
from yield_spread_model_mitas.train import l1_loss_func, l1_loss_func_numpy
from yield_spread_model_mitas.datasets_pytorch import EmbeddingsDataset, TradeHistoryDataset


NUM_ROWS_TO_PROCESS_AT_ONE_TIME = 1_000_000    # underscores are just to format the number to be easy to read (has no impact on the value, equivalent to 1000000)

ERROR_COLUMN_NAME = 'error (predictions - labels)'    # used as the column name added to the original dataframe which stores the error values
ERROR_AS_DOLLAR_VALUE_COLUMN_NAME = 'error as dollar value (ytw error * quantity * years until calc date)'    # used as the column name added to the original dataframe which stores the error values
YEARS_TO_CALC_DATE_COLUMN_NAME = 'years_to_calc_date'    # used as the column name when putting the years to calc_date into the dataframe

'''Checks that `df_encoded` is the encoded version of `df`, while allowing 
`df` to have additional columns such as identifiers.'''
def _check_df_and_df_encoded_are_similar(df, df_encoded):
    df_encoded_columns_set = set(df_encoded.columns)
    assert len(df) == len(df_encoded) and df_encoded_columns_set.issubset(set(df.columns)), 'The dataframe and the encoded version did not originate from the same dataframe'
    assert 'trade_history' in df_encoded_columns_set, 'trade_history was not found as a feature in the dataframe'
    assert 'yield_spread' in df_encoded_columns_set, 'yield_spread was not found as a feature in the dataframe'


'''Gets predictions from `model` applied to `df_encoded`. Need to process the 
data in batches in order to prevent the jupyter kernel from dying due to too 
many row at once.'''
def _get_predictions_from_model(model, df_encoded, categorical_features):
    all_predictions = []
    num_rows_processed = 0
    num_rows_total = len(df_encoded)
    while num_rows_processed < num_rows_total:
        sliced_df_encoded = df_encoded[num_rows_processed:num_rows_processed + NUM_ROWS_TO_PROCESS_AT_ONE_TIME]
        train_data_with_trade_history_as_embeddings_dataset = EmbeddingsDataset(sliced_df_encoded.drop(columns=['trade_history']), categorical_features)
        train_data_with_trade_history_as_trade_history_dataset = TradeHistoryDataset(sliced_df_encoded[['trade_history', 'yield_spread']])
        all_predictions.append(model(train_data_with_trade_history_as_embeddings_dataset.inputs_categorical, 
                                     train_data_with_trade_history_as_embeddings_dataset.inputs_binary_and_continuous, 
                                     train_data_with_trade_history_as_trade_history_dataset.inputs).detach().numpy())    # processing these results as tensors uses too much memory and causes the kernel to die
        num_rows_processed += NUM_ROWS_TO_PROCESS_AT_ONE_TIME
    return np.squeeze(np.concatenate(all_predictions))


'''Gets errors from `predictions` and `labels`. Need to process the 
data in batches in order to prevent the jupyter kernel from dying 
due to too many row at once. NOT USED since working directly with 
numpy does not require processing the data in batches.'''
def _get_errors_from_predictions_and_labels(predictions, labels):
    errors = []
    num_rows_processed = 0
    num_rows_total = len(labels)
    while num_rows_processed < num_rows_total:
        sliced_predictions = predictions[num_rows_processed:num_rows_processed + NUM_ROWS_TO_PROCESS_AT_ONE_TIME]
        sliced_labels = labels[num_rows_processed:num_rows_processed + NUM_ROWS_TO_PROCESS_AT_ONE_TIME]
        errors.append(l1_loss_func(torch.tensor(sliced_predictions), torch.tensor(sliced_labels)).detach().numpy())    # processing these results as tensors uses too much memory and causes the kernel to die
        num_rows_processed += NUM_ROWS_TO_PROCESS_AT_ONE_TIME
    errors = np.concatenate(errors)
    return errors


'''Returns the indices in `array` where the values are larger than 
`threshold`. `threshold_as_percentile` is a boolean that determines 
whether `threshold` is a value or a percentile.'''
def _get_indices_for_threshold(array, threshold, threshold_as_percentile=False):
    if threshold_as_percentile:
        assert 0 < threshold < 100, f'threshold: {threshold} must be between 0 and 100 since it represents a percentage'
        threshold = np.quantile(array, threshold / 100)
    return np.where(array >= threshold)[0]


'''Returns a subset of `df` where every item has `feature` above `threshold`. 
`threshold_as_percentile` is a boolean that determines whether `threshold` 
is a value or a percentile. `filtering_conditions` are string clauses that 
allow the resulting dataframe to be filtered; for reference, see 
`filter_data_with_string_conditions(...)`.'''
def feature_above_threshold(df, feature, threshold, threshold_as_percentile=False, filtering_conditions=None):
    indices_above_threshold = _get_indices_for_threshold(df[feature].values, threshold, threshold_as_percentile)
    return filter_data_with_string_conditions(df.iloc[indices_above_threshold], filtering_conditions)


'''Returns a dataframe with the errors above `threshold`. If the boolean 
`threshold_as_percentile` is set to `True`, then `threshold` will be interpreted 
as the top `threshold` percent of all thee errors. If `use_price_as_metric` is 
set to True, then the error is computed in terms of price, i.e., price error = 
yield spread error (in basis points) * duration of the bond * quantity, and 
this is the error metric used. Need to process the losses in batches in order 
to prevent the jupyter kernel from dying due to too many rows at once.'''
def absolute_errors_above_threshold(df, df_encoded, categorical_features, model, threshold, threshold_as_percentile=False, error_as_dollar_value=False, filtering_conditions=None):
    _check_df_and_df_encoded_are_similar(df, df_encoded)
    error_column_name = ERROR_AS_DOLLAR_VALUE_COLUMN_NAME if error_as_dollar_value else ERROR_COLUMN_NAME
    df, error_column_name = _add_error_column(df, df_encoded, categorical_features, model, error_column_name, func_to_apply_name='abs', error_as_dollar_value=error_as_dollar_value)
    indices_of_errors = _get_indices_for_threshold(df[error_column_name], threshold, threshold_as_percentile)
    return filter_data_with_string_conditions(df.iloc[indices_of_errors], filtering_conditions), error_column_name


'''Plot predictions against labels. Optional argument `size_feature` is 
used to determine the size of each point in the scatterplot. If `size_feature` 
is None, then all points are the same size, but if it is a feature that exists 
in `df`, then the value of the feature determines how large the point is.'''
def plot_predictions_against_labels(df, df_encoded, categorical_features, model, figsize, size_feature=None):
    _check_df_and_df_encoded_are_similar(df, df_encoded)
    sizes = 10
    if size_feature is not None:
        if size_feature not in df.columns:
            warnings.warn(f'size_feature: {size_feature} is not a feature in the dataframe, and so no feature will be used to size the points.')
            size_feature = None
        else:
            max_feature_value = max(df[size_feature])
            sizes = df[size_feature].to_numpy() / max_feature_value * sizes

    predictions = _get_predictions_from_model(model, df_encoded, categorical_features)
    labels = df_encoded['yield_spread'].values
    plt.figure(figsize=figsize)
    # plt.title(f'Mean Absolute Error: {round(np.mean(np.abs(predictions - labels)), 3)}')   # this line causes the kernel to die from processing too much data at once
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    plt.scatter(predictions, labels, s=sizes)
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)
    plt.show()


'''Adds the error column to `df` from `predictions` and `labels` by applying 
`func_to_apply` and storing this result into `error_column_name`.'''
def _add_error_column_from_predictions_and_labels(df, predictions, labels, func_to_apply, error_column_name, error_as_dollar_value=False):
    columns_set = set(df.columns)
    assert error_column_name not in columns_set, f'Trying to put error into a new `{error_column_name}` column, but `{error_column_name}` already exists in the dataframe'
    if error_as_dollar_value:
        errors = calculate_dollar_error(df, predictions)
    else:
        errors = func_to_apply(predictions - labels)
    df = df.copy()
    df[error_column_name] = errors
    return df


'''Add error column with name `error_column_name` to dataframe and return the new dataframe. 
`func_to_apply` is a function applied to the errors (e.g., `func_to_apply = 'abs'` would 
provide the absolute values of the true errors).'''
def _add_error_column(df, df_encoded, categorical_features, model, error_column_name, func_to_apply_name=None, error_as_dollar_value=False):
    SUPPORTED_FUNCS = {None: lambda vec: vec, 
                       'abs': lambda vec: np.absolute(vec), 
                       'squared': lambda vec: np.square(vec)}
    assert func_to_apply_name is None or func_to_apply_name in SUPPORTED_FUNCS
    if func_to_apply_name is not None:
        error_column_name = func_to_apply_name + ' ' + error_column_name

    predictions = _get_predictions_from_model(model, df_encoded, categorical_features)
    df = _add_error_column_from_predictions_and_labels(df, predictions, df['yield_spread'].values, SUPPORTED_FUNCS[func_to_apply_name], error_column_name, error_as_dollar_value)
    return df, error_column_name


'''Helper function which groups the errors for a categorical `feature` and 
outputs a bar plot which shows the average error corresponding to a  
particular feature value.'''
def _categorical_feature_average_error_by_value_plotter(df_with_errors, feature, figsize, k=8, error_column_name=ERROR_COLUMN_NAME):
    keys, error_values = _get_top_k_categories_and_values(df_with_errors, feature, error_column_name, k, 'other')
    error_values = [np.mean(error_values_for_range) for error_values_for_range in error_values]
    plt.figure(figsize=figsize)
    keys = [_split_into_multiple_lines(key) for key in keys]
    plt.bar(keys, error_values)
    plt.xlabel(feature)
    plt.ylabel(f'Avg {error_column_name}')
    plt.show()


'''Groups the errors for a numerical `feature` into `num_buckets` buckets 
with equal number of items.'''
def _get_numerical_ranges_and_errors(df_with_errors, feature, num_buckets, error_column_name):
    step = 100 / num_buckets
    feature_series = df_with_errors[feature]
    percentiles = [0] + [step * (i + 1) for i in range(num_buckets)]
    ranges = sorted({feature_series.quantile(percentile / 100) for percentile in percentiles})    # perform set operation to make sure that all quantile values are unique and sorted operation it in the right order
    ranges = [(ranges[idx], ranges[idx + 1]) for idx in range(len(ranges) - 1)]
    error_values = [df_with_errors[feature_series.between(begin, end, inclusive='both' if idx == len(ranges) - 1 else 'left')][error_column_name] for idx, (begin, end) in enumerate(ranges)]    # `between(...)` has an argument `inclusive` which determines if we include values at the end points
    keys = [f'{round(begin, 3)} -- {round(end, 3)}' for begin, end in ranges]
    return keys, error_values


'''Helper function which outputs a bar plot which shows the average error 
corresponding to a quantile of the values of numerical `feature`.'''
def _numerical_feature_average_error_by_value_plotter(df_with_errors, feature, figsize, num_buckets=10, error_column_name=ERROR_COLUMN_NAME):
    keys, error_values = _get_numerical_ranges_and_errors(df_with_errors, feature, num_buckets, error_column_name)
    error_values = [np.mean(error_values_for_range) for error_values_for_range in error_values]

    plt.figure(figsize=figsize)
    plt.bar(keys, error_values)
    plt.xlabel(f'{feature} (value)')
    plt.ylabel(f'Avg {error_column_name}')
    plt.show()


'''Helper function which outputs a histogram for each of the corresponding 
categorical `feature` values, with a maximum of `k` feature values.'''
def _categorical_feature_histogram_of_error_by_value_plotter(df_with_errors, feature, figsize, k=8, error_column_name=ERROR_COLUMN_NAME):
    return _categorical_feature_against_select_numerical_feature_plotter(df_with_errors, feature, error_column_name, figsize, k)


'''Helper function which outputs a histogram which shows the error distribution 
corresponding to a quantile of the values of numerical `feature`.'''
def _numerical_feature_histogram_of_error_by_value_plotter(df_with_errors, feature, figsize, num_buckets=10, error_column_name=ERROR_COLUMN_NAME):
    keys, error_values = _get_numerical_ranges_and_errors(df_with_errors, feature, num_buckets, error_column_name)
    
    plt.figure(figsize=figsize)
    plt.hist(error_values, label=keys)
    plt.xlabel(error_column_name)
    plt.ylabel('counts')
    plt.title(f'{feature}: numerical')
    plt.legend()
    plt.show()


'''Helper function to make DRY `plot_average_error_by_feature_value` and 
`plot_histogram_of_error_by_feature_value`.'''
def _plot_error_by_feature_value(df, df_encoded, binary_features, categorical_features, non_cat_features, model, figsize, func_to_apply_name, middle_percent_to_keep, plotter_type):
    _plot_error_by_feature_value_functions = {'average': {'categorical': _categorical_feature_average_error_by_value_plotter, 
                                                          'numerical': _numerical_feature_average_error_by_value_plotter}, 
                                              'histogram': {'categorical': _categorical_feature_histogram_of_error_by_value_plotter, 
                                                            'numerical': _numerical_feature_histogram_of_error_by_value_plotter}}
    assert plotter_type in _plot_error_by_feature_value_functions
    plotter_functions = _plot_error_by_feature_value_functions[plotter_type]

    df_with_error_column, error_column_name = _add_error_column(df, df_encoded, categorical_features, model, func_to_apply_name)
    df_with_error_column = get_middle_percent(df_with_error_column, middle_percent_to_keep, error_column_name)    # keep only the middle percent of the errors (in order to have readable histogram plots)

    binary_and_categorical_features_set = set(binary_features + categorical_features)
    non_cat_features_set = set(non_cat_features)
    columns_not_found = []
    for column in df.columns:
        if column in binary_and_categorical_features_set:
            plotter_functions['categorical'](df_with_error_column, column, figsize, error_column_name=error_column_name)
        elif column in non_cat_features_set:
            plotter_functions['numerical'](df_with_error_column, column, figsize, error_column_name=error_column_name)
        else:
            columns_not_found.append(column)
    if columns_not_found:
        print(f'No plotter found for {columns_not_found}, as the feature does not appear in BINARY, CATEGORICAL_FEATURES, or NON_CAT_FEATURES')


'''For each feature in `df`, we plot a bar for each feature value that represents 
the average error. For numerical features, we group the feature into quantiles. 
`func_to_apply_name` is a function applied to the errors (e.g., `func_to_apply_name = 'abs'` 
would take the absolute values of the errors).'''
def plot_average_error_by_feature_value(df, df_encoded, binary_features, categorical_features, non_cat_features, model, figsize, middle_percent_to_keep=100, func_to_apply_name=None):
    return _plot_error_by_feature_value(df, df_encoded, binary_features, categorical_features, non_cat_features, model, figsize, func_to_apply_name, middle_percent_to_keep, 'average')


'''For each feature in `df`, we plot a histogram for each feature value for the
error distribution. For numerical features, we group the feature into quantiles. 
`func_to_apply_name` is a function applied to the errors (e.g., `func_to_apply_name = 'abs'` 
would take the absolute values of the errors).'''
def plot_histogram_of_error_by_feature_value(df, df_encoded, binary_features, categorical_features, non_cat_features, model, figsize, middle_percent_to_keep=100, func_to_apply_name=None):
    return _plot_error_by_feature_value(df, df_encoded, binary_features, categorical_features, non_cat_features, model, figsize, func_to_apply_name, middle_percent_to_keep, 'histogram')
