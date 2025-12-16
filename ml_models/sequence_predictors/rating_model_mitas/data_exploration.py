import warnings
import matplotlib.pyplot as plt
import pandas as pd

from ficc.utils.auxiliary_variables import PURPOSE_CLASS_DICT, PURPOSE_SUB_CLASS_DICT, MUNI_ISSUE_TYPE_DICT, SALE_TYPE_DICT, ORIG_INSTRUMENT_ENHANCEMENT_TYPE_DICT, OTHER_ENHANCEMENT_TYPE_DICT, CAPITAL_TYPE_DICT, ASSET_CLAIM_CODE_DICT, MUNI_SECURITY_TYPE_DICT, USE_OF_PROCEEDS_DICT, SECURED_DICT, STATE_TAX_STATUS_DICT, EVENT_CODE_DICT


'''This dictionary maps a feature name to the corresponding dictionary where the 
dictionary is a mapping between the numerical code in the data for that feature 
and its english meaning.'''
_feature_to_dict = {'asset_claim_code': ASSET_CLAIM_CODE_DICT, 
                    'capital_type': CAPITAL_TYPE_DICT, 
                    'muni_issue_type': MUNI_ISSUE_TYPE_DICT, 
                    'orig_instrument_enhancement_type': ORIG_INSTRUMENT_ENHANCEMENT_TYPE_DICT, 
                    'other_enhancement_type': OTHER_ENHANCEMENT_TYPE_DICT, 
                    'purpose_class': PURPOSE_CLASS_DICT, 
                    'purpose_sub_class': PURPOSE_SUB_CLASS_DICT, 
                    'sale_type': SALE_TYPE_DICT, 
                    'muni_security_type': MUNI_SECURITY_TYPE_DICT, 
                    'use_of_proceeds': USE_OF_PROCEEDS_DICT, 
                    'secured': SECURED_DICT, 
                    'state_tax_status': STATE_TAX_STATUS_DICT, 
                    'most_recent_event_code': EVENT_CODE_DICT}


'''This function plots a numerical feature as a histogram against each rating in 
`ratings`.'''
def _numerical_feature_rating_plotter(df, feature, ratings, figsize):
    df = df[[feature, 'rating']]
    values = [df[df['rating'] == rating][feature].values for rating in ratings]
    plt.figure(figsize=figsize)
    plt.hist(values, label=ratings)
    plt.ylabel('counts')
    plt.title(f'{feature}: numerical')
    plt.legend()
    plt.show()


'''This function changes the labels from the number to the meaning for data interpretation 
during plotting. For example, a purpose class of 1.0 will be changed to 'Authority'.'''
def _get_new_keys(feature, keys):
    if feature not in _feature_to_dict:
        return [str(key) for key in keys]

    new_keys = []
    feature_dict = _feature_to_dict[feature]
    for key in keys:
        if key == 0 and key not in feature_dict:
            new_keys.append('null')
        elif key != 'other':
            if type(key) == str:    # this means that `key` was already passed through `feature_dict`
                new_keys.append(key)
            else:
                new_keys.append(feature_dict[key])
        else:
            new_keys.append('other')
    return [str(key) for key in new_keys]


'''This function plots a categorical / binary feature as a histogram against each 
rating in `ratings` for the first `k` categories.'''
def _categorical_feature_rating_plotter(df, feature, ratings, figsize, k=8):
    assert k < 10, 'A value of `k` larger than 10 causes colors to repeat on the plotter'

    df = df[[feature, 'rating']]
    keys = df[feature].unique()

    with warnings.catch_warnings():    # this suppresses the warning that occurs when `keys` has numeric types, and numpy is unsure how to compare `other` to the numeric types
        warnings.simplefilter(action='ignore', category=FutureWarning)
        assert 'other' not in keys, f'We assume the value `other` will not be a value of the feature: {feature}'
        
    more_than_k_values = False
    if len(keys) > k:
        more_than_k_values = True
        k_most_common_feature_values = list(df[feature].value_counts().index[:k])
        keys = k_most_common_feature_values + ['other']

    values = [[] for _ in range(len(keys))]
    key_set = set(keys)
    for rating in ratings:
        df_rating = df[df['rating'] == rating][feature].value_counts()
        for idx, key in enumerate(keys):
            if key != 'other':
                if key not in df_rating:
                    value = 0
                else:
                    value = df_rating[key]
                values[idx].append(value)
        
        if more_than_k_values:
            total = 0
            for key in df_rating.index:
                if key not in key_set:
                    total += df_rating[key]
            values[-1].append(total)

    keys = _get_new_keys(feature, keys)

    pd.DataFrame(dict(zip(keys, values)), index=ratings).plot.bar(title=f'{feature}: categorical / binary', rot=0, figsize=figsize)


'''This function  changes the labels from the number to the meaning for data interpretation. 
For example, a purpose class of 1.0 will be changed to 'Authority'.'''
def data_enumerations(df):
    def get_feature_from_dict(column):
        feature_dict = _feature_to_dict[column]
        def func(key):
            if key == 0 and key not in feature_dict:
                return 'null'
            elif key not in feature_dict:
                print(f'The value {key} was not found for {column} and was replaced with "not found"')
                return 'not found'
            else:
                return feature_dict[key]
        return func

    df = df.copy()
    for column in df.columns:
        if column in _feature_to_dict:
            df[column] = df[column].apply(get_feature_from_dict(column))
    return df


'''Iterates through the columns of `df` and plots the columns according to whether the 
feature is categorical or numerical, against the ratings in order of `ratings_order`.'''
def plot_all_features_against_ratings(df, binary_features, categorical_features, non_cat_features, ratings_order, figsize=(20, 10)):
    binary_and_categorical_features_set = set(binary_features + categorical_features)
    non_cat_features_set = set(non_cat_features + ['yield_spread'])
    columns_not_found = []
    for column in df.columns:
        if 'rating' not in column:
            if column in binary_and_categorical_features_set:
                _categorical_feature_rating_plotter(df, column, ratings_order, figsize)
            elif column in non_cat_features_set:
                _numerical_feature_rating_plotter(df, column, ratings_order, figsize)
            else:
                columns_not_found.append(column)
    if columns_not_found:
        print(f'No plotter found for {columns_not_found}, as none appear in BINARY, CATEGORICAL_FEATURES, or NON_CAT_FEATURES')