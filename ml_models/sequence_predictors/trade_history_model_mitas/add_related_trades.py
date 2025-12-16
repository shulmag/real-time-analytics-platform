from collections import defaultdict
import os

import datetime
import time

import numpy as np
import pandas as pd

from yield_spread_model_mitas.data_prep import encode_and_get_encoders

from trade_history_model_mitas.data_prep import convert_trade_type_encoding_to_actual

from ficc.utils.auxiliary_variables import VERY_LARGE_NUMBER, NUM_OF_DAYS_IN_YEAR
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.trade_dict_to_list import quantity_diff
from ficc.utils.related_trade import append_recent_trade_data, get_appended_feature_name
from ficc.utils.auxiliary_functions import flatten
from ficc.utils.trade_dict_to_list_mappings import TRADE_TYPE_MAPPING, TRADE_TYPE_CROSS_PRODUCT_MAPPING


DEFAULT_VALUES = defaultdict(int)


get_neighbor_feature = lambda feature: lambda curr, neighbor: neighbor[feature]


RELATED_TRADE_FEATURE_FUNCTIONS = {'yield_spread': get_neighbor_feature('yield_spread'), 
                                   'treasury_spread': get_neighbor_feature('ficc_treasury_spread'), 
                                   'quantity': get_neighbor_feature('quantity'), 
                                   'quantity_diff': lambda curr, neighbor: quantity_diff(10 ** neighbor['quantity'] - 10 ** curr['quantity']), 
                                   'trade_type1': lambda curr, neighbor: TRADE_TYPE_MAPPING[neighbor['trade_type']][0], 
                                   'trade_type2': lambda curr, neighbor: TRADE_TYPE_MAPPING[neighbor['trade_type']][1], 
                                   'seconds_ago': lambda curr, neighbor: np.log10(1 + (curr['trade_datetime'] - neighbor['trade_datetime']).total_seconds()), 
                                   'settlement_date_to_calc_date': lambda curr, neighbor: np.log10(1 + diff_in_days_two_dates(neighbor['calc_date'], neighbor['settlement_date'], convention='exact')), 
                                   'calc_day_cat': get_neighbor_feature('calc_day_cat'), 
                                   'trade_type_past_latest': lambda curr, neighbor: TRADE_TYPE_CROSS_PRODUCT_MAPPING[neighbor['trade_type'] + curr['trade_type']], 
                                   'same_day': lambda curr, neighbor: int(neighbor['trade_date'] == curr['trade_date'])}


RELATED_TRADE_BINARY_FEATURES = ['same_day']
RELATED_TRADE_CATEGORICAL_FEATURES = ['calc_day_cat', 'trade_type_past_latest']
RELATED_TRADE_NON_CAT_FEATURES = ['yield_spread', 'treasury_spread', 'quantity', 'quantity_diff', 'seconds_ago', 'settlement_date_to_calc_date']
RELATED_AUXILIARY_FEATURES = ['trade_type1', 'trade_type2']

assert set(RELATED_TRADE_BINARY_FEATURES + RELATED_TRADE_CATEGORICAL_FEATURES + RELATED_TRADE_NON_CAT_FEATURES + RELATED_AUXILIARY_FEATURES) == set(RELATED_TRADE_FEATURE_FUNCTIONS.keys())


RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED = 'rating_without_+-_b_nr_combined'
DAYS_TO_MATURITY_CATEGORICAL = 'days_to_maturity_categorical'
DAYS_TO_CALL_CATEGORICAL = 'days_to_call_categorical'
COUPON_CATEGORICAL = 'coupon_categorical'
COUPON_CATEGORICAL_SUDHAR = 'coupon_categorical_sudhar'
PURPOSE_CLASS_TOP_VALUES = 'purpose_class_top_values'
MUNI_SECURITY_TYPE_TOP_VALUES = 'muni_security_type_top_values'
TRADE_DATETIME_DAY = 'trade_datetime_day'
QUANTITY_CATEGORICAL = 'quantity_categorical'
quantized_features = [RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, 
                      DAYS_TO_MATURITY_CATEGORICAL, 
                      DAYS_TO_CALL_CATEGORICAL, 
                      COUPON_CATEGORICAL, 
                      COUPON_CATEGORICAL_SUDHAR, 
                      PURPOSE_CLASS_TOP_VALUES, 
                      MUNI_SECURITY_TYPE_TOP_VALUES, 
                      TRADE_DATETIME_DAY, 
                      QUANTITY_CATEGORICAL]


quantity_greater_than_100k = lambda row: row['quantity'] >= np.log10(1e5)
quantity_greater_than_1m = lambda row: row['quantity'] >= np.log10(1e6)
trade_type_is_interdealer = lambda row: row['trade_type'] == 'D'


# key: name of criteria, value: (categories to match, filtering conditions)
related_trades_criterion = {'NONE': ([], []), 
                            'trade_type': (['trade_type'], []), 
                            'incorporated_state_code': (['incorporated_state_code'], []), 
                            'days_to_maturity_categorical': ([DAYS_TO_MATURITY_CATEGORICAL], []), 
                            'quantity_categorical': ([QUANTITY_CATEGORICAL], []), 
                            'coupon_categorical': ([COUPON_CATEGORICAL], []), 
                            'trade_datetime_day': ([TRADE_DATETIME_DAY], []), 
                            'rating_without_plus_minus_B_NR_combined': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED], []), 
                            'days_to_call': ([DAYS_TO_CALL_CATEGORICAL], []), 
                            'purpose_class_top_values': ([PURPOSE_CLASS_TOP_VALUES], []), 
                            'muni_security_type_top_values': ([MUNI_SECURITY_TYPE_TOP_VALUES], []), 
                            '100k': ([], [quantity_greater_than_100k]), 
                            '1m': ([], [quantity_greater_than_1m]), 
                            'dd': ([], [trade_type_is_interdealer]), 
                            'rating': (['rating'], []), 
                            'purpose_class': (['purpose_class'], []), 
                            'coupon_categorical_sudhar': ([COUPON_CATEGORICAL_SUDHAR], []), 
                            'sudhar1': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL], []), 
                            'sudhar1_100k': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL], [quantity_greater_than_100k]), 
                            'sudhar1_1m': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL], [quantity_greater_than_1m]), 
                            'sudhar2': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, 'trade_type'], []), 
                            'sudhar2_100k': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, 'trade_type'], [quantity_greater_than_100k]), 
                            'sudhar2_1m': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, 'trade_type'], [quantity_greater_than_1m]), 
                            'sudhar3': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL], [trade_type_is_interdealer]), 
                            'sudhar3_100k': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL], [quantity_greater_than_100k, trade_type_is_interdealer]), 
                            'sudhar3_1m': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL], [quantity_greater_than_1m, trade_type_is_interdealer]), 
                            'sudhar4': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, MUNI_SECURITY_TYPE_TOP_VALUES], []), 
                            'sudhar4_100k': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, MUNI_SECURITY_TYPE_TOP_VALUES], [quantity_greater_than_100k]), 
                            'sudhar4_1m': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, MUNI_SECURITY_TYPE_TOP_VALUES], [quantity_greater_than_1m]), 
                            'sudhar5': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, TRADE_DATETIME_DAY], []), 
                            'sudhar5_100k': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, TRADE_DATETIME_DAY], [quantity_greater_than_100k]), 
                            'sudhar5_1m': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, COUPON_CATEGORICAL, TRADE_DATETIME_DAY], [quantity_greater_than_1m]), 
                            'sudhar6': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, COUPON_CATEGORICAL, TRADE_DATETIME_DAY], []), 
                            'sudhar6_100k': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, COUPON_CATEGORICAL, TRADE_DATETIME_DAY], [quantity_greater_than_100k]), 
                            'sudhar6_1m': ([RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, COUPON_CATEGORICAL, TRADE_DATETIME_DAY], [quantity_greater_than_1m]), 
                            'mitas1': ([TRADE_DATETIME_DAY, 'trade_type'], []),
                            'mitas1_100k': ([TRADE_DATETIME_DAY, 'trade_type'], [quantity_greater_than_100k]), 
                            'mitas1_1m': ([TRADE_DATETIME_DAY, 'trade_type'], [quantity_greater_than_1m]), 
                            'desmond': (['incorporated_state_code', RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, DAYS_TO_CALL_CATEGORICAL, COUPON_CATEGORICAL, PURPOSE_CLASS_TOP_VALUES], []), 
                            'desmond_100k': (['incorporated_state_code', RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, DAYS_TO_CALL_CATEGORICAL, COUPON_CATEGORICAL, PURPOSE_CLASS_TOP_VALUES], [quantity_greater_than_100k]), 
                            'desmond_1m': (['incorporated_state_code', RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, DAYS_TO_CALL_CATEGORICAL, COUPON_CATEGORICAL, PURPOSE_CLASS_TOP_VALUES], [quantity_greater_than_1m]), 
                            'yellow': (['trade_type', DAYS_TO_MATURITY_CATEGORICAL, QUANTITY_CATEGORICAL, DAYS_TO_CALL_CATEGORICAL], []), 
                            'yellow_100k': (['trade_type', DAYS_TO_MATURITY_CATEGORICAL, QUANTITY_CATEGORICAL, DAYS_TO_CALL_CATEGORICAL], [quantity_greater_than_100k]), 
                            'yellow_1m': (['trade_type', DAYS_TO_MATURITY_CATEGORICAL, QUANTITY_CATEGORICAL, DAYS_TO_CALL_CATEGORICAL], [quantity_greater_than_1m]), 
                            'yellow_lite': (['trade_type', DAYS_TO_MATURITY_CATEGORICAL], []), 
                            'yellow_lite_100k': (['trade_type', DAYS_TO_MATURITY_CATEGORICAL], [quantity_greater_than_100k]), 
                            'yellow_lite_1m': (['trade_type', DAYS_TO_MATURITY_CATEGORICAL], [quantity_greater_than_1m]), 
                            }


def add_related_trades(df: pd.DataFrame, prefix: str, num_trades: int, categorical_features: list=[]):
    df = df.copy()

    RELATED_TRADES_CRITERION_OPT = 'desmond'

    RELATED_TRADE_CATEGORICAL_FEATURES_WITH_REFERENCE_FEATURES = list(set(RELATED_TRADE_CATEGORICAL_FEATURES + categorical_features))
    categorical_features_to_add_functions = {feature: get_neighbor_feature(feature) for feature in categorical_features}
    related_trade_feature_functions = {**RELATED_TRADE_FEATURE_FUNCTIONS, **categorical_features_to_add_functions}    # RELATED_TRADE_FEATURE_FUNCTIONS | categorical_features_to_add_functions    # changed code to work with python 3.7 (| notation for combining ditionaries is for 3.9+)

    print(f'Each trade in the related trade history has the following features: {list(related_trade_feature_functions.keys())}')

    RELATED_TRADE_FEATURE_FUNCTIONS_AND_DEFAULT_VALUES = {key: (function, DEFAULT_VALUES[key]) for key, function in related_trade_feature_functions.items()}

    epsilon = 1 / VERY_LARGE_NUMBER

    df[RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED] = df['rating'].transform(lambda rating: str.rstrip(rating, '+-'))    # remove + and - from right side of string
    # group BBB, BB, B, and NR together since each have a very small number of trades
    b_ratings = df[RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED].isin(['B', 'BB', 'BBB', 'NR'])
    df.loc[b_ratings, RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED] = 'B'
    print(f'Created {RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED} feature')

    num_of_days_bins_maturity = [np.log10(days) for days in [epsilon, NUM_OF_DAYS_IN_YEAR * 2, NUM_OF_DAYS_IN_YEAR * 5, NUM_OF_DAYS_IN_YEAR * 10, VERY_LARGE_NUMBER]]    # 2 years, 5 years, 10 years; arbitrarily chosen
    df[DAYS_TO_MATURITY_CATEGORICAL] = pd.cut(df['days_to_maturity'], num_of_days_bins_maturity).astype('string')
    print(f'Created {DAYS_TO_MATURITY_CATEGORICAL} feature')

    num_of_days_bins_call = [np.log10(days) for days in [epsilon, NUM_OF_DAYS_IN_YEAR * 2, NUM_OF_DAYS_IN_YEAR * 5, VERY_LARGE_NUMBER]]    # 2 years, 5 years; arbitrarily chosen
    df[DAYS_TO_CALL_CATEGORICAL] = pd.cut(df['days_to_call'], num_of_days_bins_call).astype('string')
    print(f'Created {DAYS_TO_CALL_CATEGORICAL} feature')

    coupon_bins = [0, 3, 4, 4.5, 5.0 + epsilon, VERY_LARGE_NUMBER]   # 0 - 2.99, 3 - 3.99, 4 - 4.49, 4.5 - 5; from discussion with a team member
    df[COUPON_CATEGORICAL] = pd.cut(df['coupon'], coupon_bins, right=False).astype('string')
    print(f'Created {COUPON_CATEGORICAL} feature')

    coupon_bins = [0, 3, 4, 4.5, 5, 5.25, 5.5, 6, VERY_LARGE_NUMBER]    # from Sudhar's paper: Kolm, Purushothaman. 2021. Systematic Pricing and Trading of Municipal Bonds
    df[COUPON_CATEGORICAL_SUDHAR] = pd.cut(df['coupon'], coupon_bins, right=False).astype('string')
    print(f'Created {COUPON_CATEGORICAL_SUDHAR} feature')

    df[PURPOSE_CLASS_TOP_VALUES] = df['purpose_class']
    top6_purpose_class_values = df['purpose_class'].value_counts().head(6).index.tolist()    # select the top 6 coupon values based on frequency in the data, which are: 37 (school district), 51 (various purpose), 50 (utility), 46 (tax revenue), 9 (education), 48 (transportation) comprising about 80% of the data
    df.loc[~df['purpose_class'].isin(top6_purpose_class_values), PURPOSE_CLASS_TOP_VALUES] = -1    # arbitrary numerical value that is invalid as a purpose_class value
    print(f'Created {PURPOSE_CLASS_TOP_VALUES} feature')

    df[MUNI_SECURITY_TYPE_TOP_VALUES] = df['muni_security_type']
    top6_muni_security_type_values = df['muni_security_type'].value_counts().head(2).index.tolist()    # select the top 2 coupon values based on frequency in the data, which are: 8 (revenue), 5 (unlimited g.o.) comprising about 80% of the data
    df.loc[~df['muni_security_type'].isin(top6_muni_security_type_values), MUNI_SECURITY_TYPE_TOP_VALUES] = -1    # arbitrary numerical value that is invalid as a purpose_class value
    print(f'Created {MUNI_SECURITY_TYPE_TOP_VALUES} feature')

    df[TRADE_DATETIME_DAY] = df['trade_datetime'].transform(lambda datetime: datetime.date()).astype('string')    # remove timestamp from datetime
    print(f'Created {TRADE_DATETIME_DAY} feature')

    quantity_bins = [0, 5, 6, 7, VERY_LARGE_NUMBER]    # 0 - 100k, 100k - 1m, 1m - 10m, 10m+
    df[QUANTITY_CATEGORICAL] = pd.cut(df['quantity'], quantity_bins).astype('string')
    print(f'Created {QUANTITY_CATEGORICAL} feature')

    categories_to_match, filtering_conditions = related_trades_criterion[RELATED_TRADES_CRITERION_OPT]

    df_encoded, encoders = encode_and_get_encoders(df, [], categorical_features)

    print(f'Putting related trades into data')

    append_recent_trade_data_start_time = time.time()
    df_with_related_trades = append_recent_trade_data(df, 
                                                      num_trades, 
                                                      RELATED_TRADE_FEATURE_FUNCTIONS_AND_DEFAULT_VALUES, 
                                                      feature_prefix=prefix, 
                                                      categories=categories_to_match, 
                                                      filtering_conditions=filtering_conditions, 
                                                      return_df=True, 
                                                      multiprocessing=True, 
                                                      df_for_related_trades=df_encoded).drop(columns=quantized_features)    # drop the quantized features from the final dataframe
    print(f'Time elapsed to add related trades information: {datetime.timedelta(seconds=time.time() - append_recent_trade_data_start_time)}')

    related_trade_features = lambda features: flatten([[get_appended_feature_name(idx, feature, prefix) for feature in features] for idx in range(num_trades)])

    # convert same_day features to boolean
    same_day_features = related_trade_features(['same_day'])
    df_with_related_trades.loc[:, same_day_features] = df_with_related_trades.loc[:, same_day_features].astype(bool)

    # convert categorical features to original value
    for feature in categorical_features:
        related_trade_features_for_feature = related_trade_features([feature])
        encoder = encoders[feature]
        for related_trade_feature in related_trade_features_for_feature:
            df_with_related_trades.loc[:, related_trade_feature] = encoder.inverse_transform(df_with_related_trades.loc[:, related_trade_feature].astype(int))    # change type to int to work with `.inverse_transform(...)` function of `LabelEncoder`
        df_with_related_trades.loc[:, related_trade_features_for_feature] = df_with_related_trades.loc[:, related_trade_features_for_feature].astype('category')

    # convert trade_type1 and trade_type2 features to trade_type categorical feature
    df_with_related_trades, old_trade_type_columns, new_trade_type_features = convert_trade_type_encoding_to_actual(df_with_related_trades, 1, prefix=prefix)    # category dtype conversion already occurs in this function
    df_with_related_trades = df_with_related_trades.drop(columns=old_trade_type_columns)

    # convert trade_type_past_latest features to categorical
    trade_type_past_latest_features = related_trade_features(['trade_type_past_latest'])
    inverted_trade_type_cross_product_mapping = {value: key for key, value in TRADE_TYPE_CROSS_PRODUCT_MAPPING.items()}
    for feature in trade_type_past_latest_features:
        df_with_related_trades.loc[:, feature] = df_with_related_trades.loc[:, feature].map(inverted_trade_type_cross_product_mapping.get)
    df_with_related_trades.loc[:, trade_type_past_latest_features] = df_with_related_trades.loc[:, trade_type_past_latest_features].astype('category')

    return df_with_related_trades, related_trade_features(RELATED_TRADE_BINARY_FEATURES), related_trade_features(RELATED_TRADE_CATEGORICAL_FEATURES_WITH_REFERENCE_FEATURES) + new_trade_type_features, related_trade_features(RELATED_TRADE_NON_CAT_FEATURES)