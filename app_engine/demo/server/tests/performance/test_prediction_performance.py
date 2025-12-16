'''
'''
import numpy as np
import pandas as pd

from modules.ficc.utils.auxiliary_functions import sqltodf

from modules.test.auxiliary_variables import DIRECTORY
from modules.test.auxiliary_functions import response_from_batch_pricing, get_bq_client


LIMIT = 1000


def check_irregular(df, irregular_cusips_filename=f'{DIRECTORY}/check_irregular.csv'):
    '''Checks that the yield to worst is in a reasonable range, and that the ytw and price 
    are in the correct ranges based on the coupon values. The test fails only if the CUSIP 
    has at least 6 months to the yield_to_worst date since CUSIPs that have short duration 
    could hold the irregular condition where our model is fine.'''
    df['ytw'] = df['ytw'].astype(float)
    df_ytw_negative = df[df['ytw'] < 0]
    if len(df_ytw_negative) != 0: df_ytw_negative.to_csv(irregular_cusips_filename, index=None)    # save this csv to be able to inspect by hand later
    assert len(df_ytw_negative) == 0, df_ytw_negative    # yield to worst should be nonnegative value

    df_ytw_greater_than_10 = df[df['ytw'] > 10]
    if len(df_ytw_greater_than_10) != 0: df_ytw_greater_than_10.to_csv(irregular_cusips_filename, index=None)    # save this csv to be able to inspect by hand later
    assert len(df_ytw_greater_than_10) == 0, df_ytw_greater_than_10    # yield to worst should be less than 10

    df = df[~df['callable_at_cav']]    # the below condition is not applicable for bonds that are callable at cav
    df['coupon'] = df['coupon'].astype(float)
    df['price'] = df['price'].astype(float)
    df.loc[:, 'ytw_greater_than_coupon'] = df['ytw'] > df['coupon']
    df.loc[:, 'price_greater_than_100'] = df['price'] > 100
    df.loc[:, 'ytw_less_than_coupon'] = df['ytw'] < df['coupon']
    df.loc[:, 'price_less_than_100'] = df['price'] < 100
    df = df[(df['ytw_greater_than_coupon'] & df['price_greater_than_100']) | (df['ytw_less_than_coupon'] & df['price_less_than_100'])]

    if len(df) != 0: df.to_csv(irregular_cusips_filename, index=None)    # save this csv to be able to inspect by hand later

    df['yield_to_worst_date'] = pd.to_datetime(df['yield_to_worst_date'])
    df['months_from_yield_to_worst_date'] = (df['yield_to_worst_date'] - pd.to_datetime('today')) / np.timedelta64(1, 'M')
    df = df[df['months_from_yield_to_worst_date'] >= 6]    # trigger AssertionError (test failure) only when the irregular CUSIP is one that has a yield_to_worst date that is more than 6 months away since CUSIPs that have short duration may have the irregular condition

    assert len(df) == 0, df


def test_random_1000_to_price_irregularities():
    '''Tests 1000 CUSIPs that we should be able to price by making sure that none are giving 
    irregular predictions. See `check_irregular` for a more specific definition of irregular.'''
    bq_client = get_bq_client()

    # getting callable_at_cav value so that this feature can be used to check irregularities
    query = f'''
        SELECT DISTINCT cusip, callable_at_cav
        FROM `reference_data_v2.reference_data_flat`
        WHERE maturity_description_code = 2
          AND maturity_date > current_date
          AND next_call_date > current_date
          AND refund_date > current_date
          AND (interest_payment_frequency = 1 OR interest_payment_frequency = 16)
          AND ref_valid_to_date > current_timestamp
          AND (coupon_type = 8 OR coupon_type = 17 OR coupon_type = 4 OR coupon_type = 10)
          AND outstanding_indicator is true
        LIMIT {LIMIT}'''
    df = sqltodf(query, bq_client)
    cusip_list = df['cusip'].tolist()
    assert len(set(cusip_list)) == LIMIT    # check that there are no duplicate CUSIPs
    
    cusip_callableatcav_pairs = df.values.tolist()
    cusip_to_callableatcav = dict(cusip_callableatcav_pairs)

    filename = f'{DIRECTORY}/{LIMIT}cusips.csv'
    request_obj = response_from_batch_pricing(filename, cusip_list)

    assert request_obj.ok    # successful response; checks whether the status_code is less than 400
    content = request_obj.content.decode('utf-8')
    content = content.split('\n')
    columns = content[0].split(',')    # first row is the column names
    content = content[1:-1]    # first row is the column names, last row is an empty line
    content = [row.split(',') for row in content]

    df = pd.DataFrame(content, columns=columns)
    df['callable_at_cav'] = df['cusip'].map(cusip_to_callableatcav)    # add `callable_at_cav` column
    check_irregular(df)


def test_random_1000_compare_price_with_avg_trade_history():
    '''Query 1000 CUSIPs with at least `trade_history_length` trades in the 
    trade history and compute the mean price and mean yield and corresponding 
    standard deviations for the trade history. Only select CUSIPs that have 
    the earliest trade in the history (up to the last 32 trades) within 
    `num_days_ago_limit_for_earlist_trade` days from the current date. Check 
    if any of our predictions are more than `num_std_dev` standard deviations 
    away from the mean.'''
    trade_history_length = 5
    num_days_ago_limit_for_earlist_trade = 30    # select only CUSIPs which have the earliest trade in the history (up to the last 32 trades) within the last 30 days
    num_std_dev = 2    # the number of standard deviations away from the mean that we are tolerating
    bq_client = get_bq_client()

    query = f'''
        SELECT cusip, recent
        FROM ( SELECT cusip, recent, recent_values.trade_datetime AS recent_trade_datetime, ROW_NUMBER() OVER (PARTITION BY MSRB.cusip ORDER BY recent_values.trade_datetime ASC) AS row_num    -- order by trade_datetime ASC so that row_num = 1 corresponds to the earliest trade in the trade history
               FROM `eng-reactor-287421.auxiliary_views.trade_history_latest_ref_data_minimal_exclusions` as MSRB, UNNEST(recent) as recent_values
               WHERE maturity_description_code = 2
                 AND DATE_DIFF(MSRB.maturity_date, current_date, year) > 1
                 AND MSRB.next_call_date > current_date
                 AND MSRB.refund_date > current_date
                 AND (MSRB.interest_payment_frequency = 1
                   OR MSRB.interest_payment_frequency = 16)
                 AND ref_valid_to_date > current_timestamp
                 AND (coupon_type = 8
                   OR coupon_type = 17
                   OR coupon_type = 4
                   OR coupon_type = 10)
                 AND outstanding_indicator IS TRUE
                 AND ARRAY_LENGTH(recent) >= {trade_history_length} )    -- ensures that the history is of length at least trade_history_length
        WHERE row_num = 1    -- selecting row_num = 1 so that recent_trade_datetime refers to the earliest trade in the trade history
          AND TIMESTAMP_DIFF(current_datetime, recent_trade_datetime, DAY) <= {num_days_ago_limit_for_earlist_trade}
        LIMIT {LIMIT}'''
    df = sqltodf(query, bq_client)

    cusip_list = df['cusip'].tolist()
    history_list = df['recent'].tolist()

    ytw_mean = []
    ytw_std = []
    price_mean = []
    price_std = []
    for history in history_list:
        ytws = [trade['yield'] for trade in history]
        ytw_mean.append(np.mean(ytws))
        ytw_std.append(np.std(ytws))
        prices = [trade['dollar_price'] for trade in history]
        price_mean.append(np.mean(prices))
        price_std.append(np.std(prices))

    filename = f'{DIRECTORY}/{LIMIT}cusips_history.csv'
    request_obj = response_from_batch_pricing(filename, cusip_list)

    assert request_obj.ok    # successful response; checks whether the status_code is less than 400
    content = request_obj.content.decode('utf-8')
    content = content.split('\n')
    columns = content[0].split(',')    # first row is the column names
    content = content[1:-1]    # first row is the column names, last row is an empty line
    content = [row.split(',') for row in content]

    df = pd.DataFrame(content, columns=columns)
    df['ytw_mean'] = ytw_mean
    df['ytw_std'] = ytw_std
    df['price_mean'] = price_mean
    df['price_std'] = price_std

    df.loc[:, 'ytw_upper_bound'] = df['ytw_mean'] + num_std_dev * df['ytw_std']
    df.loc[:, 'ytw_lower_bound'] = df['ytw_mean'] - num_std_dev * df['ytw_std']
    df.loc[:, 'price_upper_bound'] = df['price_mean'] + num_std_dev * df['price_std']
    df.loc[:, 'price_lower_bound'] = df['price_mean'] - num_std_dev * df['price_std']

    df['ytw'] = df['ytw'].astype(float)
    df.loc[:, 'ytw_greater_than_upper_bound'] = df['ytw'] > df['ytw_upper_bound']
    df.loc[:, 'ytw_lesser_than_lower_bound'] = df['ytw'] < df['ytw_lower_bound']
    df['price'] = df['price'].astype(float)
    df.loc[:, 'price_greater_than_upper_bound'] = df['price'] > df['price_upper_bound']
    df.loc[:, 'price_lesser_than_lower_bound'] = df['price'] < df['price_lower_bound']

    df['history_list'] = history_list
    df = df[df['ytw_greater_than_upper_bound'] | df['ytw_lesser_than_lower_bound'] | df['price_greater_than_upper_bound'] | df['price_lesser_than_lower_bound']]
    df = df[['cusip', 'ytw', 'price', 'ytw_mean', 'ytw_std', 'price_mean', 'price_std', 'history_list']]
    if len(df) != 0: df.to_csv(filename, index=None)    # save this csv to be able to inspect later if there are errors
    assert len(df) == 0, df
