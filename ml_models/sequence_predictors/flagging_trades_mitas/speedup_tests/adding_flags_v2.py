'''
 '''

import numpy as np
import pandas as pd

from ficc.utils.auxiliary_variables import IS_REPLICA, IS_BOOKKEEPING, IS_SAME_DAY, NTBC_PRECURSOR
from ficc.utils.adding_flags import add_same_day_flag, add_ntbc_precursor_flag, add_replica_flag, add_bookkeeping_flag


def subarray_sum(lst, target_sum, indices):
    '''The goal is to find a sublist in `lst`, such that the sum of the sublist equals 
    `target_sum`. If such a sublist cannot be formed, then return `None`. Otherwise 
    return the indices that should be removed from `lst` so that summing the remaining 
    items equals `target_sum`. The sublist in `indices` is returned.'''
    len_lst = len(lst)
    current_sum = lst[0]
    start, end = 0, 1
    while end <= len_lst:
        while current_sum > target_sum and start < end - 1:    # remove items from beginning of current sublist if the current sum is larger than the target
            current_sum -= lst[start]
            start += 1
        if current_sum == target_sum:
            return indices[start:end]
        if end < len_lst:
            current_sum += lst[end]
        end += 1
    return None


def indices_to_remove_from_beginning_or_end_to_reach_sum(lst, target_sum):
    '''The goal is to find a continuous stream of items in `lst` where at least one of the 
    endpoints of `lst`, such that the sum of this stream of items equals `target_sum`. If 
    such a sublist cannot be formed, then return `None`. Otherwise return the indices that 
    should be removed from `lst` so that summing the remaining items equals `target_sum`.'''
    # forward pass
    lst_total = sum(lst)
    assert lst_total > target_sum
    indices = []
    for index, item in enumerate(lst):
        lst_total -= item
        indices.append(index)
        if lst_total == target_sum:
            return indices
    # backward pass
    lst_total = sum(lst)
    indices = []
    for index, item in reversed(list(enumerate(lst))):    # traverse a list in reverse order while preserving the indices: https://stackoverflow.com/questions/529424/traverse-a-list-in-reverse-order-in-python
        lst_total -= item
        indices.append(index)
        if lst_total == target_sum:
            return indices
    return None    # no such sublist found


def add_bookkeeping_flag_v2(df, flag_name=IS_BOOKKEEPING):
    '''Re-use implementation of `add_replica_flag(...)` for this 
    function.'''
    if flag_name in df.columns and df[flag_name].any(): return df
    # print(f'Adding {flag_name} flag to data')
    df = df.copy()
    if flag_name not in df.columns: df[flag_name] = False

    df_with_bookkeeping_flag = add_replica_flag_v2(df[df['trade_type'] == 'D'], flag_name)
    df.loc[df_with_bookkeeping_flag[df_with_bookkeeping_flag[flag_name]].index.to_list(), flag_name] = True    # mark all trades in `df` that were marked in `df_with_bookkeeping_flag` 
    return df


def add_bookkeeping_flag_v3(df, flag_name=IS_BOOKKEEPING):
    '''Re-use implementation of `add_replica_flag(...)` for this 
    function.'''
    df_with_bookkeeping_flag = add_replica_flag_v3(df[df['trade_type'] == 'D'], flag_name)
    df.loc[df_with_bookkeeping_flag[df_with_bookkeeping_flag[flag_name]].index.to_list(), flag_name] = True    # mark all trades in `df` that were marked in `df_with_bookkeeping_flag` 
    return df


def _add_same_day_flag_for_group_v2(group_df, flag_name, orig_df=None):
    '''This flag denotes a trade where the dealer had the purchase and sell lined up 
    beforehand. We mark a trade as same day when:
    1. A group of dealer sell trades are considered same day if the total cost of the 
    dealer purchase trades for that day is equal to or greater than the total cost of the 
    dealer sell trades. In this case, a group of dealer purchases trades are considered 
    same day if there is a continuous (continuous defined as a dealer purchase trade not 
    skipped over chronologically) sequence of dealer purchase trades that equal the total 
    cost of the dealer sell trades. We assume this sequence of dealer purchase trades 
    includes either the first dealer purchase trade of the day and/or the last dealer 
    purchase trade of the day. We may expand this criteria to not have to include either 
    the first and/or last dealer purchase trade.
    2. An inter-dealer trade is considered *same day* if the par_traded is equal to the total 
    cost of the dealer sell trades for that day and if the total cost of the dealer purchase 
    trades for that day is greater than or equal to the total cost of the dealer sell trades.'''

    assert flag_name in group_df.columns, '`{flag_name}` must be a column in the dataframe'
    groups_by_trade_type = group_df.groupby('trade_type', observed=True)['par_traded'].sum()
    if orig_df is None: orig_df = group_df
    if 'S' not in groups_by_trade_type.index or 'P' not in groups_by_trade_type.index: return orig_df
    dealer_sold_indices = group_df[group_df['trade_type'] == 'S'].index.values
    dealer_purchase_indices = group_df[group_df['trade_type'] == 'P'].index.values
    total_dealer_sold = groups_by_trade_type.loc['S']
    total_dealer_purchased = groups_by_trade_type.loc['P']

    indices_to_mark = []
    if total_dealer_sold <= total_dealer_purchased:
        indices_to_mark.extend(dealer_sold_indices)
        for index, par_traded in group_df[group_df['trade_type'] == 'D']['par_traded'].iteritems():
            if par_traded == total_dealer_sold:
                indices_to_mark.append(index)
        
        if total_dealer_sold == total_dealer_purchased:
            indices_to_mark.extend(dealer_purchase_indices)
        else:
            indices_to_remove_from_dealer_purchase_indices = indices_to_remove_from_beginning_or_end_to_reach_sum(group_df[group_df['trade_type'] == 'P']['par_traded'].values, total_dealer_sold)
            if indices_to_remove_from_dealer_purchase_indices is not None:
                for index_to_remove in sorted(indices_to_remove_from_dealer_purchase_indices, reverse=True):    # need to sort in reverse order to make sure future indices are still valid after removing current index; e.g., cannot remove elements at index 0 and 1 of a two element list in that order (index 1 does not exist after removing index 0)
                    dealer_purchase_indices = np.delete(dealer_purchase_indices, index_to_remove, axis=0)
                indices_to_mark.extend(dealer_purchase_indices)
    
    orig_df.loc[indices_to_mark, flag_name] = True
    return orig_df


def _add_same_day_flag_for_group_v4(group_df):
    '''This flag denotes a trade where the dealer had the purchase and sell lined up 
    beforehand. We mark a trade as same day when:
    1. A group of dealer sell trades are considered same day if the total cost of the 
    dealer purchase trades for that day is equal to or greater than the total cost of the 
    dealer sell trades. In this case, a group of dealer purchases trades are considered 
    same day if there is a continuous (continuous defined as a dealer purchase trade not 
    skipped over chronologically) sequence of dealer purchase trades that equal the total 
    cost of the dealer sell trades. We assume this sequence of dealer purchase trades 
    includes either the first dealer purchase trade of the day and/or the last dealer 
    purchase trade of the day. We may expand this criteria to not have to include either 
    the first and/or last dealer purchase trade.
    2. An inter-dealer trade is considered *same day* if the par_traded is equal to the total 
    cost of the dealer sell trades for that day and if the total cost of the dealer purchase 
    trades for that day is greater than or equal to the total cost of the dealer sell trades.'''

    group_df_by_trade_type = group_df.groupby('trade_type', observed=True)
    if 'S' not in group_df_by_trade_type.groups.keys() or 'P' not in group_df_by_trade_type.groups.keys(): return []

    dealer_sold_indices, dealer_purchase_indices = group_df_by_trade_type.get_group('S').index, group_df_by_trade_type.get_group('P').index
    group_df_by_trade_type_sums = group_df_by_trade_type['par_traded'].sum()
    total_dealer_sold, total_dealer_purchased = group_df_by_trade_type_sums['S'], group_df_by_trade_type_sums['P']

    indices_to_mark = []
    if total_dealer_sold <= total_dealer_purchased:
        indices_to_mark.extend(dealer_sold_indices)
        dd_indices_to_mark = group_df[(group_df['trade_type'] == 'D') & (group_df['par_traded'] == total_dealer_sold)].index
        indices_to_mark.extend(dd_indices_to_mark)
        
        if total_dealer_sold == total_dealer_purchased:
            indices_to_mark.extend(dealer_purchase_indices)
        else:
            # indices_to_mark_from_dealer_purchase_indices = subarray_sum(group_df_by_trade_type.get_group('P')['par_traded'].values, total_dealer_sold, dealer_purchase_indices)
            # if indices_to_mark_from_dealer_purchase_indices is not None: indices_to_mark.extend(indices_to_mark_from_dealer_purchase_indices)
            indices_to_remove_from_dealer_purchase_indices = indices_to_remove_from_beginning_or_end_to_reach_sum(group_df_by_trade_type.get_group('P')['par_traded'].values, total_dealer_sold)
            if indices_to_remove_from_dealer_purchase_indices is not None:
                for index_to_remove in sorted(indices_to_remove_from_dealer_purchase_indices, reverse=True):    # need to sort in reverse order to make sure future indices are still valid after removing current index; e.g., cannot remove elements at index 0 and 1 of a two element list in that order (index 1 does not exist after removing index 0)
                    dealer_purchase_indices = np.delete(dealer_purchase_indices, index_to_remove, axis=0)
                indices_to_mark.extend(dealer_purchase_indices)

    return indices_to_mark


def add_same_day_flag_v2(df, flag_name=IS_SAME_DAY):
    '''Call `_add_bookkeeping_flag_for_group(...)` on each group as 
    specified in the `groupby`.'''
    df['par_traded'] = df['par_traded'].astype(np.float32)    # `par_traded` type is Category so need to change it order to sum up
    groups_by_day = df.groupby(pd.Grouper(key='trade_datetime', freq='1D'), observed=True)
    for _, group_df_day in groups_by_day:
        groups_by_day_cusip = group_df_day.groupby('cusip', observed=True)    # .filter(lambda group_df: 'S' in group_df['trade_type'] and 'P' in group_df['trade_type'])
        for _, group_df in groups_by_day_cusip:
            df = _add_same_day_flag_for_group_v2(group_df, flag_name, df)
    return df


def add_same_day_flag_v3(df, flag_name=IS_SAME_DAY):
    '''Call `_add_bookkeeping_flag_for_group(...)` on each group as 
    specified in the `groupby`.'''
    df['par_traded'] = df['par_traded'].astype(np.float32)    # `par_traded` type is Category so need to change it order to sum up
    groups = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip'])
    for _, group_df in groups:
        df = _add_same_day_flag_for_group_v2(group_df, flag_name, df)
    return df


def add_same_day_flag_v4(df, flag_name=IS_SAME_DAY):
    '''Call `_add_bookkeeping_flag_for_group(...)` on each group as 
    specified in the `groupby`.'''
    df = df.astype({'par_traded': np.float32})    # `par_traded` type is Category so need to change it order to sum up

    df[flag_name] = False
    group_by_apply_object = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip'], observed=True)[['par_traded', 'trade_type']].apply(_add_same_day_flag_for_group_v4)
    indices_to_mark = group_by_apply_object.sum()
    df.loc[indices_to_mark, flag_name] = True
    return df


def add_same_day_flag_v5(df, flag_name=IS_SAME_DAY):
    '''Call `_add_bookkeeping_flag_for_group(...)` on each group as 
    specified in the `groupby`.'''
    df = df.astype({'par_traded': np.float32})    # `par_traded` type is Category so need to change it order to sum up

    df[flag_name] = False
    group_by_apply_object = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip'], observed=True)[['par_traded', 'trade_type']].parallel_apply(_add_same_day_flag_for_group_v4)
    indices_to_mark = group_by_apply_object.values.sum()
    df.loc[indices_to_mark, flag_name] = True
    return df


def _add_replica_flag_for_group_v2(group_df, flag_name, orig_df=None):
    '''Mark a trade as a replica if there is a previous trade on the same 
    day with the same price, same direction, and same quantity. The idea 
    of marking these trades is to exclude them from the trade history, as 
    these trades are probably being sold in the same block, and so having 
    all of these trades in the trade history would be less economically 
    meaningful in the trade history. All except the earliest trade in this 
    group are marked as a replica.'''
    assert flag_name in group_df.columns, '`{flag_name}` must be a column in the dataframe in order to mark that this trade is identical to another trade that occurs during the same day'
    if orig_df is None: orig_df = group_df
    if len(group_df) < 2: return orig_df    # dataframe has a size less than 2
    orig_df.loc[group_df.index.to_list(), flag_name] = True    # mark all trades in the group
    return orig_df


def add_replica_flag_v2(df, flag_name=IS_REPLICA):
    '''Call `_add_replica_flag_for_group(...)` on each group as 
    specified in the `groupby`.'''
    if flag_name in df.columns and df[flag_name].any(): return df
    print(f'Adding {flag_name} flag to data')
    df = df.copy()
    if flag_name not in df.columns: df[flag_name] = False

    # Next 3 lines ensure that correct feature is used to find the quantity. When 
    # this function is run before `process_features(...)`, the quantity is 
    # represented as the feature `par_traded`. After running `process_features(...)`,
    # the quantity is represented as the feature `quantity`.
    columns_set = set(df.columns)
    quantity_feature = 'quantity' if 'quantity' in columns_set else 'par_traded'
    assert 'par_traded' in columns_set, 'Neither "quantity" nor "par_traded" exist in the dataframe'

    groups_same_day_quantity_price_tradetype_cusip = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), quantity_feature, 'dollar_price', 'trade_type', 'cusip'], observed=True)    # considered adding SPECIAL_CONDITIONS_TO_FILTER_ON in the groupby but it makes the condition too restrictive
    groups_same_day_quantity_price_tradetype_cusip = [group_df for _, group_df in groups_same_day_quantity_price_tradetype_cusip if len(group_df) > 1]    # remove singleton groups
    for group_df in groups_same_day_quantity_price_tradetype_cusip:
        df = _add_replica_flag_for_group_v2(group_df, flag_name, df)
    return df


def add_replica_flag_v3(df, flag_name=IS_REPLICA):
    '''Call `_add_replica_flag_for_group(...)` on each group as 
    specified in the `groupby`.'''
    groups_same_day_quantity_price_tradetype_cusip = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'quantity', 'dollar_price', 'trade_type', 'cusip'], observed=True)    # considered adding SPECIAL_CONDITIONS_TO_FILTER_ON in the groupby but it makes the condition too restrictive
    groups_same_day_quantity_price_tradetype_cusip = [group_df for _, group_df in groups_same_day_quantity_price_tradetype_cusip if len(group_df) > 1]    # remove singleton groups
    for group_df in groups_same_day_quantity_price_tradetype_cusip:
        df = _add_replica_flag_for_group_v2(group_df, flag_name, df)
    return df


def _add_replica_flag_for_group_v4(group_df):
    return group_df.index.to_list() if len(group_df) >= 2 else []


def add_replica_flag_v4(df, flag_name=IS_REPLICA):
    df[flag_name] = False
    group_by_apply_object = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip', 'quantity', 'dollar_price', 'trade_type'], observed=True).apply(_add_replica_flag_for_group_v4)
    indices_to_mark = group_by_apply_object.sum()
    df.loc[indices_to_mark, flag_name] = True
    return df


def add_replica_flag_v5(df, flag_name=IS_REPLICA):
    df[flag_name] = False
    group_by_apply_object = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip', 'quantity', 'dollar_price', 'trade_type'], observed=True).parallel_apply(_add_replica_flag_for_group_v4)
    indices_to_mark = group_by_apply_object.sum()
    df.loc[indices_to_mark, flag_name] = True
    return df


# def _add_bookkeeping_flag_for_group_v4(group_df):
#     return group_df.index.to_list() if len(group_df) >= 2 else []


def add_bookkeeping_flag_v4(df, flag_name=IS_BOOKKEEPING):
    df[flag_name] = False
    group_by_apply_object = df[df['trade_type'] == 'D'].groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip', 'quantity', 'dollar_price'], observed=True).apply(_add_replica_flag_for_group_v4)
    indices_to_mark = group_by_apply_object.sum()
    df.loc[indices_to_mark, flag_name] = True
    return df


def add_bookkeeping_flag_v5(df, flag_name=IS_BOOKKEEPING):
    df[flag_name] = False
    group_by_apply_object = df[df['trade_type'] == 'D'].groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip', 'quantity', 'dollar_price'], observed=True).parallel_apply(_add_replica_flag_for_group_v4)
    indices_to_mark = group_by_apply_object.sum()
    df.loc[indices_to_mark, flag_name] = True
    return df


def add_ntbc_precursor_flag_v2(df, flag_name=NTBC_PRECURSOR, return_candidates_dict=False):
    '''This flag denotes an inter-dealer trade that occurs on the same day as 
    a non-transaction-based-compensation customer trade with the same price and 
    quantity. The idea for marking it is that this inter-dealer trade may not be 
    genuine (i.e., window-dressing). Note that we have a buffer of occurring on 
    the same day since we see examples in the data (e.g., cusip 549696RS3, 
    trade_datetime 2022-04-01) having the corresponding inter-dealer trade occurring 
    4 seconds before, instead of the exact same time, as the customer bought trade. 
    The `return_candidates_dict` argument is used for debugging only.'''
    if flag_name in df.columns and df[flag_name].any(): return df
    print(f'Adding {flag_name} flag to data')
    df = df.copy()
    if flag_name not in df.columns: df[flag_name] = False

    # add datetime date column to the dataframe
    TRADE_DATETIME_DATE = 'trade_datetime_date'
    assert TRADE_DATETIME_DATE not in df.columns
    df[TRADE_DATETIME_DATE] = df['trade_datetime'].dt.date

    if return_candidates_dict: multiple_candidates = dict()    # initialize the dictionary
    def store_trade(trade, num_candidates):
        if num_candidates not in multiple_candidates: multiple_candidates[num_candidates] = []
        multiple_candidates[num_candidates].append((trade['cusip'], trade['rtrs_control_number'], trade['trade_datetime']))
    
    dd = df[df['trade_type'] == 'D']    # any NTBC precursor candidate must be an inter-dealer trade
    # for each NTBC customer trade, the inter-dealer trade must be on that day with the same quantity, price, and cusip
    features_to_match = ['cusip', 'quantity', 'dollar_price', TRADE_DATETIME_DATE]
    ntbc_precursor_candidates_groups = dd.groupby(features_to_match, observed=True)
    ntbc_precursor_candidates_group_headers = ntbc_precursor_candidates_groups.groups.keys()
    for _, ntbc_trade in df[df['is_non_transaction_based_compensation'] & ((df['trade_type'] == 'S') | (df['trade_type'] == 'P'))].iterrows():    # need the `ntbc_trade` variable name when evaluating `condition_based_on_features_to_match`
        group_header = tuple([ntbc_trade[feature] for feature in features_to_match])    # group header must be immutable, hence the tuple
        if group_header in ntbc_precursor_candidates_group_headers:    # group header must exist in the `ntbc_precursor_candidates_groups` for there to be trades to mark
            ntbc_precursor_candidates = ntbc_precursor_candidates_groups.get_group(group_header)
            if return_candidates_dict: store_trade(ntbc_trade, len(ntbc_precursor_candidates))
            df.loc[ntbc_precursor_candidates.index.to_list(), flag_name] = True
        elif return_candidates_dict: store_trade(ntbc_trade, 0)   # logs the situation in `multiple_candidates` when no candidates are found
    df = df.drop(columns=[TRADE_DATETIME_DATE])
    return (df, multiple_candidates) if return_candidates_dict else df


def _add_ntbc_precursor_flag_for_group_v2(group_df, flag_name, orig_df=None):
    '''Each group has the same trade date, price, and quantity.'''
    assert flag_name in group_df.columns, '`{flag_name}` must be a column in the dataframe in order to mark that this trade is identical to another trade that occurs during the same day'
    if orig_df is None: orig_df = group_df
    if len(group_df[(group_df['is_non_transaction_based_compensation']) & ((group_df['trade_type'] == 'S') | (group_df['trade_type'] == 'P'))]) > 0:
        orig_df.loc[group_df[group_df['trade_type'] == 'D'].index.to_list(), flag_name] = True    # mark all DD trades in the group
    return orig_df


def _add_ntbc_precursor_flag_for_group_v4(group_df):
    is_dd_trade = group_df['trade_type'] == 'D'
    if (not group_df['is_non_transaction_based_compensation'].any()) or (not is_dd_trade.any()) or (len(group_df) < 2): return []
    return group_df[is_dd_trade].index.to_list()


def add_ntbc_precursor_flag_v4(df, flag_name=NTBC_PRECURSOR):
    df[flag_name] = False
    group_by_apply_object = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip', 'quantity', 'dollar_price'], observed=True)[['is_non_transaction_based_compensation', 'trade_type']].apply(_add_ntbc_precursor_flag_for_group_v4)
    indices_to_mark = group_by_apply_object.sum()
    df.loc[indices_to_mark, flag_name] = True
    return df


def add_ntbc_precursor_flag_v5(df, flag_name=NTBC_PRECURSOR):
    df[flag_name] = False
    group_by_apply_object = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip', 'quantity', 'dollar_price'], observed=True)[['is_non_transaction_based_compensation', 'trade_type']].parallel_apply(_add_ntbc_precursor_flag_for_group_v4)
    indices_to_mark = group_by_apply_object.values.sum()
    df.loc[indices_to_mark, flag_name] = True
    return df


def add_ntbc_precursor_flag_v3(df, flag_name=NTBC_PRECURSOR):
    '''This flag denotes an inter-dealer trade that occurs on the same day as 
    a non-transaction-based-compensation customer trade with the same price and 
    quantity. The idea for marking it is that this inter-dealer trade may not be 
    genuine (i.e., window-dressing). Note that we have a buffer of occurring on 
    the same day since we see examples in the data (e.g., cusip 549696RS3, 
    trade_datetime 2022-04-01) having the corresponding inter-dealer trade occurring 
    4 seconds before, instead of the exact same time, as the customer bought trade. 
    The `return_candidates_dict` argument is used for debugging only.'''
    # add datetime date column to the dataframe
    TRADE_DATETIME_DATE = 'trade_datetime_date'
    assert TRADE_DATETIME_DATE not in df.columns
    df[TRADE_DATETIME_DATE] = df['trade_datetime'].dt.date
    
    dd = df[df['trade_type'] == 'D']    # any NTBC precursor candidate must be an inter-dealer trade
    # for each NTBC customer trade, the inter-dealer trade must be on that day with the same quantity, price, and cusip
    features_to_match = ['cusip', 'quantity', 'dollar_price', TRADE_DATETIME_DATE]
    ntbc_precursor_candidates_groups = dd.groupby(features_to_match, observed=True)
    ntbc_precursor_candidates_group_headers = ntbc_precursor_candidates_groups.groups.keys()
    for _, ntbc_trade in df[df['is_non_transaction_based_compensation'] & ((df['trade_type'] == 'S') | (df['trade_type'] == 'P'))].iterrows():    # need the `ntbc_trade` variable name when evaluating `condition_based_on_features_to_match`
        group_header = tuple([ntbc_trade[feature] for feature in features_to_match])    # group header must be immutable, hence the tuple
        if group_header in ntbc_precursor_candidates_group_headers:    # group header must exist in the `ntbc_precursor_candidates_groups` for there to be trades to mark
            ntbc_precursor_candidates = ntbc_precursor_candidates_groups.get_group(group_header)
            df.loc[ntbc_precursor_candidates.index.to_list(), flag_name] = True
    df = df.drop(columns=[TRADE_DATETIME_DATE])
    return df


def add_all_flags(df):
    '''Procedure that adds all of the flags above. Done in one procedure to reduce 
    overhead of groupby and df copy to create overall speedup.'''
    print('Adding all flags')
    df = df.copy()
    columns_set = set(df.columns)
    for flag in (IS_SAME_DAY, IS_BOOKKEEPING, IS_REPLICA, NTBC_PRECURSOR):
        if flag not in columns_set: df[flag] = False
    df = add_same_day_flag(df)
    df = add_ntbc_precursor_flag(df)
    df = add_replica_flag(df)
    df = add_bookkeeping_flag(df)
    return df


def add_all_flags_v2(df):
    '''Procedure that adds all of the flags above. Done in one procedure to reduce 
    overhead of groupby and df copy to create overall speedup.'''
    print('Adding all flags')
    df = df.copy()
    columns_set = set(df.columns)
    for flag in (IS_SAME_DAY, IS_BOOKKEEPING, IS_REPLICA, NTBC_PRECURSOR):
        if flag not in columns_set: df[flag] = False
    df['par_traded'] = df['par_traded'].astype(np.float32)    # `par_traded` type is Category so need to change it order to sum up
    groups_by_day_cusip = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip'], observed=True)
    for _, group_df_day_cusip in groups_by_day_cusip:
        df = _add_same_day_flag_for_group_v2(group_df_day_cusip, IS_SAME_DAY, df)
        for _, group_price_quantity in group_df_day_cusip.groupby(['dollar_price', 'quantity'], observed=True):
            df = _add_ntbc_precursor_flag_for_group_v2(group_price_quantity, NTBC_PRECURSOR, df)
            for group_header, group_df_day_cusip_price_tradetype_quantity in group_price_quantity.groupby('trade_type', observed=True):
                df = _add_replica_flag_for_group_v2(group_df_day_cusip_price_tradetype_quantity, IS_REPLICA, df)
                if group_header == 'D': df = _add_replica_flag_for_group_v2(group_df_day_cusip_price_tradetype_quantity, IS_BOOKKEEPING, df)    # index 0 corresponds to trade_type
    return df


def add_all_flags_v3(df):
    '''Procedure that adds all of the flags above. Done in one procedure to reduce 
    overhead of groupby and df copy to create overall speedup.'''
    print('Adding all flags')
    df = df.copy()
    columns_set = set(df.columns)
    for flag in (IS_SAME_DAY, IS_BOOKKEEPING, IS_REPLICA, NTBC_PRECURSOR):
        if flag not in columns_set: df[flag] = False
    df = add_same_day_flag_v3(df)
    df = add_ntbc_precursor_flag_v3(df)
    df = add_replica_flag_v3(df)
    df = add_bookkeeping_flag_v3(df)
    return df


def add_all_flags_v4(df):
    '''Procedure that adds all of the flags above. Done in one procedure to reduce 
    overhead of groupby and df copy to create overall speedup.'''
    df = df.copy()
    df = add_same_day_flag_v4(df)
    df = add_ntbc_precursor_flag_v4(df)
    df = add_replica_flag_v4(df)
    df = add_bookkeeping_flag_v4(df)
    return df


def add_all_flags_v5(df):
    '''Procedure that adds all of the flags above. Done in one procedure to reduce 
    overhead of groupby and df copy to create overall speedup.'''
    df = df.copy()
    df = add_same_day_flag_v5(df)
    df = add_ntbc_precursor_flag_v5(df)
    df = add_replica_flag_v5(df)
    df = add_bookkeeping_flag_v5(df)
    return df