'''
Description: Source code to process trade history from BigQuery
'''
from datetime import datetime
import pandas as pd

from modules.ficc.utils.process_features import process_features
from modules.ficc.data.process_trade_history import process_trade_history, process_similar_trade_history
from modules.ficc.utils.auxiliary_functions import convert_dates
from modules.ficc.utils.get_treasury_rate import current_treasury_rate, get_treasury_rate
from modules.ficc.utils.trade_history_features import trade_history_derived_features, get_trade_history_derived_features_for_ys_model, get_trade_history_derived_features_for_dp_model
from modules.ficc.utils.nelson_siegel_model import get_ficc_ycl_for_target_trade


def add_derived_features(df: pd.DataFrame, current_datetime: datetime) -> pd.DataFrame:
    '''`current_datetime` is used solely for caching. The caching occurs further downstream.'''
    dollar_price_model_used = df['model_used'] == 'dollar_price'
    features_to_call_trade_history_derived_features = ['cusip', 'quantity', 'trade_type']

    def add_derived_features_for_model(df: pd.DataFrame, ys_or_dp: str) -> pd.DataFrame:
        assert ys_or_dp in ('ys', 'dp'), f'`ys_or_dp` must be either "ys" or "dp" but is: "{ys_or_dp}"'
        trade_history_column_name = 'trade_history' if ys_or_dp == 'ys' else 'trade_history_dollar_price'
        if len(df) > 0:
            temp = df[features_to_call_trade_history_derived_features + [trade_history_column_name]].apply(trade_history_derived_features, axis=1, args=[current_datetime, ys_or_dp])     # `current_datetime` is used solely for caching
            trade_history_derived_features_names = get_trade_history_derived_features_for_ys_model() if ys_or_dp == 'ys' else get_trade_history_derived_features_for_dp_model()
            with pd.option_context('mode.chained_assignment', None):    # ignore `SettingWithCopyWarning` for below line of code
                df[trade_history_derived_features_names] = pd.DataFrame(temp.values.tolist(), index=df.index)
        return df

    df_yield_spread = add_derived_features_for_model(df[~dollar_price_model_used], 'ys')
    df_dollar_price = add_derived_features_for_model(df[dollar_price_model_used], 'dp')
    df = pd.concat([df_yield_spread, df_dollar_price])
    return df.sort_index()


def process_data(data: pd.DataFrame,
                 current_datetime: datetime,    # used solely for caching
                 client,
                 num_trades_in_yield_spread_history: int,
                 num_trades_in_dollar_price_history: int, 
                 num_features_per_trade: int, 
                 min_trades_in_history: int = 0, 
                 process_ratings: bool = False, 
                 treasury_rate_df: pd.DataFrame = None, 
                 holidays=None, 
                 is_batch_pricing: bool = False, 
                 use_similar_trade_history: bool = True) -> pd.DataFrame:
                #  **kwargs):
    '''NOTE: `current_datetime` should be the same as the `trade_datetime` column in `data` (in situations where the user 
    is not intentionally trying to set the values to be different, e.g., point-in-time pricing). `current_datetime` is 
    used solely for caching.'''
    if treasury_rate_df is None: treasury_rate_df = get_treasury_rate(client)
    if (data['trade_datetime'] != current_datetime).any():    # entering this statement means that `current_datetime` is actually the `model_datetime`
        print(f'Performing point-in-time pricing since `current_datetime` is not equal to `data["trade_datetime"]`, and so preventing `current_datetime` from being used for caching.\n`current_datetime`: {current_datetime}\n`data["trade_datetime"]`:\n{data["trade_datetime"].tolist()}')
        current_datetime = None    # ensures that value is reset

    def process_data_for_trade_datetime(df: pd.DataFrame, current_datetime: datetime):
        '''Executes all of the sub-procedures for `process_data(...)` for a specific `current_datetime`. Structured as a 
        sub-function of `process_data(...)` so that it can be called with different `current_datetime`s and benefit from 
        caching that can only occur for a `current_datetime` that equals the `trade_datetime` of all of the items in the 
        dataframe.'''
        df = process_trade_history(df, 
                                   current_datetime, 
                                   num_trades_in_yield_spread_history, 
                                   num_trades_in_dollar_price_history, 
                                   num_features_per_trade,  
                                   min_trades_in_history, 
                                   process_ratings, 
                                   treasury_rate_df, 
                                   holidays, 
                                   is_batch_pricing)
        columns_needed_to_compute_duration = ['trade_date', 'trade_datetime', 'last_calc_day_cat', 'is_called', 'is_callable', 'refund_date', 'next_call_date', 'par_call_date', 'maturity_date']    # used to call `ficc.utils.yc_data.py::get_duration(...)`
        df['ficc_ycl'] = df[['cusip']    # `cusip` is used solely for caching
                            + columns_needed_to_compute_duration].apply(get_ficc_ycl_for_target_trade, axis=1, args=[current_datetime])    # yield curve level for the most recent trade in history; `current_datetime` is used solely for caching
        df['treasury_rate'] = df[['cusip']     # `cusip` is used solely for caching
                                 + columns_needed_to_compute_duration].apply(current_treasury_rate, axis=1, args=[current_datetime, treasury_rate_df])    # `current_datetime` is used solely for caching
        df['ficc_treasury_spread'] = df['ficc_ycl'] - df['treasury_rate'] * 100

        if use_similar_trade_history:
            df = process_similar_trade_history(df, 
                                               current_datetime, 
                                               num_trades_in_yield_spread_history, 
                                               num_features_per_trade,    # yield_spread, treasury_rate, par_traded, trade_type1, trade_type2, seconds_ago
                                               min_trades_in_history, 
                                               treasury_rate_df, 
                                               holidays)
        df = convert_dates(df)
        df = process_features(df)
        df = add_derived_features(df, current_datetime)
        return df
    
    return pd.concat([process_data_for_trade_datetime(df, trade_datetime) for trade_datetime, df in data.groupby('trade_datetime')]).sort_index()    # `.sort_index()` preserves the original ordering
