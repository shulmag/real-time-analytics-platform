'''
 '''
import pandas as pd

# Pandaralled is a python package that is 
# used to multi-thread df apply
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)

from modules.ficc.utils.process_features import process_features
import modules.ficc.utils.globals as globals
from modules.ficc.utils.nelson_siegel_model import yield_curve_level
from modules.ficc.utils.diff_in_days import diff_in_days_two_dates
from modules.ficc.data.process_trade_history import process_trade_history
from modules.ficc.utils.auxiliary_functions import convert_dates, create_current_trade_array, append_last_trade
from modules.ficc.utils.get_treasury_rate import current_treasury_rate, get_all_treasury_rate

from modules.ficc.utils.auxiliary_variables import RELATED_TRADE_FEATURE_PREFIX, NUM_RELATED_TRADES, CATEGORICAL_REFERENCE_FEATURES_PER_RELATED_TRADE, NUM_OF_DAYS_IN_YEAR
from modules.ficc.utils.trade_history_features import trade_history_derived_features, get_trade_history_columns


# This is used to debug, please comment before deploying
# from utils.process_features import process_features
# import utils.globals as globals
# from utils.nelson_siegel_model import yield_curve_level
# from utils.diff_in_days import diff_in_days_two_dates
# from data.process_trade_history import process_trade_history
# from utils.auxiliary_functions import convert_dates, create_current_trade_array, append_last_trade
# from utils.get_treasury_rate import current_treasury_rate
# from utils.get_treasury_rate import get_all_treasury_rate
# from utils.related_trades import add_related_trades
# from utils.auxiliary_variables import RELATED_TRADE_FEATURE_PREFIX, NUM_RELATED_TRADES, CATEGORICAL_REFERENCE_FEATURES_PER_RELATED_TRADE, NUM_OF_DAYS_IN_YEAR
# from utils.trade_history_features import trade_history_derived_features, get_trade_history_columns

# This computes the yield curve level for the target trade using the last duration
def get_ficc_ycl_for_target_trade(row):
    if row['last_calc_date'] is not None and row['last_trade_datetime'] is not None:
        time_to_maturity = diff_in_days_two_dates(row['last_calc_date'],row['last_trade_datetime'].date())/NUM_OF_DAYS_IN_YEAR
    else:
        time_to_maturity = diff_in_days_two_dates(row['maturity_date'],row['trade_datetime'].date())/NUM_OF_DAYS_IN_YEAR
    return yield_curve_level(time_to_maturity,row['trade_datetime'].strftime('%Y-%m-%d:%H:%M'))

def process_data(data,
                 trade_datetime,
                 client,
                 SEQUENCE_LENGTH,
                 NUM_FEATURES,
                 YIELD_CURVE="FICC", 
                 remove_short_maturity = True,
                 min_trades_in_history=0,
                 process_ratings=False, 
                 treasury_rate_df=None, 
                 is_batch_pricing=False, 
                 **kwargs):
    trades_df = process_trade_history(data,
                                      trade_datetime,
                                      client,
                                      SEQUENCE_LENGTH,
                                      NUM_FEATURES,
                                      remove_short_maturity,
                                      min_trades_in_history,
                                      process_ratings, 
                                      treasury_rate_df, 
                                      is_batch_pricing)

    # Yield curve level for the most recent trade in history
    trades_df['ficc_ycl'] = trades_df.apply(get_ficc_ycl_for_target_trade, axis=1)

    # Adding the treasury rate
    temp = trades_df[['trade_date']].apply(get_all_treasury_rate, axis=1, result_type='expand', args=[treasury_rate_df])
    trades_df[['t_rate_1',
               't_rate_2', 
               't_rate_3', 
               't_rate_5', 
               't_rate_7', 
               't_rate_10', 
               't_rate_20', 
               't_rate_30']] = temp
    
    trades_df['treasury_rate'] = trades_df[['trade_date', 'last_calc_date', 'settlement_date', 'maturity_date']].apply(current_treasury_rate, axis=1, args=[treasury_rate_df])
    
    trades_df['ficc_treasury_spread'] = trades_df['ficc_ycl'] - trades_df['treasury_rate'] * 100

    trades_df = convert_dates(trades_df)
    trades_df = process_features(trades_df)

    YS_COLS = get_trade_history_columns()
    temp = trades_df[['cusip','trade_history','quantity','trade_type']].apply(trade_history_derived_features, axis=1)
    trades_df[YS_COLS] = pd.DataFrame(temp.tolist(), index=trades_df.index)

    return trades_df
