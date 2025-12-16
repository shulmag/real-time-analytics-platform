'''
 '''
import pandas as pd

from modules.ficc.utils.pad_trade_history import pad_trade_history
from modules.ficc.utils.trade_list_to_array import trade_list_to_array
from modules.ficc.utils.auxiliary_functions import process_ratings
from modules.ficc.utils.get_treasury_rate import get_treasury_rate

# This is used to test, please comment before deploying
# from utils.pad_trade_history import pad_trade_history
# from utils.trade_list_to_array import trade_list_to_array
# from utils.auxiliary_functions import process_ratings
# from utils.get_treasury_rate import get_treasury_rate

def process_trade_history(trade_dataframe,
                          trade_datetime,
                          client, 
                          SEQUENCE_LENGTH, 
                          NUM_FEATURES, 
                          remove_short_maturity,   
                          min_trades_in_history, 
                          process_ratings_bool, 
                          treasury_rate_df, 
                          is_batch_pricing):
    if treasury_rate_df is None: treasury_rate_df = get_treasury_rate(client)

    trade_dataframe = process_ratings(trade_dataframe, process_ratings_bool)
    # Restricting to the Sequence length if batch pricing (doing it early will save us from having to apply `trade_list_to_array`)
    if is_batch_pricing: trade_dataframe.recent = trade_dataframe.recent.apply(lambda trade: trade[:SEQUENCE_LENGTH])
    
    # print('Starting history processing')
    if is_batch_pricing:    # do not perform parallel_apply since the parallelization is occurring more upstream
        temp = trade_dataframe.recent.apply(trade_list_to_array, args=([trade_datetime, treasury_rate_df]))
    else:
        temp = trade_dataframe.recent.parallel_apply(trade_list_to_array, args=([trade_datetime, treasury_rate_df]))
    trade_dataframe[['trade_history', 'temp_last_features', 'previous_trades_features']] = pd.DataFrame(temp.tolist(), index=trade_dataframe.index)
    
    del temp
    trade_dataframe[['last_yield_spread',
                     'last_ficc_ycl',
                     'last_rtrs_control_number',
                     'last_yield',
                     'last_dollar_price',
                     'last_seconds_ago',
                     'last_size',
                     'last_calc_date', 
                     'last_maturity_date', 
                     'last_next_call_date', 
                     'last_par_call_date', 
                     'last_refund_date',
                     'last_trade_datetime',
                     'last_calc_day_cat',
                     'last_settlement_date',
                     'last_trade_type']] = pd.DataFrame(trade_dataframe['temp_last_features'].tolist(), index=trade_dataframe.index)
    

    # print('Done history')
    trade_dataframe.drop(columns=['recent','temp_last_features'],inplace=True)

    # Restricting to the Sequence length if not batch pricing, which cannot be done early since we use the entire trade history on the front end when displaying past trades when a CUSIP is individually priced
    if not is_batch_pricing: trade_dataframe.trade_history = trade_dataframe.trade_history.apply(lambda trade: trade[:SEQUENCE_LENGTH])

    # Padding trade history
    trade_dataframe.trade_history = trade_dataframe.trade_history.apply(pad_trade_history, args=[SEQUENCE_LENGTH, NUM_FEATURES, min_trades_in_history])
     
    trade_dataframe.dropna(subset=['trade_history'], inplace=True)


    return trade_dataframe
