'''
 # function to unpack the list of dictionaries and creates a list of historical trades. 
 # With each element in the list containing all the information for that particular trade
 '''
import numpy as np
from modules.ficc.utils.trade_dict_to_list import trade_dict_to_list

# from utils.trade_dict_to_list import trade_dict_to_list

def trade_list_to_array(trade_history, trade_datetime, treasury_rate_df):
    
    if len(trade_history) == 0:
        return np.array([]), [None]*16, []

    trades_list = []
    last_trade_features = None
    previous_trades_features = []
    for entry in trade_history:
        trades, temp_last_features = trade_dict_to_list(entry,
                                                        trade_datetime, 
                                                        treasury_rate_df)
                                    
        if trades is not None:
            trades_list.append(trades)
            previous_trades_features.append(list(temp_last_features))
        
        
        if last_trade_features is None:
            last_trade_features = temp_last_features
        
    if len(trades_list) > 0:
        try:
            return np.stack(trades_list), last_trade_features, np.stack(previous_trades_features)
        except Exception as e:
            for i in trades_list:
                print(i)
            for i in last_trade_features:
                print(i)
            raise e
    else:
        return [], [None]*16, []