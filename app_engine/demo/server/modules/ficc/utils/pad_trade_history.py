'''
 # @ Create date: 2021-12-16
 # @ Modified date: 2024-01-16
 #  length equal to the sequence length. The function pads the end of trade history and creates 
 #  a single sequence. The paddings are added after the most recent trades.
 # 
 #  If the length of the trade history is equal to the sequence length the function returns
 #  the list as is. As an initial step, we are only padding trades that have at least half the 
 #  sequence length number of trades in the sequence. We will expand the model to include comps for 
 #  CUSIPs which do not have sufficient history
 '''
import numpy as np


def pad_trade_history(trade_history, num_trades_in_history, num_features_per_trade, min_trades_in_history=0):
    num_trades_currently_in_trade_history = len(trade_history)
    if num_trades_currently_in_trade_history == 0 and min_trades_in_history == 0:
        trade_history = [[0] * num_features_per_trade] * num_trades_in_history
        return np.stack(trade_history)
    elif num_trades_currently_in_trade_history < num_trades_in_history and num_trades_currently_in_trade_history >= min_trades_in_history: 
        trade_history = trade_history.tolist()
        trade_history = trade_history + [[0] * num_features_per_trade] * (num_trades_in_history - num_trades_currently_in_trade_history)
        try:
            return np.stack(trade_history)
        except Exception as e:
            print('Failed to pad trade history for')
            for trade in trade_history: print(trade)
            print(e)
    elif num_trades_currently_in_trade_history < min_trades_in_history:    # returning `None` for data less than the minimum required number of trades in history
        return None
    else:    # no padding necessary
        return trade_history
