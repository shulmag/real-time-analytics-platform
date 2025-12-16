'''
 #  length equal to the sequence length. The function pads the end of trade history and creates 
 #  a single sequence. The paddings are added after the most recent trades.
 # 
 #  If the length of the trade history is equal to the sequence length the function returns
 #  the list as is. As an initial step, we are only padding trades that have at least half the 
 #  sequence length number of trades in the sequence. We will expand the model to include comps for 
 #  CUSIPs which do not have sufficient history
 '''

import numpy as np

def pad_trade_history(x, SEQUENCE_LENGTH, NUM_FEATURES, min_trades_in_history):
    if len(x) == 0 and min_trades_in_history ==0:
        temp = [[0]*NUM_FEATURES]*(SEQUENCE_LENGTH)
        return np.stack(temp)
    
    elif len(x) < SEQUENCE_LENGTH and len(x) >= min_trades_in_history: 
        temp = x.tolist()
        temp = temp + [[0]*NUM_FEATURES]*(SEQUENCE_LENGTH - len(x))
        try:
            return np.stack(temp)
        except Exception as e:
            print("Failed to pad trade history for")
            for i in temp:
                print(i)
                
    #returning none for data less than the minimum required number of trades in history
    elif len(x) < min_trades_in_history:
        # TODO call the similarity model to get min_trades_in_history similar trades
        return None
    else:
        return x
