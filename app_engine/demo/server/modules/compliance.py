'''
Description: Functions that support compliance.
'''
import numpy as np
import pandas as pd

from modules.errors import CustomMessageError


RATINGS = ['Great', 'Good', 'Fair', 'Poor']


def determine_compliance_rating(row: pd.Series) -> str:
    if 'bid_ask_price_delta' not in row.index or np.isnan(row['bid_ask_price_delta']): return pd.NA    # `bid_ask_price_delta` not existing or being NaN means that the CUSIP was not priced due to some upstream error
    bid_ask_price_delta = row['bid_ask_price_delta']
    price = row['price']
    user_price = row['user_price']
    trade_type = row['trade_type']

    # `compliance_side` may not exist as a column, but is expected to exist if `trade_type` is `D`
    if trade_type != 'D':    # `trade_type` is 'P' or 'S'
        compliance_side = trade_type
    else:    # `trade_type` is 'D'
        if 'compliance_side' in row.index:
            compliance_side = row['compliance_side']
        else:
            print(f'WARNING: calling the compliance module with an inter-dealer trade with no explict `compliance_side`, so using the default compliance side of "S"')
            compliance_side = 'S'
    
    if compliance_side == 'S':    # Offered Side
        if user_price < price:
            return RATINGS[0]
        elif price <= user_price <= price + bid_ask_price_delta:
            return RATINGS[1]
        elif price + bid_ask_price_delta < user_price <= price + 2 * bid_ask_price_delta:
            return RATINGS[2]
        else:
            return RATINGS[3]
    elif compliance_side == 'P':    # Bid Side
        if user_price > price:
            return RATINGS[0]
        elif price - bid_ask_price_delta <= user_price <= price:
            return RATINGS[1]
        elif price - 2 * bid_ask_price_delta <= user_price < price - bid_ask_price_delta:
            return RATINGS[2]
        else:
            return RATINGS[3]
    raise CustomMessageError(f'Incompatible side of trade to use for compliance: {compliance_side}')
