'''
 '''
import warnings

import pandas as pd


DEFAULT_REFUND_PRICE = 100


def end_date_for_called_bond(trade: pd.Series):
    '''This function provides the end date for a called bond.'''
    if not pd.isnull(trade.refund_date):
        return trade.refund_date
    else:
        raise ValueError(f'CUSIP: {trade.cusip} is called, but has no refund date.')


def refund_price_for_called_bond(trade: pd.Series):
    '''This function provides the par value for a called bond.'''
    if not pd.isnull(trade.refund_price):
        return trade.refund_price
    else:
        warnings.warn(f'CUSIP: {trade.cusip} is called, but has no refund price. Using {DEFAULT_REFUND_PRICE} as the refund price.', RuntimeWarning)
        return DEFAULT_REFUND_PRICE
