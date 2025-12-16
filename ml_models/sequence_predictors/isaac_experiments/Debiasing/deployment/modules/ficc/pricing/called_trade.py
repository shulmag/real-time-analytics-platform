'''
 '''

import pandas as pd

'''
This function provides the end date for a called bond. 
'''
def end_date_for_called_bond(trade):
    if not pd.isnull(trade.refund_date):
        return trade.refund_date
    else:
        raise ValueError(f"Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) is called, but no refund date.")

'''
This function provides the par value for a called bond.
'''
def refund_price_for_called_bond(trade):
    if not pd.isnull(trade.refund_price):
        return trade.refund_price
    else:
        raise ValueError(f"Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) is called, but no refund price.")