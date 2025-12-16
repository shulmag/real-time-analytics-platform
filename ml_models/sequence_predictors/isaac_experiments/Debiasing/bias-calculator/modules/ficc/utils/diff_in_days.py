'''
 # days between two days in accordance to the provision of MSRB rule 33G
 '''

import pandas as pd

'''
This function calculates the difference in days using the 360/30 
convention specified in MSRB Rule Book G-33, rule (e). 
Note that we only handle the 360/30 convention for date calculations.
'''
def diff_in_days_two_dates(end_date, start_date, convention="360/30"):
    if convention != "360/30":
        print("unknown convention", convention)
        return None

    # print(f"end_date:{end_date}")
    # print(f"start_date:{start_date}")
    # end_date = end_date.date()
    # start_date = start_date.date()

    Y2 = end_date.year
    Y1 = start_date.year
    M2 = end_date.month
    M1 = start_date.month
    D2 = end_date.day
    D1 = start_date.day
    D1 = min(D1, 30)
    if D1 == 30: 
        D2 = min(D2, 30)
    return (Y2 - Y1) * 360 + (M2 - M1) * 30 + (D2 - D1)

def diff_in_days(trade, convention="360/30", **kwargs):
    #See MSRB Rule 33-G for details
    if 'calc_type' in kwargs:
        if kwargs['calc_type'] == 'accrual' and not pd.isnull(trade.accrual_date):
            start_date = trade.accrual_date
            end_date = trade.settlement_date
        else:
            raise ValueError('Invalid arguments')
    else:
        start_date = trade.dated_date
        end_date = trade.settlement_date

    return diff_in_days_two_dates(end_date, start_date, convention)
