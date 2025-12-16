import pandas as pd
import numpy as np
import json
import ficc_globals as globals

from dateutil.relativedelta import relativedelta

PRICE_COUPON_FREQUENCY_DICT = {0:None,
                         1:2,
                         2:12,
                         3:1,
                         4:52,
                         5:4,
                         6:0.5,
                         7:1/3,
                         8:1/4,
                         9:1/5,
                         10:1/7,
                         11:1/8,
                         12:26,
                         13:None,
                         14:360,
                         16:0,
                         23:None}

# We handle only the 360/30 convention for date calculations. For details, see MSRB Rule 33-G.

def diff_in_days(end_date, start_date, convention="360/30"):
    if convention != "360/30": 
        print("unknown convention", convention)
    Y2 = end_date.year
    Y1 = start_date.year
    M2 = end_date.month
    M1 = start_date.month
    D2 = end_date.day 
    D1 = min(start_date.day, 30)
    if D1 == 30: 
        D2 = min(D2,30)
    return (Y2 - Y1) * 360 + (M2 - M1) * 30 + (D2 - D1)

def get_next_coupon_date(first_coupon_date,settlement_date,time_delta):
    date = first_coupon_date
    while date < settlement_date:
        date = date + time_delta
    return date
    

# The following function calculates the price of a trade, where _ytw_ is a specific yield and _end__date_ is a fixed repayment date.
# All dates must be valid relative to the settlement date, as opposed to the trade date.
# Note that "yield" is a reserved word in Python and should not be used as the name of a variable or column.
#
# Formulas are from https://www.msrb.org/pdf.aspx?url=https%3A%2F%2Fwww.msrb.org%2FRules-and-Interpretations%2FMSRB-Rules%2FGeneral%2FRule-G-33.aspx.
# For all bonds, _base_ is the present value of future cashflows to the buyer. 
# The clean price is this price minus the accumulated amount of simple interest that the buyer must pay to the seller, which is called _accrued_.
# Zero-coupon bonds are handled first. For these, the yield is assumed to be compounded semi-annually, i.e., once every six months.
# For bonds with non-zero coupon, the first and last interest payment periods may have a non-standard length,
# so they must be handled separately.
def get_price(cusip, my_prev_coupon_date, first_coupon_date, my_next_coupon_date, end_date,
              settlement_date, dated_date, frequency, ytw, coupon, RV, time_delta):        
    #ABOVE: dollar_price and printme not in GIL's iteration. 
    if pd.isnull(end_date): return np.inf
    B = 360
    Y = ytw/100
    
    if frequency == 0:
        A = diff_in_days(settlement_date,dated_date)
        accrued = coupon*A/B        
        duration = diff_in_days(end_date,settlement_date)
        periods = duration/(B/2)
        denom = pow(1 + Y/2, periods)
        life = diff_in_days(end_date,dated_date)
        base = (RV + coupon*life/B) / denom
        P = base - accrued
    else:
        if my_next_coupon_date > end_date: N = 0
        else:
            N = 1
            final_coupon_date = my_next_coupon_date
            while final_coupon_date + time_delta <= end_date:
                N += 1
                final_coupon_date += time_delta            

        A = diff_in_days(settlement_date,my_prev_coupon_date)
        if A < 0:
            globals.FICC_ERROR = f"bad previous coupon date:,cusip: {cusip}, A: {A}, settlement_date:{settlement_date}, my_prev_coupon_date: {my_prev_coupon_date}"
            
        accrued = coupon*A/B
        E = B/frequency           # = number of days in interest payment period 
        assert E == round(E)
        
        F = diff_in_days(my_next_coupon_date,settlement_date)
        if my_next_coupon_date == first_coupon_date:
            G = diff_in_days(first_coupon_date,dated_date)
        else:
            G = E

        if end_date <= my_next_coupon_date:
            D = diff_in_days(end_date,settlement_date) 
            H = diff_in_days(end_date,my_prev_coupon_date) 
            base = (RV + coupon*H/B) / (1 + (Y/frequency)*D/E)
        else:
            D = diff_in_days(end_date,final_coupon_date) 
            S1 = (RV + coupon*D/B) / pow(1 + Y/frequency, F/E + N - 1 + D/E)

            S2 = coupon*G/B / pow(1 + Y/frequency, F/E)
            for K in range(2,N+1):
                S2 += coupon*E/B / pow(1 + Y/frequency, F/E + K - 1)
            base = S1 + S2
        P = base - accrued
                
    # if printme: 
    #     delta = dollar_price - P; print("\n", locals())
    
    # this is new. Is it needed?
    return round(P,3)
    
    
# The following function computes facts about a bond.
# We use _first__coupon__date_ from the reference data, but we compute our own values for other coupon dates.
# For bonds that have been called, the date of refunding is often missing.
# For most types of redemption, this date is the next call date.
# For redemption types 1 and 5, the refunding date is the maturity date.
# However, MSRB sometimes uses the next call date for type 5,
# in which case our calculation gives a different result.
# Note that the condition "x >= y" evaluates to _False_ if either _x_ or _y_ is _NaT_.
# Therefore, "not trade.first_coupon_date >= trade.dated_date" is _True_ when _first__coupon__date_ is _NaT_.    


def fixup(trade):
    frequency = trade.interest_payment_frequency
    if not frequency >= 0: # includes null frequency
        frequency = 2

    if frequency == 0:
        time_delta = 0
        my_next_coupon_date = trade.maturity_date
        my_prev_coupon_date = trade.dated_date
    else:
        time_delta = relativedelta(months = 12/frequency)
        if not trade.first_coupon_date >= trade.dated_date:
            globals.FICC_ERROR = f"bad first coupon date:, {trade.cusip}, first_coupon_date: {trade.first_coupon_date}"
            my_first_date = trade.dated_date
        else:
            my_first_date = trade.first_coupon_date
            
        my_next_coupon_date = get_next_coupon_date(my_first_date, trade.settlement_date, time_delta)
        if my_next_coupon_date == trade.first_coupon_date:
            my_prev_coupon_date = trade.dated_date
        else:
            my_prev_coupon_date = my_next_coupon_date - time_delta

    if trade.is_called:
        if pd.isnull(trade.refund_date):
            if trade.called_redemption_type in [1,5]:
                end_date = trade.maturity_date
            else:
                end_date = trade.next_call_date
        else:
            end_date = trade.refund_date
            
        if end_date < trade.settlement_date:
            print("anomalous refund date:", trade.cusip, "settlement:", trade.settlement_date, "refunding:", trade.refund_date)
            
        if not pd.isnull(trade.refund_price):
            par = trade.refund_price
        elif not pd.isnull(trade.next_call_price): 
            par = trade.next_call_price
        else: 
            par = 100
    else:
        end_date = trade.maturity_date # not used later
        par = 100
    return frequency, time_delta, my_next_coupon_date, my_prev_coupon_date, end_date, par


# Next is the main function for computing prices. For bonds that have not been called, the price is the lowest of
# three present values: to the next call date (which may be above par), to the next par call date, and to maturity.

def compute_price(trade):
    frequency, time_delta, my_next_coupon_date, my_prev_coupon_date, end_date, par = fixup(trade)

    if trade.is_called:
        final = get_price(trade.cusip,  my_prev_coupon_date, trade.first_coupon_date, my_next_coupon_date,
                         end_date, trade.settlement_date, trade.dated_date, frequency, trade['yield'], trade.coupon_rate, 
                          par, time_delta)
        calc = "refunding"
    else:
    
        next_price = get_price(trade.cusip,  my_prev_coupon_date, trade.first_coupon_date, my_next_coupon_date,
                            trade.next_call_date, trade.settlement_date, trade.dated_date, frequency,
                            trade['yield'], trade.coupon_rate, trade.next_call_price, time_delta)

        to_par_price = get_price(trade.cusip,  my_prev_coupon_date, trade.first_coupon_date, my_next_coupon_date,
                            trade.par_call_date, trade.settlement_date, trade.dated_date, frequency,
                            trade['yield'], trade.coupon_rate, trade.par_call_price, time_delta)

        maturity_price = get_price(trade.cusip, my_prev_coupon_date, trade.first_coupon_date, my_next_coupon_date,
                            trade.maturity_date, trade.settlement_date, trade.dated_date, frequency,
                            trade['yield'], trade.coupon_rate, 100, time_delta)
        
        final = next_price; calc = "next call"
        if to_par_price < final:
            final = to_par_price; calc = "par call"
        if maturity_price < final:
            final = maturity_price; calc = "maturity"
        
    #i: print(locals(), "\n==============")
    return final, calc

def transform_ref_data(df):
    df['interest_payment_frequency'] = df['interest_payment_frequency'].map(PRICE_COUPON_FREQUENCY_DICT)
    df['coupon_rate'] = df['coupon'].astype(float)
    df['yield'] = df['yield'].astype(float)
    df['deferred'] = (df.interest_payment_frequency == 0) | df.coupon_rate == 0
    
    df['next_call_price'] = df['next_call_price'].astype(float)
    return df

#@njit(parallel=True)
def get_cusip_price(df):
    df = transform_ref_data(df)
    final, calc = compute_price(df.iloc[0])
    return final,calc

