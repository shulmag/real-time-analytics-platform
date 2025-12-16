import pandas as pd
import numpy as np
import locale
import json

from dateutil.relativedelta import relativedelta

locale.setlocale( locale.LC_ALL, 'en_US' )

#import os
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/Gil/git/ficc/data_loader/Cusip Global Service Importer-2fdcdfc4edba.json"

COUPON_FREQUENCY_DICT = {0:None,
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

def get_price(cusip, my_prev_coupon_date, first_coupon_date, my_next_coupon_date, end_date,
              settlement_date, dated_date, frequency, ytw, coupon, RV, time_delta):
#     if cusip == "34061YAH3": my_prev_coupon_date += relativedelta(weeks = 2)
        
    if pd.isnull(end_date): return np.inf
    B = 360
    Y = ytw/100
    
    if frequency == 0:
        A = diff_in_days(settlement_date,dated_date)
        accrued = coupon*A/B        
        duration = diff_in_days(end_date,settlement_date)
        periods = duration/(B/2)
        denom = pow(1 + Y/2, periods)
        DIR = diff_in_days(end_date,dated_date)
        base = (RV + coupon*DIR/B) / denom
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
            print(cusip, A, settlement_date,my_prev_coupon_date)
            
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
    
    return round(P,3)

def compute_price(trade):
    
    frequency = trade.interest_payment_frequency
    if not frequency >= 0: # includes null frequency
        anomaly = True
        frequency = 2

    if frequency == 0:
        time_delta = 0
        my_next_coupon_date = trade.maturity_date
        my_prev_coupon_date = trade.dated_date
    else:
        time_delta = relativedelta(months = 12/frequency)
        if not trade.first_coupon_date >= trade.dated_date:
            anomaly = True
            print("bad first coupon date:", trade.cusip, trade.first_coupon_date)
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
            if trade.redemption_type in [1,5]:
                end_date = trade.maturity_date
            else:
                end_date = trade.next_call_date
        else:
            end_date = trade.refund_date
            
        if trade.refund_date < trade.settlement_date:
            anomaly = True
            print("anomalous refund date:", trade.cusip, "settlement:", trade.settlement_date, "refunding:", trade.refund_date)
            
        if not pd.isnull(trade.refund_price):
            par = trade.refund_price
        elif not pd.isnull(trade.next_call_price): 
            par = trade.next_call_price
        else: 
            par = 100
        final = get_price(trade.cusip,my_prev_coupon_date,trade.first_coupon_date,my_next_coupon_date,
                         end_date,trade.settlement_date, trade.dated_date, frequency,
                         trade['yield'], trade.coupon_rate, par, time_delta)
        calc = "refunding"
    else:
    
        next_price = get_price(trade.cusip,my_prev_coupon_date,trade.first_coupon_date,my_next_coupon_date,
                            trade.next_call_date,trade.settlement_date,trade.dated_date, frequency,
                            trade['yield'],trade.coupon_rate,trade.next_call_price,time_delta)

        to_par_price = get_price(trade.cusip,my_prev_coupon_date,trade.first_coupon_date,my_next_coupon_date,
                            trade.par_call_date,trade.settlement_date,trade.dated_date, frequency,
                            trade['yield'],trade.coupon_rate,trade.par_call_price,time_delta)

        maturity_price = get_price(trade.cusip,my_prev_coupon_date,trade.first_coupon_date,my_next_coupon_date,
                            trade.maturity_date,trade.settlement_date,trade.dated_date, frequency,
                            trade['yield'],trade.coupon_rate,100,time_delta)
        
        final = next_price; calc = "next call"
        if to_par_price < final:
            final = to_par_price; calc = "par call"
        if maturity_price < final:
            final = maturity_price; calc = "maturity"
        
    i: print(locals(), "\n==============")
    return final, calc

def transform_ref_data(df):
    df['interest_payment_frequency'] = df['interest_payment_frequency'].map(COUPON_FREQUENCY_DICT)
    df['coupon_rate'] = df['coupon_rate'].astype(float)
    df['yield'] = df['yield'].astype(float)
    df['deferred'] = (df.interest_payment_frequency == 0) | df.coupon_rate == 0
    
    df['next_call_price'] = df['next_call_price'].astype(float)
    fields = ['dated_date', 'first_coupon_date', 'settlement_date',
              'next_call_date', 'par_call_date', 'refund_date', 'maturity_date']
    for f in fields: df[f] = pd.to_datetime(df[f])
    return df

def main(request):
    request_json = request.get_json(silent=True)
    df = pd.DataFrame(data=request_json,index=[1])
    df = transform_ref_data(df)
    final, calc = compute_price(df.iloc[0])
    return {"price":final, "calc": calc}