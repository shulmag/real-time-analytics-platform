import pandas as pd
from pandas import NaT
import pyarrow
import pytz

import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook

import os
import numpy as np
from google.cloud import bigquery
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import scipy.optimize as optimize

tqdm.pandas()
bqclient = bigquery.Client()

PROJECT_ID = "eng-reactor-287421"
TABLE_ID = "eng-reactor-287421.reference_data.price_deltas"


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
                         13:44,
                         14:360,
                         16:0,
                         23:None}

def get_trade_data(bqclient,trade_date):
    query = f'''
    SELECT
                cusip,
                rtrs_control_number,
                trade_datetime,
                trade_date,
                time_of_trade,
                trade_type,
                par_traded,
                dollar_price,
                yield,
                coupon AS coupon_rate,
                interest_payment_frequency,
                dated_date,
                settlement_date, 
                first_coupon_date,
                is_callable,
                next_call_date,
                next_call_price,
                par_call_date,
                par_call_price,
                is_called,
                called_redemption_type AS redemption_type,
                refund_date,
                refund_price,
                maturity_date,
                FROM `eng-reactor-287421.primary_views.trade_history_with_reference_data`
                WHERE MSRB_valid_to_date > current_date
                AND trade_date = '{trade_date}'
                '''
    dataframe = bqclient.query(query).result().to_dataframe()
    return dataframe 

def transform_trade_data(df):
    df['interest_payment_frequency']= df['interest_payment_frequency'].map(COUPON_FREQUENCY_DICT)
    
    df['coupon_rate'] = df['coupon_rate'].astype(float)
    df['next_call_price'] = df['next_call_price'].astype(float)
    fields = ['trade_date', 'dated_date', 'settlement_date', 'first_coupon_date', 
              'next_call_date', 'par_call_date', 'refund_date', 'maturity_date']
    for f in fields: df[f] = pd.to_datetime(df[f])
    return df

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

def get_price(cusip, dollar_price, my_prev_coupon_date, first_coupon_date, my_next_coupon_date, end_date,
              settlement_date, dated_date, frequency, ytw, coupon, RV, time_delta, printme):
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
                
    if printme: 
        delta = dollar_price - P; print("\n", locals())
    
    return round(P,3)

def compute_price(trade):
    anomaly = trade.anomaly
    printme = trade.alert
    
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
            printme = True
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
            printme = True
            print("anomalous refund date:", trade.cusip, "settlement:", trade.settlement_date, "refunding:", trade.refund_date)
            
        if not pd.isnull(trade.refund_price):
            par = trade.refund_price
        elif not pd.isnull(trade.next_call_price): 
            par = trade.next_call_price
        else: 
            par = 100
        final = get_price(trade.cusip,trade.dollar_price,my_prev_coupon_date,trade.first_coupon_date,my_next_coupon_date,
                         end_date,trade.settlement_date, trade.dated_date, frequency,
                         trade['yield'], trade.coupon_rate, par, time_delta, printme)
        calc = "refunding"
    else:
    
        next_price = get_price(trade.cusip,trade.dollar_price,my_prev_coupon_date,trade.first_coupon_date,my_next_coupon_date,
                            trade.next_call_date,trade.settlement_date,trade.dated_date, frequency,
                            trade['yield'],trade.coupon_rate,trade.next_call_price,time_delta, printme)

        to_par_price = get_price(trade.cusip,trade.dollar_price,my_prev_coupon_date,trade.first_coupon_date,my_next_coupon_date,
                            trade.par_call_date,trade.settlement_date,trade.dated_date, frequency,
                            trade['yield'],trade.coupon_rate,trade.par_call_price,time_delta, printme)

        maturity_price = get_price(trade.cusip,trade.dollar_price,my_prev_coupon_date,trade.first_coupon_date,my_next_coupon_date,
                            trade.maturity_date,trade.settlement_date,trade.dated_date, frequency,
                            trade['yield'],trade.coupon_rate,100,time_delta, printme)
        
        final = next_price; calc = "next call"
        if to_par_price < final:
            final = to_par_price; calc = "par call"
        if maturity_price < final:
            final = maturity_price; calc = "maturity"
        
    if printme: print(locals(), "\n==============")
    return final, calc, anomaly

def get_latest_trade_datetime():
    query = f'''
    SELECT
    trade_datetime
    FROM
    `eng-reactor-287421.reference_data.price_deltas`
    order by trade_datetime desc  limit 1
    '''
    query_job = bqclient.query(query).result().to_dataframe()
    query_job = query_job.values[0][0]
    query_job = datetime.datetime.utcfromtimestamp(query_job.tolist()/1e9)
    return query_job

def getSchema():
    schema = [  bigquery.SchemaField("rtrs_control_number", "INTEGER"),
                bigquery.SchemaField("trade_datetime", "DATETIME"),
                bigquery.SchemaField("cusip", "STRING"),
                bigquery.SchemaField('my_price',"FLOAT"),
                bigquery.SchemaField('price_delta', "FLOAT")
            ]
    return schema


def uploadData(vanilla):
    client = bigquery.Client(project=PROJECT_ID, location="US")
    useful_columns = vanilla[["rtrs_control_number", "trade_datetime", "cusip",'my_price','price_delta']]

    job_config = bigquery.LoadJobConfig(schema = getSchema(),
                                       write_disposition="WRITE_APPEND"
                                       )
    
    job = client.load_table_from_dataframe(useful_columns, TABLE_ID,job_config=job_config)
    
    try:
        job.result()
        print("Upload Successful")
    except Exception as e:
        print("Failed to Upload")
        raise e

def main (args):
    trade_date = date.today()
    df = get_trade_data(bqclient,trade_date)
    latest_trade_datetime = get_latest_trade_datetime()
    df = df[df.trade_datetime>latest_trade_datetime]
    vanilla = transform_trade_data(df) 
    vanilla['anomaly'] = (vanilla.par_traded < 5000) | (vanilla['yield'] < 0) 
    vanilla['anomaly'] = vanilla['anomaly'] | pd.isnull(vanilla.settlement_date) | pd.isnull(vanilla.first_coupon_date) & (vanilla.coupon_rate > 0)
    vanilla['alert'] = False # vanilla.cusip == "89386FAD5" # vanilla.redemption_type == 5 # vanilla.cusip == "803093AM5" # 
    vanilla['my_price'], vanilla['my_date'], vanilla['anomaly'] = zip(*vanilla.progress_apply(lambda x: compute_price(x),axis=1))
    vanilla['price_delta'] = abs(vanilla.my_price - vanilla.dollar_price)
    uploadData(vanilla)



