'''
 # to process training data
 '''
import pandas as pd
import numpy as np
from datetime import datetime

'''Quote a string twice: e.g., double_quote_a_string('hello') -> "'hello'". This 
function is used to put string arguments into formatted string expressions and 
maintain the quotation.'''
double_quote_a_string = lambda potential_string: f'"{str(potential_string)}"' if type(potential_string) == str else potential_string

def sqltodf(sql, bq_client):
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()


def drop_extra_columns(df):
    df.drop(columns=[
                 'sp_stand_alone',
                 'sp_icr_school',
                 'sp_icr_school',
                 'sp_icr_school',
                 'sp_watch_long',
                 'sp_outlook_long',
                 'sp_prelim_long',
                 'MSRB_maturity_date',
                 'MSRB_INST_ORDR_DESC',
                 'MSRB_valid_from_date',
                 'MSRB_valid_to_date',
                 'upload_date',
                 'sequence_number',
                 'ref_valid_from_date',
                 'ref_valid_to_date',
                 'additional_next_sink_date',
                 'last_period_accrues_from_date',
                 'primary_market_settlement_date',
                 'assumed_settlement_date',
                 'sale_date','q','d'],
                  inplace=True)
    
    
    return df


def convert_dates(df):
    date_cols = [col for col in list(df.columns) if 'DATE' in col.upper()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    
    return df

'''
This function  
'''
def process_ratings(df, process_ratings):
    # MR is for missing ratings
    df.sp_long.fillna('MR', inplace=True)
    if process_ratings == True:
        df = df[df.sp_long.isin(['BBB+','A-','A','A+','AA-','AA','AA+','AAA','NR','MR'])] 
    df['rating'] = df['sp_long']
    return df
    

'''
This function compares two date objects whether they are in Timestamp or datetime.date. 
The different types are causing a future warning. If date1 occurs after date2, return 1. 
If date1 equals date2, return 0. Otherwise, return -1.
'''
def compare_dates(date1, date2):
    if type(date1) == pd.Timestamp:
        date1 = date1.date()
    if type(date2) == pd.Timestamp:
        date2 = date2.date()
    
    if date1 > date2:
        return 1
    elif date1 == date2:
        return 0
    elif date1 < date2:
        return -1

'''
This function directly calls `compare_dates` to check if two dates are equal.
'''
def dates_are_equal(date1, date2):
    return compare_dates(date1, date2) == 0

'''
This function converts the columns with object datatypes to category data types
'''
def convert_object_to_category(df):
    print("Converting object data type to categorical data type")
    for col_name in df.columns:
        if col_name.endswith("event") or col_name.endswith("redemption") or col_name.endswith("history") or col_name.endswith("date") or col_name.endswith("issue"):
            continue

        if df[col_name].dtype == "object" and col_name not in ['organization_primary_name','security_description','recent','issue_text','series_name','recent_trades_by_series']:
            df[col_name] = df[col_name].astype("category")
    return df

def calculate_a_over_e(df):
    if not pd.isnull(df.previous_coupon_payment_date):
        A = (df.settlement_date - df.previous_coupon_payment_date).days
        return A/df.days_in_interest_payment
    else:
        return df['accrued_days']/360

def create_current_trade_array(row):
    ''' 
    This function is only used in production to calculate the current trade array.
    The array is added in the begining the trade history.
    Paramaterts:
    input: dataframe row
    output: numpy array
    '''
    current_trade_array = []
    trade_type_mapping = {'D':[0,0],'S': [0,1],'P': [1,0]}

    current_trade_array.append(row['yield'] * 100 - row['ficc_ycl'])
    current_trade_array.append((row['yield'] - row['treasury_rate']) * 100)
    current_trade_array.append(row['quantity'])
    current_trade_array += trade_type_mapping[row['trade_type']]
    current_trade_array.append(np.log10(1 + (datetime.now() - row['trade_datetime']).total_seconds()))
    
    return np.append(np.array([current_trade_array]), row['trade_history'][:4], axis=0)

def append_last_trade(row):
    ''' 
    This function is only used in production to calculate the current trade array.
    The array is added in the begining the trade history.
    Paramaterts:
    input: dataframe row
    output: numpy array
    '''
    last_trade = []
    last_trade.append(row['last_yield_spread'])
    last_trade.append(row['ficc_ycl'])
    last_trade.append(row['rtrs_control_number'])                 
    last_trade.append(row['yield'] )
    last_trade.append(row['dollar_price'])
    last_trade.append(row['last_seconds_ago'])
    last_trade.append(float(row['par_traded']))
    last_trade.append(row['calc_date'])
    last_trade.append(row['maturity_date'])
    last_trade.append(row['next_call_date'])
    last_trade.append(row['par_call_date'])
    last_trade.append(row['last_refund_date'])
    last_trade.append(row['trade_datetime'])
    last_trade.append(row['calc_day_cat'])
    last_trade.append(row['settlement_date'])
    last_trade.append(row['trade_type'])
    
    return np.append(np.array([last_trade]), row['previous_trades_features'], axis=0)