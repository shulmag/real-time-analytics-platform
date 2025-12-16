# -*- coding: utf-8 -*-
# @Date:   2021-09-03 14:47:45

import pandas as pd
import numpy as np
import ficc_globals as globals
from pandarallel import pandarallel
from datetime import datetime
from yield_value import yield_curve_level
from ficc_calc_end_date import calc_end_date
pandarallel.initialize()


COUPON_FREQUENCY_DICT = {0:"Unknown",
                        1:"Semiannually",
                        2:"Monthly",
                        3:"Annually",
                        4:"Weekly",
                        5:"Quarterly",
                        6:"Every 2 years",
                        7:"Every 3 years",
                        8:"Every 4 years",
                        9:"Every 5 years",
                        10:"Every 7 years",
                        11:"Every 8 years",
                        12:"Biweekly",
                        13:"Changeable",
                        14:"Daily",
                        15:"Term mode",
                        16:"Interest at maturity",
                        17:"Bimonthly",
                        18:"Every 13 weeks",
                        19:"Irregular",
                        20:"Every 28 days",
                        21:"Every 35 days",
                        22:"Every 26 weeks",
                        23:"Not Applicable",
                        24:"Tied to prime",
                        25:"One time",
                        26:"Every 10 years",
                        27:"Frequency to be determined",
                        28:"Mandatory put",
                        29:"Every 52 weeks",
                        30:"When interest adjusts-commercial paper",
                        31:"Zero coupon",
                        32:"Certain years only",
                        33:"Under certain circumstances",
                        34:"Every 15 years",
                        35:"Custom",
                        36:"Single Interest Payment"
                        }


IDENTIFIERS = ['rtrs_control_number', 'cusip']


BINARY = ['callable',
          'sinking',
          'zerocoupon',
          'is_non_transaction_based_compensation',
          'is_general_obligation',
          'callable_at_cav',           
          'extraordinary_make_whole_call', 
          'make_whole_call',
          'has_unexpired_lines_of_credit',
          'escrow_exists',
          ]



CATEGORICAL_FEATURES = ['rating',
                        'incorporated_state_code',
                        'trade_type',
                        'transaction_type',
                        'maturity_description_code',
                        'purpose_class']

NON_CAT_FEATURES = ['quantity',
                    'days_to_maturity',
                    'days_to_call',
                    'coupon',
                    'issue_amount',
                    'last_seconds_ago',
                    'last_yield_spread',
                    'days_to_settle',
                    'days_to_par',
                    'maturity_amount',
                    'issue_price', 
                    'orig_principal_amount',
                    'max_amount_outstanding']

TRADE_HISTORY_input = ['trade_history_input']

PREDICTORS = BINARY+ NON_CAT_FEATURES  + CATEGORICAL_FEATURES + TRADE_HISTORY_input

'''
This function takes in an SQL query and returns a dataframe
with the data fetched from big query
'''
def sqltodf(sql,bq_client):
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()


'''
The pad_trade_history_input function pads the trade histories if needed to make their length equal to the sequence length.
The function pads the beginning of trade history and creates a single sequence.

If the length of the trade history is equal to the sequence length the function returns the list as is. 
As an initial step, we are only padding trades that have at least 16 trades in the sequence. 
We will expand the model to include comps for CUSIPs which do not have sufficient history
'''
def pad_trade_history_input(x, SEQUENCE_LENGTH, NUM_FEATURES):
    '''
    Pads the sequence of historical trades
    x : list
    
    '''
    # Need to pad with similiar embeddings / comps / KNN: 
    if len(x) < SEQUENCE_LENGTH:  #and len(x) > SEQUENCE_LENGTH//2: 
        temp = x.tolist()
        temp = temp + [[0]*NUM_FEATURES]*(SEQUENCE_LENGTH - len(x))
        return np.stack(temp)
    
    #returning none for data less than sequence length
    elif len(x) < SEQUENCE_LENGTH:
        return None

    else:
        return x

'''
The trade_dict_to_list converts the recent trade dictionary to a list.
The SQL arrays from BigQuery are converted to a dictionary when read as a pandas dataframe. 

A few blunt normalization are performed on the data. We will experiment with others as well. 
Multiplying the yield spreads by 100 to convert into basis points. 
Taking the log of the size of the trade to reduce the absolute scale 
Taking the log of the number of seconds between the historical trade and the latest trade
'''

def trade_dict_to_list(trade_dict: dict, calc_date) -> list:
    trade_type_mapping = {'D':[0,0],'S': [0,1],'P': [1,0]}
    trade_list = []

    if trade_dict['trade_datetime'] < datetime(2021,7,27):
        target_date = datetime(2021,7,27).date()
    else:
        target_date = trade_dict['trade_datetime'].date()
    
    #calculating the time to maturity in years from the trade_date
    time_to_maturity = (calc_date - target_date).days/365.25
    yield_at_that_time = yield_curve_level(time_to_maturity,
                                           target_date.strftime('%Y-%m-%d'),
                                           globals.nelson_params,
                                           globals.scalar_params)

    #trade_list.append(trade_dict['yield_spread'] * 100)
    trade_list.append(trade_dict['yield'] * 100 - yield_at_that_time)
    trade_list.append(np.float32(np.log10(trade_dict['par_traded'])))        
    trade_list += trade_type_mapping[trade_dict['trade_type']]
    #trade_list.append(trade_dict['dollar_price'])
    #trade_list.append(time_to_maturity * 365.25)

    if trade_dict['seconds_ago'] < 0:
        trade_list.append(0)
    else:
        trade_list.append(np.log10(1+trade_dict['seconds_ago']))

    return np.stack(trade_list)
'''
The trade_list_to_array function uses the trade_dict_to_list function to unpack the list of dictionaries
and creates a list of historical trades. 
With each element in the list containing all the information for that particular trade
'''
def trade_list_to_array(trade_history):
    if len(trade_history) == 0:
        return np.array([])

    calc_date = trade_history[-1]
    trade_history = trade_history[:-1] 
    # calc_date = None
    trades_list = []

    for entry in trade_history:
        trades = trade_dict_to_list(entry,calc_date)
        if trades is not None:
            trades_list.append(trades)

    if len(trades_list) > 0:
        return np.stack(trades_list)
    else:
        return []

def pocess_trade_history_input(df,SEQUENCE_LENGTH, NUM_FEATURES):
    print("Dropping empty trades")
    df['empty_trade'] = df.recent.apply(lambda x: x[0]['rtrs_control_number'] is None)
    df = df[df.empty_trade == False]
    
    print("Dropping trades less that 10000$")
    df = df[df.par_traded > 10000]
    #Taking only the most recent trades
    df.recent = df.recent.apply(lambda x: x[:SEQUENCE_LENGTH])

    if len(df) == 0: 
        return 'no trade history'
    
    print('Estimating calculation date')
    df['calc_date'] = df.parallel_apply(calc_end_date, axis=1)
    df.recent =  df.apply(lambda x: np.append(x['recent'],np.array(x['calc_date'])),axis=1 )
    
    print('Creating trade history')
    df['trade_history_input'] = df.recent.parallel_apply(trade_list_to_array)
    df.drop(columns=['recent', 'empty_trade'],inplace=True)
    print('Trade history created')

    df.trade_history_input = df.trade_history_input.apply(pad_trade_history_input, args=[SEQUENCE_LENGTH, NUM_FEATURES])

    if df.trade_history_input.isnull().values[0] == True:
        return 'no trade history'
        
    return df

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
                 'security_description',
                 'ref_valid_from_date',
                 'ref_valid_to_date',
                 'additional_next_sink_date',
                 'first_coupon_date',
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

def convert_numerical(df):
    global NON_CAT_FEATURES
    for feature in NON_CAT_FEATURES:
        df[feature] = df[feature].astype(float)
    return df

def add_rating(df):
    df = df[df.sp_long.isin(['A-','A','A+','AA-','AA','AA+','AAA','NR'])] 
    df['rating'] = df.sp_long
    # TODO: test later
    # df['rating'].fillna("NR",inplace=True) 
    return df

def fill_missing_values(df):
    df.dropna(subset=['instrument_primary_name'], inplace=True)
    df.purpose_sub_class.fillna(1,inplace=True)
    df.call_timing.fillna(0, inplace=True) #0 = Unknown
    df.call_timing_in_part.fillna(0, inplace=True) #0 = Unknown
    df.sink_frequency.fillna(10, inplace=True) #10 = Under special circumstances
    df.sink_amount_type.fillna(0, inplace=True) #0 = Unknown
    df.issue_text.fillna('No issue text', inplace=True)
    df.state_tax_status.fillna(0, inplace=True)
    df.interest_payment_frequency.fillna(0, inplace=True)
    df.series_name.fillna('No series name', inplace=True)

    df.next_call_price.fillna(100, inplace=True)
    df.par_call_price.fillna(100, inplace=True)
    df.min_amount_outstanding.fillna(0, inplace=True)
    df.max_amount_outstanding.fillna(0, inplace=True)
    df.days_to_par.fillna(0, inplace=True)
    df.maturity_amount.fillna(0, inplace=True)
    df.issue_price.fillna(df.issue_price.mean(), inplace=True)
    df.orig_principal_amount.fillna(df.orig_principal_amount.mean(), inplace=True)
    df.original_yield.fillna(0, inplace=True)
    df.par_price.fillna(100, inplace=True)

    df.extraordinary_make_whole_call.fillna(False, inplace=True)
    df.make_whole_call.fillna(False, inplace=True)
    df.default_indicator.fillna(False, inplace=True)
    df.called_redemption_type.fillna(0, inplace=True)
    
    return df

def get_latest_trade_feature(x, feature):
    recent_trade = x[0]
    if feature == 'yield_spread':
        return recent_trade[0]
    elif feature == 'seconds_ago':
        return recent_trade[-1]
    elif feature == 'par_traded':
        return recent_trade[1]

def settlement_pace(x):
    if x <= 3:
        return 'Fast'
    elif x>3 and x <=15:
        return 'Medium'
    else:
        return 'Slow'

def create_feature(df):
    global COUPON_FREQUENCY_DICT
    df.interest_payment_frequency.fillna(0, inplace=True)
    df.interest_payment_frequency = df.interest_payment_frequency.apply(lambda x: COUPON_FREQUENCY_DICT[x])
    
    df['quantity'] = np.log10(df.quantity.astype(float))
    df.coupon = df.coupon.astype(float)
    df.issue_amount = np.log10(df.issue_amount.astype(float))
    df['yield_spread'] = df['yield_spread'] * 100
    
    df['callable'] = df.is_callable  
    df['called'] = df.is_called 
    df['zerocoupon'] = df.coupon == 0
    df['whenissued'] = df.delivery_date >= df.trade_date
    df['sinking'] = ~df.next_sink_date.isnull()
    df['deferred'] = (df.interest_payment_frequency == 'Unknown') | df.zerocoupon
    
    df['days_to_settle'] = (df.settlement_date - df.trade_date).dt.days.fillna(0)
    df = df[df.days_to_settle <= 31]
    df['settle_pace'] = df.days_to_settle.apply(settlement_pace)

    df['days_to_maturity'] =  np.log10(1 + (df.maturity_date - df.settlement_date).dt.days.fillna(0))
    df['days_to_call'] = np.log10(1 + (df.next_call_date - df.settlement_date).dt.days.fillna(0))
    df['days_to_refund'] = np.log10(1 + (df.refund_date - df.settlement_date).dt.days.fillna(0))
    df['days_to_par'] = np.log10(1 + (df.par_call_date - df.settlement_date).dt.days.fillna(0))
    df['call_to_maturity'] = np.log10(1 + (df.maturity_date - df.next_call_date).dt.days.fillna(0))
    df = df[df.incorporated_state_code != 'PR']
    
    df['last_seconds_ago'] = df.trade_history_input.apply(get_latest_trade_feature, args=["seconds_ago"])
    df['last_yield_spread'] = df.trade_history_input.apply(get_latest_trade_feature, args=["yield_spread"])
    df['last_size'] = df.trade_history_input.apply(get_latest_trade_feature, args=["par_traded"])


    df.maturity_amount = np.log10(1.0 + df.maturity_amount.astype(float))
    df.orig_principal_amount = np.log10(1.0 + df.orig_principal_amount.astype(float))
    df.max_amount_outstanding = np.log10(1.0 + df.max_amount_outstanding.astype(float))
    
    return df

def get_ficc_ycl(trade):
    duration = (trade.calc_date - trade.settlement_date).days/365.25
    ficc_yl = yield_curve_level(duration,
                                trade.trade_date.strftime('%Y-%m-%d'),
                                globals.nelson_params, 
                                globals.scalar_params)
    return ficc_yl

def process_data(df,SEQUENCE_LENGTH,NUM_FEATURES):
    # df = drop_extra_columns(df)
    df = pocess_trade_history_input(df,SEQUENCE_LENGTH,NUM_FEATURES)
    
    if type(df) == str:
        return 'no trade history'
    
    df = convert_dates(df)
    df = add_rating(df)

    df = create_feature(df)
    df = fill_missing_values(df)
    
    df = convert_numerical(df)

    print("Calculating Yield Spreads")
    df['ficc_ycl'] = df.parallel_apply(get_ficc_ycl, axis=1)
    df['yield_spread'] = df['yield'] * 100 - df['ficc_ycl']

    return df
