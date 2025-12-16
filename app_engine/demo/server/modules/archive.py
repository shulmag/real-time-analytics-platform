'''
 # @ Create date: 2024-04-05
 # @ Modified date: 2024-04-05
 '''
import requests
from flask import jsonify, make_response
import pandas as pd

from modules.ficc.utils.auxiliary_functions import sqltodf

from modules.auxiliary_variables import bq_client


def get_access_token(audience):
    '''This function retrieves a Firebase authentication token.'''
    token_response = requests.get(
        'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=' + audience,
        headers={'Metadata-Flavor': 'Google'})     
    return token_response.content.decode('utf-8')


def make_response_if_cusip_found(df, cusip):
    if len(df) == 0:
        return f'No results for {cusip}'
    response = make_response(jsonify(df.to_dict('records')), 200)
    return response


def liquidity_indicators(cusip, api_call):
    '''This function returns some stats about a given CUSIP.'''
    query = f'''
        SELECT 
            (select count(cusip)  from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date = CURRENT_DATE('US/Eastern') and cusip = '{cusip}') as num_trades_today,
            (select round(avg(par_traded),2) from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date = CURRENT_DATE('US/Eastern') and cusip = '{cusip}') as avg_trade_size_today,
            (select count(cusip)  from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date >= DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 1 WEEK) and cusip = '{cusip}') as num_trades_week,
            (select round(avg(par_traded),2) from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date >= DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 1 WEEK) and cusip = '{cusip}') as avg_trade_size_week,
            (select count(cusip)  from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date >= DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 1 MONTH) and cusip = '{cusip}') as num_trades_month,
            (select round(avg(par_traded),2) from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date >= DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 1 MONTH) and cusip = '{cusip}') as avg_trade_size_month,
            (select count(cusip)  from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date >= DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 1 YEAR) and cusip = '{cusip}') as num_trades_year,
            (select round(avg(par_traded),2) from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date >= DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 1 YEAR) and cusip = '{cusip}') as avg_trade_size_year,
            (select round(avg(dollar_price),3)  from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date = CURRENT_DATE('US/Eastern') and cusip = '{cusip}') as avg_price_today,
            (select round(avg(yield),3) from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date = CURRENT_DATE('US/Eastern') and cusip = '{cusip}') as avg_yield_today,
            (select round(avg(dollar_price),3)  from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date >= DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 1 WEEK) and cusip = '{cusip}') as avg_price_week,
            (select round(avg(yield),3) from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date >= DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 1 WEEK) and cusip = '{cusip}') as avg_yield_week,
            (select round(avg(dollar_price),3)  from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date >= DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 1 MONTH) and cusip = '{cusip}') as avg_price_month,
            (select round(avg(yield),3) from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date >= DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 1 MONTH) and cusip = '{cusip}') as avg_yield_month,
            (select round(avg(dollar_price),3)  from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date >= DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 1 YEAR) and cusip = '{cusip}') as avg_price_year,
            (select round(avg(yield),3) from `eng-reactor-287421.MSRB.msrb_trade_messages` where trade_date >= DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 1 YEAR) and cusip = '{cusip}') as avg_yield_year            
        '''

    df = sqltodf(query, bq_client)
    df = df.fillna(0)
    return make_response_if_cusip_found(df, cusip)


def get_last_trade_date(cusip, api_call):
    query = f'''
        SELECT trade_date, dollar_price, yield, trade_type, par_traded FROM `eng-reactor-287421.MSRB.msrb_trade_messages`
        WHERE cusip = '{cusip}'
        ORDER BY trade_date DESC, time_of_trade DESC
        LIMIT
        1
    '''
    df = sqltodf(query, bq_client)
    if pd.isnull(df.iloc[0]['yield']):
        df['yield'] = 'Not Found'
    return make_response_if_cusip_found(df, cusip)


def convert_calc_date_to_cat(row):
    if row.yield_spread_calc_date == row.next_call_date: 
        return 0
    elif row.yield_spread_calc_date == row.par_call_date:
        return 1
    elif row.yield_spread_calc_date == row.maturity_date: 
        return 2
    elif row.yield_spread_calc_date == row.refund_date: 
        return 3
    else: 
        return 4
