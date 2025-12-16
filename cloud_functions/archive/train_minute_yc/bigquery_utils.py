import numpy as np
import pandas as pd
from google.cloud import bigquery
from pandas.tseries.offsets import BDay
import datetime


project_id = "eng-reactor-287421"

sp_etf_daily_dataset = 'ETF_daily_alphavantage'
sp_etf_hourly_dataset = 'ETF_hourly_alphavantage'
sp_index_dataset = 'spBondIndex'
sp_maturity_dataset = 'spBondIndexMaturities'

sp_maturity_tables = ['sp_7_12_year_national_amt_free_index',
                      'sp_high_quality_index',
                      'sp_high_quality_intermediate_managed_amt_free_index',
                      'sp_high_quality_short_intermediate_index',
                      'sp_high_quality_short_index'                     
                     ]

sp_index_tables = ['sp_7_12_year_national_amt_free_municipal_bond_index_yield',
                   'sp_muni_high_quality_index_yield',
                   'sp_high_quality_intermediate_managed_amt_free_municipal_bond_index_yield',
                   'sp_high_quality_short_intermediate_municipal_bond_index_yield',
                   'sp_high_quality_short_municipal_bond_index_yield'
                     ]

etfs = ['HYD', 'HYMB', 'IBMJ', 'IBMK', 'IBML', 'IBMM', 'ITM', 'MLN', 'MUB', 'PZA', 'SHM', 'SHYD', 'SMB', 'SUB', 'TFI', 'VTEB']

name_dict = dict(zip(sp_index_tables,sp_maturity_tables))

def last_business_day():
    today = datetime.datetime.now().date()
    lb_datetime = today - BDay(1)
    lb_date=lb_datetime.date()
    return lb_date
    
def sqltodf(sql,limit = ""):
    bq_client = bigquery.Client()
    if limit != "": 
        limit = f"LIMIT {limit}"
    bqr = bq_client.query(sql + limit).result()
    return bqr.to_dataframe()

def load_daily_etf_prices_bq():
    '''
    This function loads the maturity data from the specified bigquery tables in the global etfs list and concatenates them
    into a single dataframe.
    '''
        
    client = bigquery.Client()
    etf_data  = {}
    
    for table in etfs:
        query = '''
                SELECT DISTINCT * FROM {}.{} 

                '''.format(sp_etf_daily_dataset,table)
        
        df = pd.read_gbq(query, project_id=project_id, dialect='standard')
        
        df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')

        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True, drop=True)
        etf_data[table] = df.drop_duplicates()
        
    assert list(etf_data.keys()) == etfs
   
    
    return etf_data

def load_hourly_etf_prices_bq():
    '''
    This function loads the maturity data from the specified bigquery tables in the global etfs list and concatenates them
    into a single dataframe.
    '''
        
    client = bigquery.Client()
    etf_data  = {}
    
    for table in etfs:
        query = '''
                SELECT DISTINCT * FROM {}.{} 

                '''.format(sp_etf_hourly_dataset,table)
        
        df = pd.read_gbq(query, project_id=project_id, dialect='standard')
        
        df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')

        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True, drop=True)
        etf_data[table] = df 
        
    assert list(etf_data.keys()) == etfs
   
    
    return etf_data

def load_index_yields_bq():
    '''
    This function loads the index yield data from the specified bigquery tables in the global sp_index_tables list and concatenates them
    into a single dataframe.
    '''
    
    client = bigquery.Client()
    index_data  = {}
    
    for table in sp_index_tables:
        query = '''
                SELECT DISTINCT * FROM {}.{} 

                '''.format(sp_index_dataset,table)
        
        df = pd.read_gbq(query, project_id=project_id, dialect='standard')
        
        assert list(df.columns) == ['date','ytw']
        
        df = df.drop_duplicates('date')
        df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')
        df.sort_values('date', inplace=True, ascending=True)
        df.set_index('date', inplace=True, drop=True)

        df['ytw'] = df['ytw']/0.01 #convert to basis points
        
        name = name_dict[table] #standardize names between maturity and yield data
        index_data[name] = df 
        
    assert list(index_data.keys()) == sp_maturity_tables
        
    return index_data

def load_maturity_bq():
    '''
    This function loads the maturity data from the specified bigquery tables in the global sp_maturity_tables list and concatenates them
    into a single dataframe.
    '''
    
    client = bigquery.Client()
    maturity_data  = {}
    
    for table in sp_maturity_tables:
        query = '''
                SELECT DISTINCT * FROM {}.{} 

                '''.format(sp_maturity_dataset,table)
        
        df = pd.read_gbq(query, project_id=project_id, dialect='standard')
        
        assert list(df.columns) == ['effectivedate','weightedAverageMaturity','weightedAverageDuration']
        
        df['effectivedate'] = pd.to_datetime(df['effectivedate'], format = '%Y-%m-%d')
        df = df.drop_duplicates('effectivedate')
        df.sort_values('effectivedate', inplace=True)
        df.set_index('effectivedate', inplace=True, drop=True)
        df = df[['weightedAverageMaturity']]
        maturity_data[table] = df 
        
    assert list(maturity_data.keys()) == sp_maturity_tables
    
    maturity_data = pd.concat(maturity_data, axis=1)
    maturity_data.columns = maturity_data.columns.droplevel(-1)
    
    return maturity_data

def get_scalar_lbd():

    """
    This function retrieves the latest standard scalar coefficient from a BigQuery table
    (populated by the Cloud Function train_daily_yield_curve) which will be the standard 
    scalar coefficient for the most recent business date. 

    """

    scalar = sqltodf(f"SELECT * FROM `eng-reactor-287421.yield_curves.standardscaler_parameters_daily` ORDER BY date DESC LIMIT 1")
    return scalar