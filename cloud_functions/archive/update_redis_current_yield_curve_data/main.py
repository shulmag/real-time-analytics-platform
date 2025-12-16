# -*- coding: utf-8 -*-
# @Date:   2021-09-14 12:08:35
from google.cloud import bigquery
import pandas as pd
import pickle
import redis

#Specify dataset names
project_id = "eng-reactor-287421"
sp_etf_daily_dataset = 'ETF_daily_alphavantage'
sp_etf_hourly_dataset = 'ETF_hourly_alphavantage'
sp_index_dataset = 'spBondIndex'
sp_maturity_dataset = 'spBondIndexMaturities'

#Define name of bigquery tables containing maturity data
sp_maturity_tables = ['sp_7_12_year_national_amt_free_index',
                      'sp_high_quality_index',
                      'sp_high_quality_intermediate_managed_amt_free_index',
                      'sp_high_quality_short_intermediate_index',
                      'sp_high_quality_short_index'                     
                     ]


#Define name of bigquery tables containing S&P index data
sp_index_tables = ['sp_7_12_year_national_amt_free_municipal_bond_index_yield',
                   'sp_muni_high_quality_index_yield',
                   'sp_high_quality_intermediate_managed_amt_free_municipal_bond_index_yield',
                   'sp_high_quality_short_intermediate_municipal_bond_index_yield',
                   'sp_high_quality_short_municipal_bond_index_yield'
                     ]


etfs = ['HYD','HYMB', 'IBMJ', 'IBMK', 'IBML', 'IBMM', 'ITM', 'MLN', 'MUB', 'PZA', 'SHM', 'SHYD', 'SMB', 'SUB', 'TFI', 'VTEB']

name_dict = dict(zip(sp_index_tables,sp_maturity_tables))

def load_daily_etf_prices_bq():
    '''
    This function loads the maturity data from the specified bigquery tables in the global etfs list and concatenates them
    into a single dataframe.
    '''
        
    client = bigquery.Client()
    etf_data  = {}
    
    for table in etfs:
        query = '''
                SELECT * FROM {}.{} 

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
                SELECT * FROM {}.{} 

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
                SELECT * FROM {}.{} 

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
                SELECT * FROM {}.{} 

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

def load_scaler_daily_bq():
    '''
    This function loads the scaler parameters used in the sklearn StandardScaler to scale the input data for the daily nelson-siegel model
    during training.
    '''
    bq_client = bigquery.Client()
    query = '''
            SELECT * FROM yield_curves.standardscaler_parameters_daily

            '''
    query_job = bq_client.query(query)
    df = query_job.result().to_dataframe()
    df['date'] = pd.to_datetime(df['date'])    
    df = df.sort_values(by='date',ascending=True).set_index('date', drop=True)
    return df


def main(args):
  redis_data = {'daily_etf_data':load_daily_etf_prices_bq(),
              'hourly_etf_data': load_hourly_etf_prices_bq(),
              'sp_index_data': load_index_yields_bq(),
              'sp_index_maturity_data': load_maturity_bq(),
              'scalar_coefficient': load_scaler_daily_bq()}
    
  redis_client = redis.Redis(host='10.146.62.92', port=6379, db=0)
  redis_client.set('current_yield_data',pickle.dumps(redis_data,protocol=pickle.HIGHEST_PROTOCOL))
  return "Success!"

