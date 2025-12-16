# data_fetchers.py

from typing import Dict, List
import pandas as pd
from google.cloud import bigquery
from datetime import datetime, timedelta
import requests
import time

# Old ETFs
# MUNI_ETFS = [
#     'HYD', 'HYMB', 'IBMM', 'ITM', 'MLN', 'MUB', 'PZA', 'SHM', 
#     'SHYD', 'SMB', 'SUB', 'TFI', 'VTEB', 'FMHI', 'MMIN', 'SMMU', 
#     'NEAR', 'MEAR'
# ]

# All: 
MUNI_ETFS = [
   'AVMU',     'BAB',      'BSMV',     'CGMU',     'CGSM',     'CMF',      
   'DFNM',     'FLMB',     'FLMI',     'FMB',      'FMHI',     'FMNY',     
   'FSMB',     'FUMB',     'GMUN',     'HMOP',     'HYD',      'HYMB',     
   'HYMU',     'IBMM',     'IMSI',     'INMU',     'ITM',      'JMST',     
   'JMUB',     'MBND',     'MBNE',     'MEAR',     'MFLX',     'MINN',     
   'MLN',      'MMIN',     'MMIT',     'MUB',      'MUNI',     'MUST',     
   'NEAR',     'NYF',      'OVM',      'PVI',      'PWZ',      'PZA',      
   'RTAI',     'RVNU',     'SHM',      'SHYD',     'SMB',      'SMI',      
   'SMMU',     'SUB',      'TAFI',     'TAFL',     'TAFM',     'TAXF',     
   'TFI',      'VTEB',     'VTES'
]

SP_INDEX_TABLES = [
    'sp_12_22_year_national_amt_free_index',
    'sp_15plus_year_national_amt_free_index',
    'sp_7_12_year_national_amt_free_municipal_bond_index_yield',
    'sp_muni_high_quality_index_yield',
    'sp_high_quality_intermediate_managed_amt_free_municipal_bond_index_yield',
    'sp_high_quality_short_intermediate_municipal_bond_index_yield',
    'sp_high_quality_short_municipal_bond_index_yield',
    'sp_long_term_national_amt_free_municipal_bond_index_yield'
]

def get_etf_data(symbol, start_date, end_date, resolution, token):
    """
    Fetches historical data for a given ETF symbol from Finnhub API.
    """
    try:
        # Convert dates to UNIX timestamps
        start_unix = int(time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
        end_unix = int(time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple()))
        
        url = f'https://finnhub.io/api/v1/stock/candle'
        params = {
            'symbol': symbol,
            'resolution': resolution,
            'from': start_unix,
            'to': end_unix,
            'token': token
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check for error in response
        if 'error' in data:
            print(f"API Error for {symbol}: {data['error']}")
            return None
            
        # Check for required data fields
        required_keys = ['t', 'o', 'h', 'l', 'c', 'v']
        if not all(key in data for key in required_keys):
            print(f"Missing data fields for {symbol}")
            return None
            
        # Create DataFrame
        df = pd.DataFrame({
            'date': pd.to_datetime(data['t'], unit='s'),
            'symbol': symbol,
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        })
        
        return df
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None

def fetch_all_etf_data(api_key: str) -> Dict[str, pd.DataFrame]:
    """
    Fetches historical data for all ETFs in the predefined list.
    
    Parameters:
    - finnhub_api_key (str): Your Finnhub API key
    
    Returns:
    - dict: Dictionary of DataFrames containing historical data for each ETF
    """
    # Calculate dates
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # List of ETFs
    etf_list = MUNI_ETFS
    
    # Dictionary to store dataframes
    etf_data = {}
    
    # Fetch data for each ETF
    for etf in etf_list:
        print(f"Fetching data for {etf}")
        df = get_etf_data(
            etf, 
            start_date, 
            end_date, 
            resolution='D', 
            token=api_key
        )
        
        if df is not None:
            etf_data[etf] = df
            
        time.sleep(1)  # Rate limiting
    
    return etf_data

def fetch_sp_index_data() -> pd.DataFrame:
    """
    Fetches last 12 months of data from all S&P index tables in BigQuery.
    Handles potential duplicate values per day by keeping the latest value.
    Returns a DataFrame with date and ytw values for each index.
    """
    client = bigquery.Client()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    tables = SP_INDEX_TABLES
    
    dfs = {}
    
    for table in tables:
        # First, check for duplicates
        check_query = f"""
        SELECT date, COUNT(*) as count
        FROM `eng-reactor-287421.spBondIndex.{table}`
        WHERE date BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
        GROUP BY date
        HAVING COUNT(*) > 1
        ORDER BY date
        """
        
        duplicates_df = client.query(check_query).to_dataframe()
        if not duplicates_df.empty:
            print(f"\nFound duplicates in {table}:")
            # print(duplicates_df)
        
        # Fetch data with deduplication in the query
        query = f"""
        WITH RankedData AS (
            SELECT 
                date,
                ytw,
                ROW_NUMBER() OVER(PARTITION BY date ORDER BY date DESC) as rn
            FROM `eng-reactor-287421.spBondIndex.{table}`
            WHERE date BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
        )
        SELECT date, ytw
        FROM RankedData
        WHERE rn = 1
        ORDER BY date
        """
        
        print(f"Fetching data for {table}")
        df = client.query(query).to_dataframe()
        
        # Add table name as identifier
        dfs[table] = df
        print(f"Fetched {len(df)} rows")
    
    # Merge all DataFrames on date
    final_df = None
    for table, df in dfs.items():
        df = df.rename(columns={'ytw': f'ytw_{table}'})
        
        if final_df is None:
            final_df = df
        else:
            final_df = final_df.merge(df, on='date', how='outer')
    
    # Sort by date and handle any missing values
    final_df = final_df.sort_values('date')
    
    return final_df