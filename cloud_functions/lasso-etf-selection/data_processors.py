# data_processors.py

import pandas as pd
import numpy as np
from typing import Dict, Tuple

def calculate_etf_returns(etf_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate daily returns for each ETF and combine into a single DataFrame.
    
    Parameters:
    -----------
    etf_data : Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing ETF historical data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with daily returns for each ETF
    """
    etf_returns = pd.DataFrame()
    
    for etf, df in etf_data.items():
        # Ensure date is index and properly formatted
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        
        # Compute daily percentage returns
        etf_returns[etf] = df['close'].pct_change().dropna()
    
    return etf_returns

def calculate_yield_changes(sp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily changes in yield for each S&P index.
    
    Parameters:
    -----------
    sp_data : pd.DataFrame
        DataFrame containing S&P index YTW data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with daily yield changes for each index
    """
    # Ensure date is index and properly formatted
    sp_data = sp_data.set_index('date')
    sp_data.index = pd.to_datetime(sp_data.index)
    
    # Calculate daily changes
    yield_changes = sp_data.diff().dropna()
    
    return yield_changes

def combine_data(etf_returns: pd.DataFrame, yield_changes: pd.DataFrame) -> pd.DataFrame:
    """
    Combine ETF returns and yield changes into a single DataFrame.
    
    Parameters:
    -----------
    etf_returns : pd.DataFrame
        DataFrame containing ETF daily returns
    yield_changes : pd.DataFrame
        DataFrame containing S&P index yield changes
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with both ETF returns and yield changes
    """
    # Inner join to ensure we only have dates where both are available
    combined_data = etf_returns.join(yield_changes, how='inner').dropna()
    return combined_data