'''
'''
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from auxiliary_functions import KEY_MATURITIES, date_format


### INTRA-DAY FUNCTIONS
def exponential_mean(data, weights):
    """Calculate exponentially weighted mean of a time series"""
    if not data or not weights:
        return None
    return np.average(data, weights=weights[:len(data)])


def intraday_groupby(df):
    """Group dataframe by date, ensuring proper date formatting"""
    if df is None or df.empty:
        return None
    
    try:
        df['date'] = pd.to_datetime(df.index.date)
        df.sort_index(ascending=True, inplace=True)
        return df.groupby('date')
    except Exception as e:
        print(f"Error in intraday_groupby: {str(e)}")
        return None


def _apply_intraday_func(df, func):
    """Apply a function to each day's data for each maturity"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    try:
        df['date'] = df.index.date
        summary = pd.DataFrame(index=df['date'].unique())
        
        for maturity in KEY_MATURITIES:
            if maturity in df.columns:
                # Round values to 2 decimal places for consistency
                group = intraday_groupby(df)
                if group is not None:
                    result = group[maturity].apply(func)
                    # Round to 2 decimal places
                    if result is not None:
                        result = result.round(2)
                    summary = summary.join(result)
        
        summary.index = pd.to_datetime(summary.index)
        return summary
    except Exception as e:
        print(f"Error in _apply_intraday_func: {str(e)}")
        return pd.DataFrame()


def intraday_largest_move(df):
    """Find the largest move from market open for each day"""
    # lambda function will be applied after groupby 'date'; finds the largest absolute difference between first entry
    # (market open) and any other datapoint within the same day 
    func = lambda x: max([x.max() - x.iloc[0], x.min() - x.iloc[0]], key=abs)
    result = _apply_intraday_func(df, func)
    return result


def intraday_open_close(df):
    """Calculate the change from market open to market close for each day"""
    # lambda function will be applied after groupby 'date'; finds the difference between the first (market open)
    # and last (market close) entry
    func = lambda x: x.iloc[-1] - x.iloc[0]
    result = _apply_intraday_func(df, func)
    return result


def intraday_mean(df):
    """Calculate the mean value for each day"""
    result = _apply_intraday_func(df, np.mean)
    return result


def intraday_SD(df):
    """Calculate the standard deviation for each day"""
    result = _apply_intraday_func(df, np.std)
    return result


### INTER-DAY FUNCTIONS

def get_close_open(group, one_business_day_before_today, today):
    """Get the closing value for the previous day and opening value for today"""
    if group is None or not one_business_day_before_today or not today:
        return None
        
    try:
        date_str = group.name.strftime(date_format)
        if date_str == one_business_day_before_today:
            # Round to 2 decimal places
            close_vals = group.iloc[[-1]].round(2)
            return close_vals  # Close value for the previous business day
        elif date_str == today:
            # Round to 2 decimal places
            open_vals = group.iloc[[0]].round(2)
            return open_vals  # Open value for today
        else:
            return None
    except Exception as e:
        print(f"Error in get_close_open: {str(e)}")
        return None


def validate_dates(one_business_day_before_today, today):
    """Validate input dates, providing defaults if necessary"""
    if not one_business_day_before_today or not today:
        # Default to yesterday and today
        today_date = datetime.now().strftime(date_format)
        yesterday_date = (datetime.now() - timedelta(days=1)).strftime(date_format)
        return yesterday_date, today_date
    return one_business_day_before_today, today


def get_overnight_change(yield_curve_df, one_business_day_before_today=None, today=None):
    """Calculate the change from previous day's close to today's open"""
    if yield_curve_df is None or yield_curve_df.empty:
        print("Error: Empty or None yield curve dataframe provided")
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        # Validate dates
        one_business_day_before_today, today = validate_dates(one_business_day_before_today, today)
        
        df = yield_curve_df.copy()
        df['date'] = yield_curve_df.index.date
        df.sort_index(ascending=True, inplace=True)

        # Take market open for today and close for the previous business day
        df = (
            df.reset_index()
            .groupby('date')
            .apply(lambda group: get_close_open(group, one_business_day_before_today, today))
            .dropna()  # Remove None/Null values from get_close_open
            .reset_index(drop=True)
        )
        
        # Handle edge case where we don't have data for both days
        if df.shape[0] < 2:
            print(f"Warning: Insufficient data for calculating overnight change. Dates requested: {one_business_day_before_today}, {today}")
            return pd.DataFrame(), pd.DataFrame()
            
        # Set index properly
        if 'index' in df.columns:
            df = df.set_index('index', drop=True)
        df = df.drop('date', axis=1, errors='ignore')
        df = df.rename_axis(None, axis=0)
        
        # Calculate overnight deltas and round to 2 decimal places
        overnight_deltas = df[KEY_MATURITIES].copy().round(2)
        overnight_deltas.loc['Overnight Delta', :] = df[KEY_MATURITIES].diff().iloc[-1].round(2)
        
        return df.round(2), overnight_deltas
    except Exception as e:
        print(f"Error in get_overnight_change: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()