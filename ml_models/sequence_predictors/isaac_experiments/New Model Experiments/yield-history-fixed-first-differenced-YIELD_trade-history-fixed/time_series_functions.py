import pandas as pd
from datetime import datetime 
from pandas.tseries.offsets import BDay

### INTRADAY FUNCTIONS
def intraday_groupby(df): 
    df['date'] = df.index.date
    df.sort_index(ascending=True, inplace=True)
    return df.groupby('date')

def _apply_intraday_func(df, func):
    df['date'] = df.index.date
    
    summary = pd.DataFrame(index = df['date'].unique())
    for maturity in key_maturities:
        summary = summary.join(intraday_groupby(df)[maturity].apply(func))
    
    return summary

def intraday_largest_move(df):
     #Lambda function will be applied after groupby 'date'; finds the largest absolute difference between first entry
    #(market open) and any other datapoint within the same day 
    func = lambda x: np.max([np.abs(x.iloc[0] - x.max()), 
                        np.abs(x.iloc[0] - x.min())]
                       )
    
    return _apply_intraday_func(df, func)

def intraday_open_close(df):
     #Lambda function will be applied after groupby 'date'; finds the difference between the first (market_ pen)
     #and last (market close) entry
        
    func = lambda x: np.abs(x.iloc[0]-x.iloc[-1])
    
    return _apply_intraday_func(df, func)

def intraday_mean(df):
    return _apply_intraday_func(df, np.mean)

def intraday_SD(df):
    return _apply_intraday_func(df, np.std)


### BETWEEN-DAY FUNCTIONS
def filter_consecutive_biz_days(df):
    df['flag'] = False
    for i in range(len(df)-1):
        if df.iloc[i, :].name.date() != (df.iloc[i+1, :].name.date() - BDay(1)):
            idx = df.iloc[i+1].name
            df.loc[idx,'flag']=True
    return df

def filter_open_close_entries(df): 
    df['date'] = df.index.date
    df.sort_index(ascending=True, inplace=True)
    
    #Take market open and close for each date 
    df = df.reset_index().groupby('date').apply(lambda x: x.iloc[[0,-1]])\
    .droplevel(1).drop('date', axis=1)\
    .rename({'index':'datetime'}, axis=1)
    
    #Some days, like 2023-02-07, don't have first and last entries at the correct times 
    weird_dates = df[(df['datetime'].dt.hour!=15) & (df['datetime'].dt.hour!=9)].index.tolist()
    df = df.drop(weird_dates)
    
    assert len(df[(df['datetime'].dt.hour!=15) & (df['datetime'].dt.hour!=9)].index.tolist()) == 0
    
    df.set_index('datetime', inplace=True)
    
    return df

def between_day_overnight_move(df, absolute = False):
    if absolute:
        df = filter_consecutive_biz_days(filter_open_close_entries(ycl).diff().apply(np.abs))
    else:
        df = filter_consecutive_biz_days(filter_open_close_entries(ycl).diff())
        
    return df[~df.flag].drop('flag', axis = 1)

def between_day_close_close_move(df):
    
    df = filter_consecutive_biz_days(filter_open_close_entries(ycl)[1::2])
        
    return df[~df.flag].drop('flag', axis = 1).diff()
