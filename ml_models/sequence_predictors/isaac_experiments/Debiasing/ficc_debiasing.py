import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_error

### Descriptive Statistics and Filtering: 
def groupby_check(df, groupby_cols, small_group_threshold = 50):
    '''Checks dataframe for groupby groups with size less than small_group_threshold

    Small_group_threshold = 50 is the default since, generally, CLT requires N > 50 to apply
    '''

    for col in groupby_cols:
        if col not in df.columns: 
            raise KeyError(f'Col {col} not in data frame')
    temp = train_dataframe_rmval.groupby(groupby_cols).size()
    n_groups = len(temp)
    zero_groups = temp[temp == 0]
    small_groups = temp[temp < 50]
    print(f'Groupby cols: {groupby_cols}\nTotal Groups = {n_groups}\nGroups with zero trades: {len(zero_groups)}({100*len(zero_groups)/n_groups:.2f}%)\nGroups with <{small_group_threshold} groups: {100*len(small_groups)}({100*len(small_groups)/n_groups:.2f}%)')

    if len(zero_groups) <= 30:
        display(zero_groups)
    if len(small_groups) <= 30:
        display(small_groups)  

def summarize_col(data, name):
    '''Print descriptive statistics of dataframe column'''
    print(f'{name:20}: {np.mean(data):.2f}, SD: {data.std():.2f}, Max: {data.max():.2f}, Min: {data.min():.2f}, 75%: {data.quantile(.75):.2f}, 25%: {data.quantile(.25):.2f}')

def create_hour_col(df):
    '''Create hour column that is equal to 0 if before market, 25 if after market and the actual hour if during market '''

    from datetime import timedelta
    df['hour'] = df.trade_datetime.dt.hour
    bef_market = df.trade_datetime < df.trade_datetime.dt.normalize() + timedelta(hours=9, minutes=30)
    aft_market = df.trade_datetime > df.trade_datetime.dt.normalize() + timedelta(hours=4, minutes=0)
    df.loc[bef_market,'hour'] = 0
    df.loc[aft_market,'hour'] = 25    

def filter_column_outliers(df, col_name, upper, lower):
    '''Filter large errors from dataframe based on quantiles'''

    return df[(df[col_name] < df[col_name].quantile(upper)) & (df[col_name] > df[col_name].quantile(lower))]    

def debias_constant(pred, truth, bias_correction):
    corrected_pred = pred - bias_correction
    print(f'Original bias: {np.mean(pred-truth):.2f}, Original MAE: {mean_absolute_error(pred, truth):.2f}, Corrected bias: {np.mean(corrected_pred-truth):.2f}, Corrected MAE: {mean_absolute_error(corrected_pred, truth):.2f}')

# ## Debiasing Functions:

####: For Experimentation
def debias_series(pred, truth, bias_correction):
    '''Subtract bias correction from prediction and calculates MAE relative to true values
    
    The bias_correction values must be of the same length as pred and truth. Each row should correspond to the bias correction for the corresponding prediction.
    '''
    
    pred = np.array(pred).flatten()
    truth = np.array(truth).flatten()
    bias_correction = np.array(bias_correction).flatten()
    if len(bias_correction) != len(pred) != len(truth): raise ValueError('Pred, truth, bias_correction must be same shape')
    
    corrected_pred = pred - bias_correction
    print(f'Original bias: {np.mean(pred-truth):.3f}, Original MAE: {mean_absolute_error(pred, truth):.3f}, Corrected bias: {np.mean(corrected_pred-truth):.3f}, Corrected MAE: {mean_absolute_error(corrected_pred, truth):.3f}')

def debias_subgroups(df, groupby_cols, prediction_col ='predicted', target_col = 'new_ys', error_col = 'error', biases = None):
    '''Performs debiasing within subgroups based on groupby column
    
    Subgroup biases can either be calculated beforehand and passed as a dictionary or they will default to the average bias within the group. 
    A summary is printed after debiasing.
    Note that this is experimental and cannot be used in production since we are calculating biases using the entire dataframe, not on a rolling basis. In production we do not have 
    visibility of all biases at once at aggregate. 
    '''
    
    if not biases: biases = df.groupby(groupby_cols)[error_col].mean().fillna(0).to_dict()
    corrected_pred = df.groupby(groupby_cols)[prediction_col].apply(lambda x: x - biases[x.name])
    
    pred = df[prediction_col]
    truth = df[target_col]
    
    print(f'Original total bias: {np.mean(pred-truth):.2f}, Original MAE: {mean_absolute_error(pred, truth):.2f}')
    print(f'MAE under bias correction using overall bias: {mean_absolute_error(pred-np.mean(pred-truth), truth):.2f}')
    print(f'MAE under bias correction using subgroup biases: {mean_absolute_error(corrected_pred, truth):.2f}')
    print(f'Largest Bias: {max(biases.values()):.2f}, Group: {max(zip(biases.values(), biases.keys()))[1]}')
    print(f'Smallest Bias: {min(biases.values()):.2f}, Group: {min(zip(biases.values(), biases.keys()))[1]}')
    print(f'Average Subgroup Bias: {np.mean(list(biases.values())):.2f}')
    print()


#### Weighted Average Functions: 
def calculate_weighted_average(data, weighting_col, error_col, method = 'default', mask_large = 35):
    '''Calculates weighted average of error_col based on weighting_col
    
    If weighted average is to be calculated based on error magnitude, weighting_col should be set to error_col. 
    Different ways of calculating the weighted average are dictated by the method kwarg.
    '''
    
    data = data.iloc[:-1]
    if len(data) == 0: return 0 

    errors = data[error_col].to_numpy()
    weights = data[weighting_col].to_numpy()

    if method == 'simple_average':
        weights = np.ones(len(errors))
    
    if method == 'default':
        # weights = weights
        pass
        
    if method == 'reciprocal':
        #this gives larger weight to small errors and for large errors, should disregard them almost entirely 
        weights = np.abs(1/weights)
        
    if method == 'log':
        #this moderates large errors
        weights = np.log(np.abs(weights) + 1)
        
    if method == 'log_reciprocal':
        #this gives larger weight to small errors but sets also moderates how much extremely small errors can be weighted 
        weights = np.log(np.abs(1/weights)) 
        
    if mask_large:
        #this gives larger weight to small errors but sets also moderates how much extremely small errors can be weighted 
        weights = np.where(np.abs(errors) <= mask_large, weights, 0) 
        
    return np.sum(np.multiply(errors,weights))/np.sum(weights)

####: For Simulating Production 
def simulate_weighted_average(df, weighting_col, error_col, groupby_cols = ['trade_date'], window_size = 2000, weighting_method = 'default', mask_large = 35):
    '''Simulates debiasing procedure in production by calculating rolling average bias 
    
    Window_size dictates the N preceding trades to estimate bias correction for each row. 
    Setting window_size larger than the dataframe is equivalent to using pd.expanding(). 
    '''
    
    #if we are weigthing by the error column then don't slice the column twice
    subset = [weighting_col, error_col]
    if weighting_col == error_col:  subset = [error_col] 
    
    if window_size > len(df): window_size = len(df)
    
    #groupby.rolling.apply is not used here because it is both problematic and inefficient; converting to a list and iterating is faster
    groupby_dfs = list(df.groupby(groupby_cols)[subset].rolling(window_size, method='table')) 
    
    biases = []
    if mask_large:
        print(f'Ignoring trades with errors larger than {mask_large}bps in bias correction calculations.')
        
    for sub_df in groupby_dfs:
        biases.append(calculate_weighted_average(sub_df, weighting_col, error_col, method = weighting_method, mask_large = mask_large))
    
    return biases 


def bias_warm_start(bias_correction, df, N):
    '''Masks the first N trades of each day with 0.'''
    
    if len(bias_correction) != len(df): raise ValueError('Pred, truth, bias_correction must be same shape')
    
    def mask(S, N):
        S[:N] = 0
        return S

    temp = pd.DataFrame()
    temp['trade_date'] = df['trade_date']
    temp['bias_correction'] = bias_correction
    
    return temp.groupby('trade_date').apply(lambda x: mask(x, N))['bias_correction'].values


def calculate_weighted_average_masked(data, weighting_col, error_col, method = 'default', mask_large = None, seconds_ago_mask = 60):
    '''Calculates weighted average of error_col based on weighting_col AND masks trades less than X seconds ago. 
    
    If weighted average is to be calculated based on error magnitude, weighting_col should be set to error_col. 
    Different ways of calculating the weighted average are dictated by the method keyword argument.
    By default, trades less than 60 seconds ago are not used to estimate biases. 
    '''
    
    target_datetime = data.iloc[-1]['first_published_datetime']
    data = data.iloc[:-1] #removes current trade, though this is redundant now with the filtering by published datetime 
    
    #remove all trades that were published less than X seconds before given trade's trade_datetime
    data = data[(target_datetime - data['first_published_datetime']).dt.total_seconds() >= seconds_ago_mask]
    if len(data) == 0: return 0 

    errors = data[error_col].to_numpy()
    weights = data[weighting_col].to_numpy()
    
    if method == 'simple_average':
        weights = np.ones(len(errors))

    if method == 'default':
        #if method is default, weights are left as is 
        pass

    if method == 'reciprocal':
        #this gives larger weight to small errors and for large errors, should disregard them almost entirely 
        weights = np.abs(1/weights)

    if method == 'log':
        #this moderates large errors
        weights = np.log(np.abs(weights) + 1)

    if method == 'log_reciprocal':
        #this gives larger weight to small errors but sets also moderates how much extremely small errors can be weighted 
        weights = np.log(np.abs(1/weights)) 

    if mask_large:
        #this gives larger weight to small errors but sets also moderates how much extremely small errors can be weighted 
        weights = np.where(np.abs(errors) <= mask_large, weights, 0) 

    return np.sum(np.multiply(errors,weights))/np.sum(weights)

# ###: Simulating Production 

def simulate_weighted_average_masked(df, weighting_col, error_col, groupby_cols = ['trade_date'], window_size = 2000, weighting_method = 'default', mask_large = 35, seconds_ago_mask = 60):
    '''Simulates debiasing procedure in production by calculating rolling average bias AND masks trades less than X seconds ago. 
    
    Window_size dictates the N preceding trades to estimate bias correction for each row. 
    Setting window_size larger than the dataframe is equivalent to using pd.expanding(). 
    '''
    
    subset = [weighting_col, error_col, 'published_datetime']
    if weighting_col == error_col:  subset.remove(error_col) #if we are weigthing by the error column then don't slice the column twice
    
    if window_size > len(df): window_size = len(df)
    groupby_dfs = list(df.groupby(groupby_cols)[subset].rolling(window_size, method='table'))
    
    biases = []
    
    if mask_large:
        print(f'Ignoring trades with errors larger than {mask_large}bps in bias correction calculations.')
    for sub_df in groupby_dfs:
        biases.append(calculate_weighted_average_masked(sub_df, weighting_col, error_col, method = weighting_method, mask_large = mask_large, seconds_ago_mask = seconds_ago_mask))
    
    return biases 

####: Analysis of Results
def analyze_debiasing_MAE(df, bias_correction, date = None, prediction_col = 'prediction', target_col='new_ys'):
    '''Calculates debiased predictions and compares MAE before and after debiasing.'''
    
    if len(bias_correction) != len(df): raise ValueError('df, bias_correction must be same length')
    
    summary = df[['trade_date', prediction_col, target_col]].set_index('trade_date')
    summary.loc[:, 'bias_correction'] = bias_correction
    summary.loc[:,'debiased_prediction'] = summary[prediction_col] - bias_correction
    
    if not date: 
        date = summary.index[0].strftime('%Y-%m-%d')
    
    day_summary = pd.DataFrame(columns=['Original MAE', 'Corrected MAE'])
    for day in summary.index.unique().strftime('%Y-%m-%d'):
        day_data = summary.loc[day]
        day_summary = day_summary.append(dict(zip(day_summary.columns,[mean_absolute_error(day_data[prediction_col], day_data[target_col]),
                                                                      mean_absolute_error(day_data['debiased_prediction'], day_data[target_col])])), 
                                        ignore_index=True)
    day_summary.index = summary.index.unique().strftime('%Y-%m-%d')
    day_summary['Corrected - Original MAE'] = day_summary['Corrected MAE'] - day_summary['Original MAE'] 
    return day_summary

def analyze_debiasing_day(date, df, bias_correction, prediction_col = 'prediction', target_col='new_ys', first_N_trades = 500, last_N_trades = 200, print_graphs = True):
    '''Calculates debiased predictions for a given day and prints graphs for comparison of MAE.'''
    
    
    if len(bias_correction) != len(df): raise ValueError('df, bias_correction must be same length')
    
    
    # relevant_cols = ['cusip', 'rtrs_control_number', 'trade_datetime', 'trade_date', 'trade_type', 'quantity', prediction_col, target_col]
    relevant_cols = ['cusip', 'trade_datetime', 'trade_date', 'trade_type',  prediction_col, target_col]
    # if 'first_published_datetime' in df.columns: 
    #     relevant_cols.insert(2, 'first_published_datetime')
    
    # summary = df[relevant_cols].set_index('trade_datetime')
    # if date not in summary.index.unique().strftime('%Y-%m-%d'): raise KeyError(f'{date} is not in the provided dataframe')
    summary = df[relevant_cols]
    summary.loc[:, 'bias_correction'] = bias_correction
    summary.loc[:,'debiased_prediction'] = summary[prediction_col] - bias_correction
    summary.loc[:,'original_error'] = summary[target_col] - summary[prediction_col]
    summary.loc[:,'debiased_error'] = summary[target_col] - summary['debiased_prediction']
    summary = summary.set_index('trade_datetime')
    summary = summary.loc[date].sort_values(by='trade_datetime', ascending=True)
    
    
    if print_graphs:
        N = len(summary)
        
        if N < first_N_trades:
            first_N_trades = N
            print(f'Only {N} trades in data for {date}, defaulting {first_N_trades} to {N}')

        if N < last_N_trades:
            last_N_trades = N
            print(f'Only {N} trades in data for {date}, defaulting {last_N_trades} to {N}')


        win_rates_df = pd.DataFrame()
        win_rates_df['win_rate_original'] = np.cumsum(np.where(summary['debiased_error'].apply(abs) > summary['original_error'].apply(abs), 1, 0))
        win_rates_df['win_rate_debiased'] = np.cumsum(np.where(summary['debiased_error'].apply(abs) < summary['original_error'].apply(abs), 1, 0))


        rolling_MAE_original = summary.expanding(min_periods=1)['original_error'].apply(lambda x: np.average(np.abs(x)))
        rolling_MAE_debiased = summary.expanding(min_periods=1)['debiased_error'].apply(lambda x: np.average(np.abs(x)))
        print(f'OVERALL ORIGINAL MAE: {rolling_MAE_original[-1].mean():.2f}, OVERALL DEBIASED MAE: {rolling_MAE_debiased[-1].mean():.2f}')

        fig, ax = plt.subplots(4, 1, figsize=(20, 40))
        s = 8

        ax[0].scatter(range(last_N_trades), summary[prediction_col][-last_N_trades:], label='Original', s = s, c = 'blue')
        ax[0].scatter(range(last_N_trades), summary[target_col][-last_N_trades:], label='Truth', s = s, c='green')
        ax[0].scatter(range(last_N_trades), summary['debiased_prediction'][-last_N_trades:], label='Corrected', s = s, c = 'red')
        ax[0].set_title(f'Visualization of Actual, Original Predicted, Debiased Predicted {target_col} on {date} for last {last_N_trades} trades')
        ax[0].legend()

        ax[1].plot(range(N), win_rates_df['win_rate_original'], label='Original Prediction Wins')
        ax[1].plot(range(N), win_rates_df['win_rate_debiased'], label='Debiased Prediction Wins')       
        # ax[1].scatter(win_rates_df['win_rate_original'].dropna().index, win_rates_df['win_rate_original'].dropna(), label='ORIGINAL', c='red')
        # ax[1].scatter(win_rates_df['win_rate_debiased'].dropna().index, win_rates_df['win_rate_debiased'].dropna(), label='DEBIASED', c='blue')    
        ax[1].set_title(f'Model Wins for {date}')
        ax[1].legend()


        ax[2].plot(range(N), rolling_MAE_original, label='rolling_MAE_original')
        ax[2].plot(range(N), rolling_MAE_debiased, label='rolling_MAE_debiased')       
        ax[2].set_title(f'Rolling Average MAE for {date}')
        ax[2].legend()

        ax[3].plot(range(first_N_trades), rolling_MAE_original[:first_N_trades], label='rolling_MAE_original')
        ax[3].plot(range(first_N_trades), rolling_MAE_debiased[:first_N_trades], label='rolling_MAE_debiased')        
        ax[3].set_title(f'Rolling Average MAE for {date} for first {first_N_trades} trades') 
        ax[3].legend()
    
    return summary    



