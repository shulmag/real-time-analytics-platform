import numpy as np
import gcsfs
import os
import pickle5 as pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from IPython.display import display

calc_day_cats = {0: 'next_call_date', 1: 'par_call_date', 2: 'maturity_date', 3: 'refund_date'}

def load_data_from_pickle(path):
    if os.path.isfile(path):
        print('File available, loading pickle')
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        print(f'File not available, downloading from cloud storage and saving to {path}')
        fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
        gc_path = os.path.join('isaac_data',path)
        print(gc_path)
        with fs.open(gc_path) as gf:
            data = pd.read_pickle(gf)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    return data

###### ENCODERS ######    
categorical_feature_values = {'purpose_class' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                                 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                                 47, 48, 49, 50, 51, 52, 53],
                              'rating' : ['A', 'A+', 'A-', 'AA', 'AA+', 'AA-', 'AAA', 'B', 'B+', 'B-', 'BB', 'BB+', 'BB-',
                                         'BBB', 'BBB+', 'BBB-', 'CC', 'CCC', 'CCC+', 'CCC-' , 'D', 'NR', 'MR'],
                              'trade_type' : ['D', 'S', 'P'],
                              'incorporated_state_code' : ['AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'GU',
                                                         'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
                                                         'MO', 'MP', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
                                                         'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'US', 'UT', 'VA', 'VI',
                                                         'VT', 'WA', 'WI', 'WV', 'WY'] }

##### EXTRA DATA PREPROCESSING #####
ttype_dict = { (0,0):'D', (0,1):'S', (1,0):'P' }
ys_variants = ["max_ys", "min_ys", "max_qty", "min_ago", "D_min_ago", "P_min_ago", "S_min_ago"]
ys_feats = ["_ys", "_ttypes", "_ago", "_qdiff"]
D_prev = dict()
P_prev = dict()
S_prev = dict()

def get_trade_history_columns():
    '''
    This function is used to create a list of columns
    '''
    YS_COLS = []
    for prefix in ys_variants:
        for suffix in ys_feats:
            YS_COLS.append(prefix + suffix)
    return YS_COLS

def extract_feature_from_trade(row, name, trade):
    yield_spread = trade[0]
    ttypes = ttype_dict[(trade[3],trade[4])] + row.trade_type
    seconds_ago = trade[5]
    diff = 10**trade[2] - 10**row.quantity
    quantity_diff = np.sign(diff) * np.log10(1 + np.abs(diff))
    return [yield_spread, ttypes,  seconds_ago, quantity_diff]

def trade_history_derived_features(row):
    trade_history = row.trade_history
    trade = trade_history[0]
    
    D_min_ago_t = D_prev.get(row.cusip,trade)
    D_min_ago = 9        

    P_min_ago_t = P_prev.get(row.cusip,trade)
    P_min_ago = 9
    
    S_min_ago_t = S_prev.get(row.cusip,trade)
    S_min_ago = 9
    
    max_ys_t = trade; max_ys = trade[0]
    min_ys_t = trade; min_ys = trade[0]
    max_qty_t = trade; max_qty = trade[2]
    min_ago_t = trade; min_ago = trade[5]
    
    for trade in trade_history[0:]:
        #Checking if the first trade in the history is from the same block
        if trade[5] == 0: 
            continue
 
        if trade[0] > max_ys: 
            max_ys_t = trade
            max_ys = trade[0]
        elif trade[0] < min_ys: 
            min_ys_t = trade; 
            min_ys = trade[0]

        if trade[2] > max_qty: 
            max_qty_t = trade 
            max_qty = trade[2]
        if trade[5] < min_ago: 
            min_ago_t = trade; 
            min_ago = trade[5]
            
        side = ttype_dict[(trade[3],trade[4])]
        if side == "D":
            if trade[5] < D_min_ago: 
                D_min_ago_t = trade; D_min_ago = trade[5]
                D_prev[row.cusip] = trade
        elif side == "P":
            if trade[5] < P_min_ago: 
                P_min_ago_t = trade; P_min_ago = trade[5]
                P_prev[row.cusip] = trade
        elif side == "S":
            if trade[5] < S_min_ago: 
                S_min_ago_t = trade; S_min_ago = trade[5]
                S_prev[row.cusip] = trade
        else: 
            print("invalid side", trade)
    
    trade_history_dict = {"max_ys":max_ys_t,
                          "min_ys":min_ys_t,
                          "max_qty":max_qty_t,
                          "min_ago":min_ago_t,
                          "D_min_ago":D_min_ago_t,
                          "P_min_ago":P_min_ago_t,
                          "S_min_ago":S_min_ago_t}

    return_list = []
    for variant in ys_variants:
        feature_list = extract_feature_from_trade(row,variant,trade_history_dict[variant])
        return_list += feature_list
    
    return return_list

def addflag(flag, condition, name):
    empty = flag == "none"
    flag[condition & empty] = name
    flag[condition & ~empty] = flag[condition & ~empty] + " & " + name
    
def addcol(data, newname, newvals, warn=False):
    if newname in data.columns:
        if warn: print( f"Warning: replacing duplicate column {newname}" )
        data[newname] = newvals
    else:
        newcol = pd.Series(newvals, index = data.index, name=newname)
        data = pd.concat([data,newcol],axis=1)
    return data

def mkcases(df):
    flag = pd.Series("none", index=df.index)

    addflag(flag, df.last_yield.isna(), "no last yld")
    addflag(flag, df.last_yield < 150, "last yld < 1.5%")
    addflag(flag, df.last_yield.between(150,700), "1.5% <= last yld <= 7%")
    addflag(flag, df.last_yield > 700, "last yld > 7%")
    addflag(flag, df.when_issued, "when issued")
    
    print( flag.value_counts(dropna=False) )
    return flag.astype('category')


def mean_absolute_deviation(pred, truth):
    pred, truth = np.array(pred).reshape(-1,1), np.array(truth).reshape(-1,1)
    err = abs(pred - truth)
    return np.median(err)

def compare_mae(df, prediction_cols, groupby_cols, target_variable):
    
    if not isinstance(prediction_cols, list):
        raise TypeError(f'prediction_cols must be a list, got {type(prediction_cols)}, {type(groupby_cols)} instead')
    
    if groupby_cols and not isinstance(groupby_cols, list):
        raise TypeError(f'groupby_cols must be a list or None, got {type(groupby_cols)} instead')
    
    print(f'{f" Analysis for target: {target_variable} ":=^75}')
    
    nan_counts = df[prediction_cols].isna().sum() 
    
    for x,y  in df[prediction_cols].isna().sum().iteritems():
        print(f'Prediction col {x} has {y} nan values')
    
    df = df.dropna(subset=prediction_cols)

    if groupby_cols:
        temp = df[[target_variable, 'cases'] + prediction_cols + groupby_cols]\
                .groupby(groupby_cols, observed=True)\
                .apply(lambda x: [mean_absolute_error(x[target_variable], x[col]) for col in prediction_cols] + \
                        [mean_absolute_deviation(x[target_variable], x[col]) for col in prediction_cols] + [len(x)])   
        temp = pd.DataFrame(temp.to_list(), index = zip(['Overall']*len(temp),temp.index))

        temp2 = df[[target_variable, 'cases'] + prediction_cols + groupby_cols]\
                .groupby(['cases']+ groupby_cols, observed=True)\
                .apply(lambda x: [mean_absolute_error(x[target_variable], x[col]) for col in prediction_cols] + \
                        [mean_absolute_deviation(x[target_variable], x[col]) for col in prediction_cols] + [len(x)])   
        temp2 = pd.DataFrame(temp2.to_list(), index = temp2.index)
        summary = pd.concat([temp, temp2], axis=0)

    else:
        
        temp2 = df[[target_variable, 'cases'] + prediction_cols]\
                .groupby('cases', observed=True)\
                .apply(lambda x: [mean_absolute_error(x[target_variable], x[col]) for col in prediction_cols] + \
                        [mean_absolute_deviation(x[target_variable], x[col]) for col in prediction_cols] + [len(x)])   
        temp2 = pd.DataFrame(temp2.to_list(), index = temp2.index)
        
        temp = pd.DataFrame([mean_absolute_error(df[target_variable], df[col]) for col in prediction_cols] + \
                        [mean_absolute_deviation(df[target_variable], df[col]) for col in prediction_cols] + [len(df)], columns=['Overall']).T
    
    summary = pd.concat([temp, temp2], axis=0)  
    mae_col = ['MAE']*len(prediction_cols)
    mad_col = ['MAD']*len(prediction_cols)
    columns= list(zip(mae_col, prediction_cols)) + list(zip(mad_col, prediction_cols)) + [('', 'N')]
    summary.columns=pd.MultiIndex.from_tuples(columns)
    
    if groupby_cols:
        summary.index=pd.MultiIndex.from_tuples(summary.index, names = ['cases']+groupby_cols)
    else:
        pass
    
    summary[('', 'N')] = summary[('', 'N')].astype(int)
    return summary