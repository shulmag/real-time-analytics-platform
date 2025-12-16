import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers
import numpy as np
import gcsfs
import os
import pickle5 as pickle
import matplotlib.pyplot as plt
NUM_EPOCHS = None
import pandas as pd
calc_day_cats = {0: 'next_call_date', 1: 'par_call_date', 2: 'maturity_date', 3: 'refund_date'}

train_start = '2023-01-01'
train_end = '2023-03-01'
test_start = '2023-03-01'
test_end = '2023-04-01'
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 0.0001
BATCH_SIZE = 1024
NUM_EPOCHS = 100
DROPOUT = 0.1


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

class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.wq = layers.Dense(depth, name='weights_query', dtype='float32') 
        self.wk = layers.Dense(depth, name='weights_key', dtype='float32')
        self.wv = layers.Dense(depth, name='weights_value', dtype='float32')

    def scaled_dot_product_attention(self, q, v, k):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        scaling = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(scaling)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=1) 
        output = tf.matmul(attention_weights, v)
        
        return output
    
    def call(self, q, v, k):
        
        q = self.wq(q)
        v = self.wv(v)
        k = self.wk(k)

        output = self.scaled_dot_product_attention(q, v, k)
        
        return output   
    
    def get_config(self):
        cfg = super().get_config()
        return cfg   
    
import time

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

    def on_train_end(self, logs):
        print(f'Model training time was {sum(self.times)/60:.2f} minutes.')
        print(f'Average time for each epoch was {(sum(self.times)/60)/NUM_EPOCHS:.2f} minutes.')
        

class CSVLoggerTimeHistory(tf.keras.callbacks.CSVLogger):
  
    def on_train_begin(self, logs={}):
        super().on_train_begin()
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        super().on_epoch_begin(batch)
        self.epoch_time_start = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        import csv, collections
        logs = logs or {}
        self.times.append(time.time() - self.epoch_time_start)
        
        ###Following block is from CSVLogger source
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif (
                isinstance(k, collections.abc.Iterable)
                and not is_zero_dim_ndarray
            ):
                return f"\"[{', '.join(map(str, k))}]\""
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict(
                (k, logs[k]) if k in logs else (k, "NA") for k in self.keys
            )

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep
            
            ### Modification made here to add time taken per epoch
            fieldnames = ["epoch", "time taken"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        ### Modification made here to add time taken per epoch
        row_dict = collections.OrderedDict({"epoch": epoch, "time taken":np.round(self.times[-1],2)})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
        
        
    def on_train_end(self, logs):
        import collections
        row_dict = collections.OrderedDict({"epoch": "TOTAL", "time taken": np.round(sum(self.times),2)})
        self.writer.writerow(row_dict)
        self.csv_file.flush()
        super().on_train_end()
        
        print(f'Model training time was {sum(self.times)/60:.2f} minutes ({sum(self.times):.2f} seconds).')
        print(f'Average time for each epoch was {(sum(self.times)/60)/NUM_EPOCHS:.2f} minutes ({sum(self.times)/NUM_EPOCHS:.2f} seconds).')
        
        
def sanity_check(data, cols): 
    
    if len(cols)!= 2 or not set(cols).issubset(set(data.columns)): return 'Use valid cols'
    
    col1 = cols[0]
    col2 = cols[1]
    
    fig, ax = plt.subplots(2, 3, figsize=(20,20))
    ax[0,0].scatter(data[col1],  data[col2], s = 5)
    ax[0,0].set_title('Overall comparison for all calc day cat') 
    ax[0,0].set_xlabel(col1)
    ax[0,0].set_ylabel(col2)
    
    for i, axes in enumerate(ax.flatten()[1:]):
        if i > 3: break
        subset = data.last_calc_day_cat == i
        axes.scatter(data[col1][subset], data[col2][subset], s = 5)
        axes.set_title(f'Calc Day Cat == {i}, {calc_day_cats[i]}') 
        axes.set_xlabel(col1)
        axes.set_ylabel(col2)
        

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
    quantity_diff = np.log10(1 + np.abs(10**trade[2] - 10**row.quantity))
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