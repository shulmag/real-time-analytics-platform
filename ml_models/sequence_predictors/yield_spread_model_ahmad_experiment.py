import pandas as pd
import time
import datetime

import numpy as np
from google.cloud import bigquery
from google.cloud import storage
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from sklearn import preprocessing
import pickle

import lightgbm
from lightgbm import LGBMRegressor

import os

from trade_history_model_mitas.add_related_trades import add_related_trades

from ficc.utils.auxiliary_variables import PREDICTORS, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES, IDENTIFIERS


print(f'Tensorflow version: {tf.__version__}')


print('Setting the environment variables')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="mitas_creds.json"
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None


bq_client = bigquery.Client()
storage_client = storage.Client()


TRAIN_TEST_SPLIT = 0.85
LEARNING_RATE = 0.0001
BATCH_SIZE = 1000
NUM_EPOCHS = 100

DROPOUT = 0.01
SEQUENCE_LENGTH = 5
NUM_FEATURES = 6


print('We grab the data from a GCP bucket. The data is prepared using the ficc python package.')
import gcsfs
fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
loading_data_start_time = time.time()
# with fs.open('ahmad_data/processed_data_2022-10-24-20:56.pkl') as f:
# with fs.open('ahmad_data/processed_data_2022-10-24-20:56_c_spread.pkl') as f:
# with fs.open('ahmad_data/processed_data_2022-12-02-17:52.pkl') as f:
with fs.open('ficc_training_data_latest/processed_data_2022-12-28-09:42.pkl') as f:
    data = pd.read_pickle(f)
print(f'Time elapsed to load ficc_training_data_latest/processed_data_2022-12-28-09:42.pkl from GCP: {datetime.timedelta(seconds=time.time() - loading_data_start_time)}')


print(f'Most recent trade date: {data.trade_date.max()}')


print(f'Size of data: {len(data)}')


print(f'Printing identifiers for all rows where the calc_date is before the settlement_date')
print(data[data['settlement_date'] > data['calc_date']][IDENTIFIERS + ['calc_date', 'settlement_date']])
print(f'Remove trades where the settlement_date comes after the calc_date')
data = data[data.settlement_date <= data.calc_date]
print(f'Size of data: {len(data)}')


ytw = data['yield']
print( ((ytw - data.yield_spread - data.ficc_ycl)**2).mean() )

lys = data.last_yield_spread

last_ytw = 0 * ytw
same_ys = 0 * lys

sub = data.last_calc_day_cat == 0
last_ytw[sub] = lys[sub] + data[sub].last_ficc_ycl_to_next_call 
same_ys[sub] =  ytw[sub] - data[sub].ficc_ycl_to_next_call

sub = data.last_calc_day_cat == 1
last_ytw[sub] = lys[sub] + data[sub].last_ficc_ycl_to_par_call
same_ys[sub] =  ytw[sub] - data[sub].ficc_ycl_to_par_call

sub = data.last_calc_day_cat == 2
last_ytw[sub] = lys[sub] + data[sub].last_ficc_ycl_to_maturity 
same_ys[sub] =  ytw[sub] - data[sub].ficc_ycl_to_maturity 

sub = data.last_calc_day_cat == 3
last_ytw[sub] = lys[sub] + data[sub].last_ficc_ycl_to_refund 
same_ys[sub] =  ytw[sub] - data[sub].ficc_ycl_to_refund 

data['same_ys'] = same_ys
data['ytw'] = ytw
data['last_ytw'] = last_ytw


print(f'Size of data: {len(data)}')


print('Drop rows with null value for same_ys')
data.dropna(subset=['same_ys'],inplace=True)


print(f'Size of data: {len(data)}')


print('Adding features from Charles notebook')
data['ttypes'] = (data.last_trade_type.astype(str) + data.trade_type.astype(str)).astype('category')
data['diff_size'] = (data.par_traded - data.last_size).astype(np.float32)
data['abs_last_yield_spread'] = np.abs(data['last_yield_spread'])
data['abs_diff_size'] = np.abs(data['diff_size'])
data['days_duration'] = (data.last_calc_date - data.last_settlement_date).dt.days


additional_features = ['ttypes', 'diff_size', 'abs_last_yield_spread', 'abs_diff_size', 'days_duration']


data['trade_history_sum'] = data.trade_history.parallel_apply(lambda x: np.sum(x))


data.purpose_sub_class.fillna(0, inplace=True)


print('Adding the concatenated trade type, target trade attention, and treasury spread over ficc ycl in PREDICTORS to train the model.')


if 'ttypes' not in CATEGORICAL_FEATURES:
    CATEGORICAL_FEATURES.append('ttypes')
    PREDICTORS.append('ttypes')


data['ted-rate'] = (data['t_rate_10'] - data['t_rate_2']) * 100


if 'target_attention_features' not in PREDICTORS:
    PREDICTORS.append('target_attention_features')


if 'ficc_treasury_spread' not in PREDICTORS:
    PREDICTORS.append('ficc_treasury_spread')
    NON_CAT_FEATURES.append('ficc_treasury_spread')


auxiliary_features = ['dollar_price',
                     'calc_date', 
                     'trade_date',
                     'trade_datetime', 
                     'purpose_sub_class', 
                     'called_redemption_type', 
                    #  'calc_day_cat',    # removing since it is already included in PREDICTORS since PREDICTORS includes TARGET
                     'yield',
                     'ficc_ycl',
                     'same_ys',
                     'trade_history_sum',
                     'sale_type',
                     'days_to_refund', 
                     'muni_security_type', 
                     'settlement_date', 
                     'is_called', 
                     'par_traded']


processed_data = data[IDENTIFIERS + PREDICTORS + auxiliary_features]


print(f'Dropping NaN values in trade history.')
processed_data.dropna(inplace=True, subset=PREDICTORS+['trade_history_sum'])


print(f'Size of data: {len(processed_data)}')


processed_data.issue_amount = processed_data.issue_amount.replace([np.inf, -np.inf], np.nan)


processed_data.dropna(inplace=True, subset=PREDICTORS)


print(f'Size of data: {len(processed_data)}')


print('Adding most recent related trade')
CATEGORICAL_REFERENCE_FEATURES_TO_ADD = ['rating', 'incorporated_state_code']
assert set(CATEGORICAL_REFERENCE_FEATURES_TO_ADD) <= set(CATEGORICAL_FEATURES)
related_trade_feature_prefix = 'related_last_'

adding_related_trades_start_time = time.time()
processed_data, related_trades_binary_features, related_trades_categorical_features, related_trades_non_cat_features = add_related_trades(processed_data, related_trade_feature_prefix, 1, CATEGORICAL_REFERENCE_FEATURES_TO_ADD)
print(f'Time elapsed to add related trades: {datetime.timedelta(seconds=time.time() - adding_related_trades_start_time)}')

print(f'Added binary features: {related_trades_binary_features}, categorical features: {related_trades_categorical_features}, non cat features: {related_trades_non_cat_features}')
BINARY.extend(related_trades_binary_features)
CATEGORICAL_FEATURES.extend(related_trades_categorical_features)
NON_CAT_FEATURES.extend(related_trades_non_cat_features)
PREDICTORS.extend(related_trades_binary_features + related_trades_categorical_features + related_trades_non_cat_features)


print(f'Printing identifiers for all rows with nan values for settlement_date_to_calc_date feature for related trade')
print(processed_data[processed_data[related_trade_feature_prefix + 'settlement_date_to_calc_date'].isna()][IDENTIFIERS])
processed_data.dropna(inplace=True, subset=PREDICTORS)


print(f'Size of data: {len(processed_data)}')


print(f'Fitting encoders to the categorical features: {CATEGORICAL_FEATURES}. These encoders are then used to encode the categorical features of the train and test set')
encoders = {}
fmax = {}
for f in CATEGORICAL_FEATURES:
    print(f)
    fprep = preprocessing.LabelEncoder().fit(processed_data[f].drop_duplicates())
    fmax[f] = np.max(fprep.transform(fprep.classes_))
    encoders[f] = fprep
    
with open('encoders.pkl','wb') as file:
    pickle.dump(encoders,file)


print('Splitting the data into train and test sets')
train_dataframe = processed_data[(processed_data.trade_date < '11-01-2022')]
test_dataframe = processed_data[(processed_data.trade_date >= '11-01-2022') & (processed_data.trade_date <= '11-30-2022') ]


print(f'Size of train data: {len(train_dataframe)}')
print(f'Size of test data: {len(test_dataframe)}')


def create_input(df):
    global encoders
    datalist = []
    datalist.append(np.stack(df['trade_history'].to_numpy()))
    datalist.append(np.stack(df['target_attention_features'].to_numpy()))

    noncat_and_binary = []
    for f in NON_CAT_FEATURES + BINARY:
        noncat_and_binary.append(np.expand_dims(df[f].to_numpy().astype('float32'), axis=1))
    datalist.append(np.concatenate(noncat_and_binary, axis=-1))
    
    for f in CATEGORICAL_FEATURES:
        encoded = encoders[f].transform(df[f])
        datalist.append(encoded.astype('float32'))
    
    return datalist


x_train = create_input(train_dataframe)
y_train = train_dataframe.yield_spread
# y_train = train_dataframe.same_ys


x_test = create_input(test_dataframe)
y_test = test_dataframe.yield_spread
# y_test = test_dataframe.same_ys


print(f'len(x_train): {len(x_train)}')
print(f'x_train[0].shape: {x_train[0].shape}')
print(f'x_train[1].shape: {x_train[1].shape}')
print(f'x_test[2].shape: {x_test[2].shape}')


def gbmprep(df):
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype('category')
    df[NON_CAT_FEATURES + BINARY] = df[NON_CAT_FEATURES + BINARY].astype('float64')
    df = df.drop(columns = ['trade_history','target_attention_features','yield_spread','calc_day_cat'])
    return df


trainlabel = train_dataframe.yield_spread
testlabel = test_dataframe.yield_spread


gbt_td = gbmprep(train_dataframe[PREDICTORS])


gbtmodel = LGBMRegressor(num_iterations=300, max_depth=12, num_leaves=300, objective='mae', verbosity=-1)


print('Training LightGBM model')
gbtmodel.fit(gbt_td, trainlabel)


gbt_pred = gbtmodel.predict( gbmprep(test_dataframe[PREDICTORS]) )
delta = testlabel - gbt_pred
# print(delta.mean())
print(f'Test loss: {delta.abs().mean()}')


lightgbm.plot_importance(gbtmodel, figsize=(10,10), importance_type='gain').figure.savefig('lightgbm_feature_importance_plot.pdf', bbox_inches='tight')


trade_history_normalizer = Normalization(name='Trade_history_normalizer')
trade_history_normalizer.adapt(x_train[0], batch_size=BATCH_SIZE)


noncat_binary_normalizer = Normalization(name='Numerical_binary_normalizer')
noncat_binary_normalizer.adapt(x_train[2], batch_size=BATCH_SIZE)


tf.keras.utils.set_random_seed(10)


class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, depth):
        super(CustomAttention, self).__init__()
        self.depth = depth
        self.wq = layers.Dense(depth, name='weights_query') 
        self.wk = layers.Dense(depth, name='weights_key')
        self.wv = layers.Dense(depth, name='weights_value')

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


inputs = []
layer = []

############## INPUT BLOCK ###################
trade_history_input = layers.Input(name="trade_history_input", 
                                   shape=(SEQUENCE_LENGTH,NUM_FEATURES), 
                                   dtype = tf.float32) 

target_attention_input = layers.Input(name="target_attention_input", 
                                   shape=(SEQUENCE_LENGTH, 3), 
                                   dtype = tf.float32)

print('Finished creating input layer initializations')

inputs.append(trade_history_input)
inputs.append(target_attention_input)

inputs.append(layers.Input(
    name="NON_CAT_AND_BINARY_FEATURES",
    shape=(len(NON_CAT_FEATURES + BINARY),)
))

noncat_binary_normalized = noncat_binary_normalizer(inputs[2])

print('Finished normalizing the the noncat and binary features')

layer.append(noncat_binary_normalized)
####################################################

print('Finished creating input block')

############## TRADE HISTORY MODEL #################

lstm_layer = layers.LSTM(50, 
                         activation='tanh',
                         input_shape=(SEQUENCE_LENGTH,NUM_FEATURES),
                         return_sequences = True,
                         name='LSTM')

lstm_attention_layer = CustomAttention(50)

lstm_layer_2 = layers.LSTM(100, 
                           activation='tanh',
                           input_shape=(SEQUENCE_LENGTH,50),
                           return_sequences = False,
                           name='LSTM_2')

trade_history_normalized = trade_history_normalizer(inputs[0])

print('Finished normalizing the trade history')

features = lstm_layer(trade_history_normalized)
features = lstm_attention_layer(features, features, inputs[1])
features = layers.BatchNormalization()(features)
# features = layers.Dropout(DROPOUT)(features)

print('Finished first batch normalization in the LSTM')

features = lstm_layer_2(features)
features = layers.BatchNormalization()(features)
# features = layers.Dropout(DROPOUT)(features)

print('Finished second batch normalization in the LSTM')

trade_history_output = layers.Dense(100, 
                                    activation='relu')(features)

####################################################

print('Finished creating trade history model')

############## REFERENCE DATA MODEL ################
# global encoders
for f in CATEGORICAL_FEATURES:
    fin = layers.Input(shape=(1,), name = f)
    inputs.append(fin)
    embedded = layers.Flatten(name = f + "_flat")( layers.Embedding(input_dim = fmax[f]+1,
                                                                    output_dim = max(30,int(np.sqrt(fmax[f]))),
                                                                    input_length= 1,
                                                                    name = f + "_embed")(fin))
    layer.append(embedded)

    
reference_hidden = layers.Dense(400,
                                activation='relu',
                                name='reference_hidden_1')(layers.concatenate(layer, axis=-1))

reference_hidden = layers.BatchNormalization()(reference_hidden)
reference_hidden = layers.Dropout(DROPOUT)(reference_hidden)

reference_hidden2 = layers.Dense(200,activation='relu',name='reference_hidden_2')(reference_hidden)
reference_hidden2 = layers.BatchNormalization()(reference_hidden2)
reference_hidden2 = layers.Dropout(DROPOUT)(reference_hidden2)

reference_output = layers.Dense(100,activation='tanh',name='reference_hidden_3')(reference_hidden2)

####################################################

print('Finished creating reference data model')

feed_forward_input = layers.concatenate([reference_output, trade_history_output])

hidden = layers.Dense(300,activation='relu')(feed_forward_input)
hidden = layers.BatchNormalization()(hidden)
hidden = layers.Dropout(DROPOUT)(hidden)

hidden2 = layers.Dense(100,activation='tanh')(hidden)
hidden2 = layers.BatchNormalization()(hidden2)
hidden2 = layers.Dropout(DROPOUT)(hidden2)

final = layers.Dense(1)(hidden2)

model = keras.Model(inputs=inputs, outputs=final)


model.summary()


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()


fit_callbacks = [
    #WandbCallback(save_model=False),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=0,
        mode="auto",
        restore_best_weights=True,
    ),
    time_callback
]


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
          loss=keras.losses.MeanAbsoluteError(),
          metrics=[keras.metrics.MeanAbsoluteError()])

training_start_time = time.time()
history = model.fit(x_train, 
                    y_train, 
                    epochs=100, 
                    batch_size=BATCH_SIZE, 
                    verbose=1, 
                    validation_split=0.1, 
                    callbacks=fit_callbacks,
                    use_multiprocessing=True,
                    workers=8)
print(f'Time elapsed to train model: {datetime.timedelta(seconds=time.time() - training_start_time)}')


n = len(x_train[0])
p = model.count_params()
avg_time = np.mean(time_callback.times)
gflops = ((n*p*2*3)/avg_time)/10**9
print(f'gflops: {gflops}')


_, mae = model.evaluate(x_test, y_test, verbose=1, batch_size=BATCH_SIZE)
print(f"Test loss: {round(mae, 3)}")


########## RESULTS ##########
#### No related trades: 12.7683 ####
#### Single related trade as reference feature: 12.5411 (local) ####
#### Single related trade as reference feature: 12.5328 (VM) ####