import pandas as pd
import numpy as np 
import pytz
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from sklearn.metrics import mean_absolute_error
import pickle5 as pickle
from google.cloud import storage
import gcsfs

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

BINARY, CATEGORICAL_FEATURES, NON_CAT_FEATURES = None, None, None
train_start = '2023-09-01'
train_end = '2023-09-25'
test_start = '2023-09-25'
test_end = '2023-09-29'

VALIDATION_SPLIT = 0.1
LEARNING_RATE = 0.0001
BATCH_SIZE = 1000
NUM_EPOCHS = 1
DROPOUT = 0.1 
shuffle = True
shuffle_buffer = 0.75
TRADE_SEQUENCE_LENGTH = 5
NUM_FEATURES = 6
target_variable = 'new_ys' 
trade_history_col = 'trade_history'


def reset_model_features():
    '''Function resets the model features, which are global variables, to their original state for convenience when running new architectures'''
    global NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES
    
    BINARY = ['callable',
          'sinking',
          'zerocoupon',
          'is_non_transaction_based_compensation',
          'is_general_obligation',
          'callable_at_cav',
          'extraordinary_make_whole_call',
          'make_whole_call',
          'has_unexpired_lines_of_credit',
          'escrow_exists']

    CATEGORICAL_FEATURES = ['rating',
                            'incorporated_state_code',
                            'trade_type',
                            'purpose_class',
                            'max_ys_ttypes',
                            'min_ys_ttypes',
                            'max_qty_ttypes',
                            'min_ago_ttypes',
                            'D_min_ago_ttypes',
                            'P_min_ago_ttypes',
                            'S_min_ago_ttypes']

    NON_CAT_FEATURES = ['quantity',
                        'days_to_maturity',
                         'days_to_call',
                         'coupon',
                         'issue_amount',
                         'last_seconds_ago',
                         'last_yield_spread',
                         'days_to_settle',
                         'days_to_par',
                         'maturity_amount',
                         'issue_price',
                         'orig_principal_amount',
                         'max_amount_outstanding',
                         'accrued_days',
                         'days_in_interest_payment',
                         'A/E',
                         'ficc_treasury_spread',
                         'max_ys_ys',
                         'max_ys_ago',
                         'max_ys_qdiff',
                         'min_ys_ys',
                         'min_ys_ago',
                         'min_ys_qdiff',
                         'max_qty_ys',
                         'max_qty_ago',
                         'max_qty_qdiff',
                         'min_ago_ys',
                         'min_ago_ago',
                         'min_ago_qdiff',
                         'D_min_ago_ys',
                         'D_min_ago_ago',
                         'D_min_ago_qdiff',
                         'P_min_ago_ys',
                         'P_min_ago_ago',
                         'P_min_ago_qdiff',
                         'S_min_ago_ys',
                         'S_min_ago_ago',
                         'S_min_ago_qdiff']

reset_model_features()

fs = gcsfs.GCSFileSystem()
with fs.open('automated_training/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# with open('gs://automated_training/encoders.pkl', 'rb') as f: 
# with open('encoders.pkl', 'rb') as f: 
    # encoders = pickle.load(f)
    
fmax = {key: len(value.classes_) for key, value in encoders.items()}

#MODEL AND DATA FUNCTIONS
def create_input(df, trade_history_col):
    global encoders
    datalist = []
        
    datalist.append(np.stack(df[trade_history_col].to_numpy()))
    datalist.append(np.stack(df['target_attention_features'].to_numpy()))

    noncat_and_binary = []
    for f in NON_CAT_FEATURES + BINARY:
        noncat_and_binary.append(np.expand_dims(df[f].to_numpy().astype('float64'), axis=1))
    datalist.append(np.concatenate(noncat_and_binary, axis=-1))
    
    for f in CATEGORICAL_FEATURES:
        encoded = encoders[f].transform(df[f])
        datalist.append(encoded.astype('float32'))
    
    return datalist

def generate_model_default(TRADE_SEQUENCE_LENGTH, trade_history_normalizer, noncat_binary_normalizer):
    inputs = []
    layer = []
    
    trade_history_input = layers.Input(name="trade_history_input", 
                                       shape=(TRADE_SEQUENCE_LENGTH, NUM_FEATURES), 
                                       dtype = tf.float32) 

    target_attention_input = layers.Input(name="target_attention_input", 
                                       shape=(1, 3), 
                                       dtype = tf.float32) 
    inputs.append(trade_history_input)
    inputs.append(target_attention_input)

    inputs.append(layers.Input(
        name="NON_CAT_AND_BINARY_FEATURES",
        shape=(len(NON_CAT_FEATURES + BINARY),)
    ))


    layer.append(noncat_binary_normalizer(inputs[2]))
    ####################################################


    ############## TRADE HISTORY MODEL #################

    lstm_layer = layers.Bidirectional(layers.LSTM(50, 
                             activation='tanh',
                             input_shape=(TRADE_SEQUENCE_LENGTH,NUM_FEATURES),
                             return_sequences = True,
                             name='LSTM'))

    lstm_layer_2 = layers.Bidirectional(layers.LSTM(100, 
                                                    activation='tanh',
                                                    input_shape=(TRADE_SEQUENCE_LENGTH, 50),
                                                    return_sequences = True,
                                                    name='LSTM_2'))



    features = lstm_layer(trade_history_normalizer(inputs[0]))
    features = lstm_layer_2(features)  
    
    
    attention_sequence = layers.Dense(200, activation='relu', name='attention_dense')(target_attention_input)
    attention = layers.Dot(axes=[2, 2])([features, attention_sequence])
    attention = layers.Activation('softmax')(attention)

    context_vector = layers.Dot(axes=[1, 1])([features, attention])
    context_vector = layers.Flatten(name='context_vector_flatten')(context_vector)


    trade_history_output = layers.Dense(100, 
                                        activation='relu')(context_vector)
    
 
    ####################################################

    ############## REFERENCE DATA MODEL ################
    global encoders
    for f in CATEGORICAL_FEATURES:
        
        fin = layers.Input(shape=(1,), name = f)
        inputs.append(fin)
        embedded = layers.Flatten(name = f + "_flat")( layers.Embedding(input_dim = fmax[f],
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

    reference_output = layers.Dense(100,activation='relu',name='reference_hidden_3')(reference_hidden2)
    ####################################################

    feed_forward_input = layers.concatenate([reference_output, trade_history_output])
    
    hidden = layers.Dense(300,activation='relu')(feed_forward_input)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(DROPOUT)(hidden)

    hidden2 = layers.Dense(100,activation='relu')(hidden)
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dropout(DROPOUT)(hidden2)

    final = layers.Dense(1)(hidden2)

    model = keras.Model(inputs=inputs, outputs=final)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
          loss=keras.losses.MeanAbsoluteError())
    
    return model
    

def generate_model_bottleneck(TRADE_SEQUENCE_LENGTH, trade_history_normalizer, noncat_binary_normalizer):
    inputs = []
    layer = []
    
    trade_history_input = layers.Input(name="trade_history_input", 
                                       shape=(TRADE_SEQUENCE_LENGTH, NUM_FEATURES), 
                                       dtype = tf.float32) 

    target_attention_input = layers.Input(name="target_attention_input", 
                                       shape=(1, 3), 
                                       dtype = tf.float32) 
    inputs.append(trade_history_input)
    inputs.append(target_attention_input)

    inputs.append(layers.Input(
        name="NON_CAT_AND_BINARY_FEATURES",
        shape=(len(NON_CAT_FEATURES + BINARY),)
    ))


    layer.append(noncat_binary_normalizer(inputs[2]))
    ####################################################


    ############## TRADE HISTORY MODEL #################

    lstm_layer = layers.Bidirectional(layers.LSTM(50, 
                             activation='tanh',
                             input_shape=(TRADE_SEQUENCE_LENGTH,NUM_FEATURES),
                             return_sequences = True,
                             name='LSTM'))

    lstm_layer_2 = layers.Bidirectional(layers.LSTM(100, 
                                                    activation='tanh',
                                                    input_shape=(TRADE_SEQUENCE_LENGTH, 50),
                                                    return_sequences = True,
                                                    name='LSTM_2'))



    features = lstm_layer(trade_history_normalizer(inputs[0]))
    features = lstm_layer_2(features)  
    
    
    attention_sequence = layers.Dense(200, activation='relu', name='attention_dense')(target_attention_input)
    attention = layers.Dot(axes=[2, 2])([features, attention_sequence])
    attention = layers.Activation('softmax')(attention)

    context_vector = layers.Dot(axes=[1, 1])([features, attention])
    context_vector = layers.Flatten(name='context_vector_flatten')(context_vector)


    trade_history_output = layers.Dense(100, 
                                        activation='relu')(context_vector)
    
    trade_history_output = layers.Dense(1)(trade_history_output)
 
    ####################################################

    ############## REFERENCE DATA MODEL ################
    global encoders
    for f in CATEGORICAL_FEATURES:
        
        fin = layers.Input(shape=(1,), name = f)
        inputs.append(fin)
        embedded = layers.Flatten(name = f + "_flat")( layers.Embedding(input_dim = fmax[f],
                                                                        output_dim = max(30,int(np.sqrt(fmax[f]))),
                                                                        input_length= 1,
                                                                        name = f + "_embed")(fin))
        layer.append(embedded)


    reference_hidden = layers.Dense(300,
                                    activation='relu',
                                    name='reference_hidden_1')(layers.concatenate(layer, axis=-1))
    reference_hidden = layers.BatchNormalization()(reference_hidden)
    reference_hidden = layers.Dropout(DROPOUT)(reference_hidden)

    reference_hidden2 = layers.Dense(200,activation='relu',name='reference_hidden_2')(reference_hidden)
    reference_hidden2 = layers.BatchNormalization()(reference_hidden2)
    reference_hidden2 = layers.Dropout(DROPOUT)(reference_hidden2)

    reference_hidden3 = layers.Dense(100,activation='relu',name='reference_hidden_3')(reference_hidden2)
    reference_hidden3 = layers.BatchNormalization()(reference_hidden3)
    reference_hidden3 = layers.Dropout(DROPOUT)(reference_hidden3)
    
    reference_output = layers.Dense(1, name='reference_output')(reference_hidden3)
    ####################################################

    feed_forward_input = layers.concatenate([reference_output, trade_history_output])
    
    final = layers.Dense(1)(feed_forward_input)

    model = keras.Model(inputs=inputs, outputs=final)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
          loss=keras.losses.MeanAbsoluteError())
    
    return model

def generate_model_ensemble(TRADE_SEQUENCE_LENGTH, trade_history_normalizer, noncat_binary_normalizer, ensemble_size):
    models = []
    for i in range(ensemble_size):
        if i < ensemble_size/2:
            models.append(generate_model_default(TRADE_SEQUENCE_LENGTH, trade_history_normalizer, noncat_binary_normalizer))
        else:
            models.append(generate_model_bottleneck(TRADE_SEQUENCE_LENGTH, trade_history_normalizer, noncat_binary_normalizer))
            
    input_layer = models[0].input
    output_list = [model(input_layer) for model in models]
    ensemble_model = keras.Model(inputs = input_layer, outputs = output_list)
    ensemble_model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
                 loss=[keras.losses.MeanAbsoluteError() for i in range(ensemble_size)])

    return ensemble_model

def create_data_set_and_model(train_dataframe, test_dataframe, trade_history_col):
    
    if not isinstance(trade_history_col, str):
        raise ValueError('trade_history_col must be a string')
    
    TRADE_SEQUENCE_LENGTH = train_dataframe[trade_history_col][0].shape[0] 
    
    params = {'TRADE_SEQUENCE_LENGTH':TRADE_SEQUENCE_LENGTH}
    
    np.random.seed(1)
    val_idx = np.random.choice(range(len(train_dataframe)), 
                     size = int(VALIDATION_SPLIT*len(train_dataframe)),
                     replace=False)

    print(f'TRAINING DATA: N = {len(train_dataframe)-len(val_idx)}, MIN DATE = {train_dataframe.drop(val_idx, axis=0).trade_date.min()}, MAX DATE = {train_dataframe.drop(val_idx, axis=0).trade_date.max()}')
    print(f'VALIDATION DATA: N = {len(val_idx)}, MIN DATE = {train_dataframe.iloc[val_idx].trade_date.min()}, MAX DATE = {train_dataframe.iloc[val_idx].trade_date.max()}')
    print(f'TEST DATA: N = {len(test_dataframe)}, MIN DATE = {test_dataframe.trade_date.min()}, MAX DATE = {test_dataframe.trade_date.max()}')

    x_train = create_input(train_dataframe.drop(val_idx, axis=0), trade_history_col)
    y_train = train_dataframe.drop(val_idx, axis=0)[target_variable]

    x_val = create_input(train_dataframe.iloc[val_idx], trade_history_col)
    y_val = train_dataframe.iloc[val_idx][target_variable]

    x_test = create_input(test_dataframe, trade_history_col)
    y_test = test_dataframe[target_variable]    
    
    with tf.device('/cpu:0'):
        # Normalization layer for the trade history
        trade_history_normalizer = Normalization(name='Trade_history_normalizer')
        trade_history_normalizer.adapt(x_train[0],batch_size=BATCH_SIZE)

        # Normalization layer for the non-categorical and binary features
        noncat_binary_normalizer = Normalization(name='Numerical_binary_normalizer')
        noncat_binary_normalizer.adapt(x_train[2], batch_size = BATCH_SIZE)

    normalizers = {'trade_history_normalizer': trade_history_normalizer,
                  'noncat_binary_normalizer': noncat_binary_normalizer}

    return  params, normalizers, x_train, y_train, x_val, y_val, x_test, y_test, val_idx


def create_tf_data(x_train, y_train, shuffle=False, shuffle_buffer=1):
                     
    X=()
    for x in x_train:
        X += (tf.data.Dataset.from_tensor_slices(x),)
        

    temp = tf.data.Dataset.zip((X))
    del X
    dataset = tf.data.Dataset.zip((temp,
                        tf.data.Dataset.from_tensor_slices(y_train)))
    del temp
    if shuffle:
        shuffle_buffer = int(len(x_train[0])*shuffle_buffer)
        dataset = dataset.shuffle(shuffle_buffer)
        
    return dataset

def train_model(params, normalizers, x_train, y_train, x_val, y_val, shuffle, shuffle_buffer=1):

    TRADE_SEQUENCE_LENGTH = params.get('TRADE_SEQUENCE_LENGTH')
    trade_history_normalizer = normalizers.get('trade_history_normalizer')
    noncat_binary_normalizer = normalizers.get('noncat_binary_normalizer')
       
    tf.keras.utils.set_random_seed(10)
    
#     if model_to_use == 'default':
#         model = generate_model_default(TRADE_SEQUENCE_LENGTH,trade_history_normalizer, noncat_binary_normalizer)
        
#     elif model_to_use=='bottleneck': 
#         model = generate_model_bottleneck(TRADE_SEQUENCE_LENGTH, trade_history_normalizer, noncat_binary_normalizer)
        
#     elif model_to_use=='ensemble': 
#         model = generate_model_ensemble(TRADE_SEQUENCE_LENGTH, trade_history_normalizer, noncat_binary_normalizer, ensemble_size)
    
#     else:
#         raise ValueError(f'Invalid model specified, {model_to_use}')
        
    model = generate_model_default(TRADE_SEQUENCE_LENGTH,trade_history_normalizer, noncat_binary_normalizer)
    
    
    fit_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=0,
        mode="auto",
        restore_best_weights=True),
        ]
    
    with tf.device('/cpu:0'):
        train_ds = create_tf_data(x_train, y_train, shuffle, shuffle_buffer)
        train_ds = train_ds.batch(BATCH_SIZE).prefetch(2).cache()
        val_ds = create_tf_data(x_val, y_val, shuffle = False)
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(2).cache()

    history= model.fit(train_ds,
                                      validation_data=val_ds,
                                        epochs=NUM_EPOCHS,     
                                        verbose=1, 
                                        callbacks=fit_callbacks,
                                        use_multiprocessing=True,
                                        workers=8)
    
    return history, model

def train_model_func(df):
    # df = pd.read_pickle(path)
    
    train_filter = (df.trade_date < train_end) & (df.trade_date >= train_start)
    test_filter = (df.trade_date >= test_start) & (df.trade_date <test_end)

    train_dataframe = df[train_filter]\
    .sort_values(by='trade_date', ascending=True)\
    .reset_index(drop=True).copy()

    test_dataframe = df[test_filter]\
    .sort_values(by='trade_date', ascending=True)\
    .reset_index(drop=True).copy()

    train_dataframe['last_seconds_ago'] = train_dataframe['last_seconds_ago'].fillna(0)
    train_dataframe['last_yield_spread'] = train_dataframe['last_yield_spread'].fillna(0)

    test_dataframe['last_seconds_ago'] = test_dataframe['last_seconds_ago'].fillna(0)
    test_dataframe['last_yield_spread'] = test_dataframe['last_yield_spread'].fillna(0)
    
    reset_model_features()
    params, normalizers, x_train, y_train, x_val, y_val, x_test, y_test, val_idx = create_data_set_and_model(train_dataframe, 
                                                                                                             test_dataframe, 
                                                                                                             trade_history_col)
    
    history, model = train_model(params, normalizers, x_train, y_train, x_val, y_val, shuffle, shuffle_buffer=1)
    
    return history, model


def val_model_func(df, model):
    train_filter = (df.trade_date < train_end) & (df.trade_date >= train_start)
    test_filter = (df.trade_date >= test_start) & (df.trade_date <test_end)

    train_dataframe = df[train_filter]\
    .sort_values(by='trade_date', ascending=True)\
    .reset_index(drop=True).copy()

    test_dataframe = df[test_filter]\
    .sort_values(by='trade_date', ascending=True)\
    .reset_index(drop=True).copy()

    train_dataframe['last_seconds_ago'] = train_dataframe['last_seconds_ago'].fillna(0)
    train_dataframe['last_yield_spread'] = train_dataframe['last_yield_spread'].fillna(0)

    test_dataframe['last_seconds_ago'] = test_dataframe['last_seconds_ago'].fillna(0)
    test_dataframe['last_yield_spread'] = test_dataframe['last_yield_spread'].fillna(0)
    
    reset_model_features()
    params, normalizers, x_train, y_train, x_val, y_val, x_test, y_test, val_idx = create_data_set_and_model(train_dataframe, 
                                                                                                             test_dataframe, 
                                                                                                             trade_history_col)
    
    preds = model.predict(x_test)
    
    return mean_absolute_error(preds, y_test)
