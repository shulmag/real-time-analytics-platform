#Last Updated 29/9/2023

import os
import pandas as pd
import time
import numpy as np
import gcsfs
import pickle5 as pickle
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from datetime import datetime

from auxiliary_variables import PREDICTORS, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES, IDENTIFIERS, PURPOSE_CLASS_DICT, NUM_OF_DAYS_IN_YEAR
from helper_functions import *
import hypertune

#Define global variables 
fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
TRADE_SEQUENCE_LENGTH = 5
NUM_FEATURES = 6
TARGET = 'new_ys' 
trade_history_col = 'trade_history'
dformat = '%Y-%m-%d'
hpt = hypertune.HyperTune()
with fs.open('gs://automated_training/encoders.pkl') as f:
    encoders = pickle.load(f)
fmax = {key: len(value.classes_) for key, value in encoders.items()}

def load_data_from_pickle(path, bucket = 'isaac_data'):
    if os.path.isfile(path):
        print('File available, loading pickle')
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        print(f'File not available, downloading from cloud storage and saving to {path}')
        gc_path = os.path.join(bucket, path)
        print(gc_path)
        with fs.open(gc_path) as gf:
            data = pd.read_pickle(gf)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    return data

def get_args():
    '''This function parses arguments to the python script'''
    
    parser = argparse.ArgumentParser(description='Description of your script')
    
    #MODEL HYPERPARAMETERS ARGUMENTS 
    parser.add_argument('--LEARNING_RATE', default = 0.0001, type=float)
    parser.add_argument('--VALIDATION_SPLIT', default = 0.15, type=float)
    parser.add_argument('--BATCH_SIZE', default = 1000, type=int)
    parser.add_argument('--NUM_EPOCHS', default = 5, type=int)
    parser.add_argument('--DROPOUT', default = 0.1, type=float)
    
    #DATA ARGUMENTS 
    parser.add_argument('--bucket', help='Storage bucket for training data', default = 'custom-train-job-test', type=str)
    parser.add_argument('--file', help='File name for training data', default = 'data.pkl', type=str)
    parser.add_argument('--verbose', '-v', type=bool, default = True)
    parser.add_argument('--target_date', help='Test date', type=str)
    parser.add_argument('--train_months', help='Maximimum number of months of training data', type=int, default = 6)
    
    #MODEL ARCHITECTURE  ARGUMENTS 
    parser.add_argument('--model', type=str, default = 'default')
    parser.add_argument('--ensemble_size', type=int, default = 0)
    
    #convert args to dict
    args = vars(parser.parse_args())
    
    if args['verbose']:
        print(f'Running experiment with the following parameters: {args}')
    return args

#MODEL AND DATA FUNCTIONS
def create_input_new(df, trade_history_col):
    global encoders
    datalist = []
        
    datalist.append(np.stack(df[trade_history_col].to_numpy()))
    datalist.append(np.stack(df['target_attention_features'].to_numpy()))

    noncat_and_binary = []
    for f in NON_CAT_FEATURES + BINARY:
        noncat_and_binary.append(np.expand_dims(df[f].to_numpy().astype('float32'), axis=1))
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

    x_train = create_input_new(train_dataframe.drop(val_idx, axis=0), trade_history_col)
    y_train = train_dataframe.drop(val_idx, axis=0)[TARGET]

    x_val = create_input_new(train_dataframe.iloc[val_idx], trade_history_col)
    y_val = train_dataframe.iloc[val_idx][TARGET]

    x_test = create_input_new(test_dataframe, trade_history_col)
    y_test = test_dataframe[TARGET]    
    
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

def train_model_new(params, normalizers, x_train, y_train, x_val, y_val, shuffle, shuffle_buffer=1):

    TRADE_SEQUENCE_LENGTH = params.get('TRADE_SEQUENCE_LENGTH')
    trade_history_normalizer = normalizers.get('trade_history_normalizer')
    noncat_binary_normalizer = normalizers.get('noncat_binary_normalizer')
       
    tf.keras.utils.set_random_seed(10)
    
    if model_to_use == 'default':
        model = generate_model_default(TRADE_SEQUENCE_LENGTH,trade_history_normalizer, noncat_binary_normalizer)
        
    elif model_to_use=='bottleneck': 
        model = generate_model_bottleneck(TRADE_SEQUENCE_LENGTH, trade_history_normalizer, noncat_binary_normalizer)
        
    elif model_to_use=='ensemble': 
        model = generate_model_ensemble(TRADE_SEQUENCE_LENGTH, trade_history_normalizer, noncat_binary_normalizer, ensemble_size)
    
    else:
        raise ValueError(f'Invalid model specified, {model_to_use}')
        
        
    timestamp = datetime.now().strftime('%Y-%m-%d %H-%M')
    
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

def report_mae_error(hyperparameter_metric_tag):
    '''Reports MAE=9999 to Vertex AI in the event of an error. This is to avoid Vertex AI cancelling the job due to invalid mae.'''
    hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag=hyperparameter_metric_tag,
                                                metric_value=9999,
                                                global_step=0)

def report_mae(metric_value, hyperparameter_metric_tag, global_step):
    '''Reports model MAE after training'''
    hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag=hyperparameter_metric_tag,
                                                metric_value=metric_value,
                                                global_step=global_step)
    
if __name__=="__main__":
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')
    
    ### PARSE ARGUMENTS FOR EXPERIMENT ### 
    args = get_args()    
    LEARNING_RATE = args.get('LEARNING_RATE', 0.0001) 
    VALIDATION_SPLIT = args.get('VALIDATION_SPLIT', 0.1) 
    BATCH_SIZE = args.get('BATCH_SIZE', 1000)
    NUM_EPOCHS = args.get('NUM_EPOCHS', 25)
    DROPOUT = args.get('DROPOUT', 0.1)
    bucket = args.get('bucket', 'custom-train-job-test')
    file = args.get('file', 'data.pkl')
    model_to_use = args.get('model', 'default')
    ensemble_size = args.get('ensemble_size', None)
    train_months = args.get('train_months', 6)
    target_date = args.get('target_date', None)
    if not target_date:
        report_mae_error('mae')
        report_mae_error('mae_with_exclusions')
        print('No target_date specified, exiting function')
        exit()
    train_start = (pd.to_datetime(target_date) - pd.DateOffset(months=train_months)).strftime(dformat)
    
    ### SELECT MODEL TO USE ### 
    if model_to_use not in ['default', 'ensemble', 'bottleneck']:
        print(f'Model to use [{model_to_use}] is invalid. It should be one of "default", "ensemble" or "bottleneck". Defaulting to "default".')
        model_to_use = 'default'
        
    if model_to_use == 'ensemble' and not ensemble_size:
        print(f'Model to use "{model_to_use}" but ensemble_size not specified. Defaulting to ensemble_size=2.')
        ensemble_size = 2
    
    ### LOAD AND PROCESS DATA ### 
    processed_data = load_data_from_pickle(file, bucket = bucket)    
    print(f'Entire dataset min date: {processed_data.trade_date.min()}, Max date: {processed_data.trade_date.max()}')
    train_filter = (processed_data.trade_date < target_date) & (processed_data.trade_date >= train_start)
    test_filter = (processed_data.trade_date == target_date)
    if sum(test_filter) == 0:
        report_mae_error('mae')
        report_mae_error('mae_with_exclusions')
        print(f'No test data for {target_date}, exiting function.')
        exit()
    train_dataframe = processed_data[train_filter].sort_values(by='trade_date', ascending=True).reset_index(drop=True)
    test_dataframe = processed_data[test_filter].sort_values(by='trade_date', ascending=True).reset_index(drop=True)
    train_dataframe['last_seconds_ago'] = train_dataframe['last_seconds_ago'].fillna(0)
    train_dataframe['last_yield_spread'] = train_dataframe['last_yield_spread'].fillna(0)
    test_dataframe['last_seconds_ago'] = test_dataframe['last_seconds_ago'].fillna(0)
    test_dataframe['last_yield_spread'] = test_dataframe['last_yield_spread'].fillna(0)
    print('Test data start: {}, end: {}'.format(test_dataframe.trade_date.min(),test_dataframe.trade_date.max()))
    

    ### PREPARE MODEL INPUT ###
    params, normalizers, x_train, y_train, x_val, y_val, x_test, y_test, val_idx = create_data_set_and_model(train_dataframe, test_dataframe, trade_history_col)

    history, model = train_model_new(params, normalizers, x_train, y_train, x_val, y_val, True, shuffle_buffer = .75)
    pred = model.predict(x_test, batch_size=5000)
    
    if model_to_use == 'ensemble':
        pred = np.concatenate(pred, axis=1).mean(axis=1)
    
    test_dataframe['new_ys_prediction'] = pred
    
    #Stop gap measure incase we get nonsensical predictions that skew mae. Happened in July. 
    pred_filter = (-5000 < test_dataframe['new_ys_prediction']) & (test_dataframe['new_ys_prediction'] < 5000)
    if len(test_dataframe) - sum(pred_filter):
        print(f'There are {len(test_dataframe) - sum(pred_filter)} trades with extreme outlier predictions beyond +-5000bps. Removing them for MAE calculation. Check saved predictions for those specific trades.')
    
    test_dataframe = addcol(test_dataframe, 'cases', mkcases(test_dataframe))
    short_filter = ((test_dataframe.days_to_call == 0) | (test_dataframe.days_to_call > np.log10(400))) & \
                ((test_dataframe.days_to_maturity == 0) | (test_dataframe.days_to_maturity > np.log10(400))) & \
                ((test_dataframe.days_to_refund == 0) | (test_dataframe.days_to_refund > np.log10(400))) & \
                (test_dataframe.days_to_maturity < np.log10(30000))  
    
    ### EVALUATE MODEL RESULTS ###
    mae = mean_absolute_error(test_dataframe[pred_filter]['new_ys_prediction'], 
                              test_dataframe[pred_filter][TARGET])
                           
    mae_all_exclusions = mean_absolute_error(test_dataframe[pred_filter & short_filter]['new_ys_prediction'], 
                                             test_dataframe[pred_filter& short_filter][TARGET])
                           
    print('='*25+f' OVERALL MAE: {mae:.2f} '+'='*25)
    print('='*25+f' MAE EXCLUDING SHORT MATURITY: {mae_all_exclusions:.2f} '+'='*25)                       
                           
    report_mae(mae, 'mae', len(history.history['loss']))
    report_mae(mae_all_exclusions, 'mae_with_exclusions', len(history.history['loss']))
                           
                           
    summary = compare_mae(df=test_dataframe, 
                          prediction_cols = ['new_ys_prediction'], 
                          groupby_cols = None,
                      target_variable=TARGET)
    display(summary)

    summary = compare_mae(df=test_dataframe[short_filter], 
                              prediction_cols = ['new_ys_prediction'],   
                              groupby_cols = None,
                          target_variable=TARGET)
    display(summary)

    ### UPLOAD MODEL AND PREDICTIONS ###
    base_path = f'gs://ficc-historical-results/{target_date}/{timestamp}'
    model_path = os.path.join(base_path, 'model')
    model.save(model_path)
    print(f'Model saved to {model_path}.')
    args_path =  os.path.join(base_path, 'args.pkl')
    with fs.open(args_path, 'wb') as gf:
        pickle.dump(args, gf)
        print(f'Model args saved to {args_path}.')
    
    save_cols = ['rtrs_control_number', 'cusip', 'trade_date', 'dollar_price', 'yield', 'new_ficc_ycl', 'new_ys', 'new_ys_prediction', 'prediction_datetime']
    test_dataframe = test_dataframe.rename({'new_real_time_ficc_ycl':'new_ficc_ycl'}, axis=1)
    test_dataframe['prediction_datetime'] = pd.to_datetime(timestamp)
    
    predictions_path = os.path.join(base_path, 'predictions.pkl')
    test_dataframe[save_cols].to_pickle(predictions_path)
    print(f'Predictions saved to {predictions_path}')
    uploadData(test_dataframe[save_cols],
            "eng-reactor-287421.historical_predictions_test.historical_predictions_test",
            getSchema())
    
    print('Training completed. Exiting.')
    
    
    
    