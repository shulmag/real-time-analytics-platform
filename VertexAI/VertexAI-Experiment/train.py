#Last Updated 28/12/2023

import os
import pandas as pd
import time
import numpy as np
import gcsfs
import pickle5 as pickle
import argparse
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import json
from helper_functions import *
from models import *
import globals
import hypertune


fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
TARGET = 'new_ys' 
trade_history_col = 'trade_history'
attention_col = 'target_attention_features'
dformat = '%Y-%m-%d'
hpt = hypertune.HyperTune()


def get_args():
    '''This function parses arguments to the train.py script from the command line.
    
    When Vertex AI runs a container instance with user-specified hyperparameters, it automatically passes those arguments to the 
    container instance. The container instance then passes these arguments to the script run at ENTRYPOINT in the docker file, which 
    in this case is train.py.
    '''
    
    parser = argparse.ArgumentParser(description='Description of your script')
    
    #MODEL HYPERPARAMETER ARGUMENTS 
    parser.add_argument('--LEARNING_RATE', default = 0.0001, type=float)
    parser.add_argument('--VALIDATION_SPLIT', default = 0.15, type=float)
    parser.add_argument('--BATCH_SIZE', default = 1000, type=int)
    parser.add_argument('--NUM_EPOCHS', default = 5, type=int)
    parser.add_argument('--DROPOUT', default = 0.1, type=float)
    
    #DATA ARGUMENTS 
    parser.add_argument('--data_bucket', help='Storage bucket for training data', default = 'ficc-pipelines-test', type=str)
    parser.add_argument('--file', help='File name for training data', default = 'similar_trades_df.pkl', type=str)
    parser.add_argument('--verbose', '-v', help='Option to print arguments', type=bool, default = True)
    parser.add_argument('--train_start', help='Start of training window, inclusive', type=str)
    parser.add_argument('--train_end', help='End of training window, inclusive', type=str)
    parser.add_argument('--test_start', help='Start of test window, inclusive', type=str)
    parser.add_argument('--test_end', help='End of test window, inclusive', type=str) 
    parser.add_argument('--experiment_name', help='Name of folder to store results in gs://ficc-model-experiments', type=str, default = '')
    parser.add_argument('--experiment_number', help='Experiment ID, used to run multiple experiments', type=int, default=1)
    
    #MODEL ARCHITECTURE ARGUMENTS 
    parser.add_argument('--model', help='If running default experiment, can be default, bottleneck or ensemble',type=str, default = 'default')
    parser.add_argument('--ensemble_size', help='If running default experiment, ensemble size for model',type=int, default = 0)
    parser.add_argument('--custom_model_path', help='If running custom experiment, file path to customer model and model_features.json. Overrides --model and --ensemble_size arguments if provided.', type=str)
    
    #convert args to dictionary object for convenient reference later 
    args = vars(parser.parse_args())
    
    if args['verbose']:
        print(f'Running experiment with the following parameters: {args}')
        
    return args

#MODEL AND DATA FUNCTIONS
def create_input_new(df, trade_history_col):
    '''Creates input for neural network'''
    datalist = []
        
    for col in globals.ADDITIONAL_SEQUENCES:
        datalist.append(np.stack(df[col].to_numpy()))
        
    datalist.append(np.stack(df[trade_history_col].to_numpy()))
    datalist.append(np.stack(df[attention_col].to_numpy()))
    
    noncat_and_binary = []
    for f in globals.NON_CATEGORICAL_FEATURES + globals.BINARY_FEATURES:
        noncat_and_binary.append(np.expand_dims(df[f].to_numpy().astype('float32'), axis=1))
    datalist.append(np.concatenate(noncat_and_binary, axis=-1))
    
    for f in globals.CATEGORICAL_FEATURES:
        encoded = encoders[f].transform(df[f])
        datalist.append(encoded.astype('float32'))
    
    return datalist

def create_dataset(train_dataframe, test_dataframe, trade_history_col):
    '''Processes train and test dataframes and fits normalizers for use by neural network.
    
    Note that if a custom model is being run, the normalizers are still being fit but are not used downstream.
    '''
    
    if not isinstance(trade_history_col, str):
        raise ValueError('trade_history_col must be a string')
    
    np.random.seed(1)
    val_idx = np.random.choice(range(len(train_dataframe)), 
                     size = int(globals.VALIDATION_SPLIT*len(train_dataframe)),
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
        trade_history_normalizer.adapt(x_train[0],batch_size=globals.BATCH_SIZE)

        # Normalization layer for the non-categorical and binary features
        noncat_binary_normalizer = Normalization(name='Numerical_binary_normalizer')
        noncat_binary_normalizer.adapt(x_train[2], batch_size = globals.BATCH_SIZE)

    normalizers = {'trade_history_normalizer': trade_history_normalizer,
                  'noncat_binary_normalizer': noncat_binary_normalizer}

    return  normalizers, x_train, y_train, x_val, y_val, x_test, y_test, val_idx


def create_tf_data(x_train, y_train, shuffle=False, shuffle_buffer=1):
    '''Converts input data for model to tf.Dataset'''
                     
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

def train_model_default(normalizers, x_train, y_train, x_val, y_val, shuffle, shuffle_buffer=1):
    '''Function for training a model with default architecture.'''

    trade_history_normalizer = normalizers.get('trade_history_normalizer')
    noncat_binary_normalizer = normalizers.get('noncat_binary_normalizer')
       
    tf.keras.utils.set_random_seed(10)
    
    if model_to_use == 'default':
        model = generate_model_default(trade_history_normalizer, noncat_binary_normalizer)
        
    elif model_to_use=='bottleneck': 
        model = generate_model_bottleneck( trade_history_normalizer, noncat_binary_normalizer)
        
    elif model_to_use=='ensemble': 
        model = generate_model_ensemble( trade_history_normalizer, noncat_binary_normalizer, ensemble_size)
    
    else:
        raise ValueError(f'Invalid model specified, {model_to_use}')
        
        
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
        train_ds = train_ds.batch(globals.BATCH_SIZE).prefetch(2).cache()
        val_ds = create_tf_data(x_val, y_val, shuffle = False)
        val_ds = val_ds.batch(globals.BATCH_SIZE).prefetch(2).cache()

    history= model.fit(train_ds,
                                      validation_data=val_ds,
                                        epochs=globals.NUM_EPOCHS,     
                                        verbose=1, 
                                        callbacks=fit_callbacks,
                                        use_multiprocessing=True,
                                        workers=8)
    
    return history, model

def train_model_custom(model, x_train, y_train, x_val, y_val, shuffle, shuffle_buffer=1):
    '''Function for training a custom model. Unlike train_model_default, model is predefined and is not instantiated here.'''
    
    #Reintializes weights for the model. Because we are using a model template the initial weights would otherwise be the same across all runs.
    reinitialize_model(model)
    
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
        train_ds = train_ds.batch(globals.BATCH_SIZE).prefetch(2).cache()
        val_ds = create_tf_data(x_val, y_val, shuffle = False)
        val_ds = val_ds.batch(globals.BATCH_SIZE).prefetch(2).cache()

    history= model.fit(train_ds,
                                      validation_data=val_ds,
                                        epochs=globals.NUM_EPOCHS,     
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
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    
    #Parse experiment parameters 
    args = get_args()
    train_start = args.get('train_start', None)
    train_end = args.get('train_end', None)
    test_start = args.get('test_start', None)
    test_end = args.get('test_end', None)
    
    if not all([train_start, train_end, test_start, test_end]):
        raise ValueError('Did not receive valid dates for train start, train_end, test_start, test_end')
    
    globals.LEARNING_RATE = args.get('LEARNING_RATE', 0.0007) 
    globals.VALIDATION_SPLIT = args.get('VALIDATION_SPLIT', 0.1) 
    globals.BATCH_SIZE = args.get('BATCH_SIZE', 10000)
    globals.NUM_EPOCHS = args.get('NUM_EPOCHS', 150)
    globals.DROPOUT = args.get('DROPOUT', 0.2)
    
    data_bucket = args.get('data_bucket', 'custom-train-job-test')
    file = args.get('file', 'data.pkl')
    model_to_use = args.get('model', 'default')
    ensemble_size = args.get('ensemble_size', None)
    experiment_number = args.get('experiment_number', 1)
    experiment_name = args.get('experiment_name', None)
    
    #If experiment name is not specified, place results in a default folder
    if not experiment_name: 
        experiment_name = 'unlabeled_experiments'
        
    custom_model_path = args.get('custom_model_path', '')
    
    if custom_model_path:
        #If we have defined a custom model, we must load the model template and model features at custom_model_path
        #When using a custom model, we must have already built and compiled (but not yet fitted) the model. 
        #This means that the model has to take inputs in a specified order. This is precisely the purpose of model_features.json.
        #model_features.json provides the ordering of binary, categorical, non-categorical and additional features 
        
        custom_experiment = True
        
        with fs.open(os.path.join(custom_model_path,'model_features.json')) as f:
            features = json.load(f)
            
        globals.BINARY_FEATURES = features.get('BINARY_FEATURES', None)
        globals.CATEGORICAL_FEATURES = features.get('CATEGORICAL_FEATURES', None)
        globals.NON_CATEGORICAL_FEATURES = features.get('NON_CATEGORICAL_FEATURES', None)
        globals.ADDITIONAL_SEQUENCES = features.get('ADDITIONAL_SEQUENCES', []) 
        #additional sequences allows for the model to have additional sequence components, such as LSTM for similar trades
        
        if not globals.BINARY_FEATURES or not globals.CATEGORICAL_FEATURES or not globals.NON_CATEGORICAL_FEATURES:
            raise ValueError(f'Custom job specified but features were invalid! Loaded parameters.json from {custom_model_path} and received the following: {features}')
            
        model = keras.models.load_model(os.path.join(custom_model_path,'model'))
    
    else: 
        #If we have not specified a custom model, then the script defaults to using the default models specified in model.py/
        
        custom_experiment = False
    
        if model_to_use not in ['default', 'ensemble', 'bottleneck']:
            print(f'Model to use [{model_to_use}] is invalid. It should be one of "default", "ensemble" or "bottleneck". Defaulting to "default".')
            model_to_use = 'default'

        if model_to_use == 'ensemble' and not ensemble_size:
            print(f'Model to use "{model_to_use}" but ensemble_size not specified. Defaulting to ensemble_size=2.')
            ensemble_size = 2

    args.update({'BINARY': globals.BINARY_FEATURES})
    args.update({'CATEGORICAL_FEATURES': globals.CATEGORICAL_FEATURES})
    args.update({'NON_CAT_FEATURES': globals.NON_CATEGORICAL_FEATURES})
    args.update({'ADDITIONAL_SEQUENCES': globals.ADDITIONAL_SEQUENCES})
    
    #Load and process data
    processed_data = load_data_from_pickle(file, bucket = data_bucket)    
    print(f'Entire dataset min date: {processed_data.trade_date.min()}, Max date: {processed_data.trade_date.max()}')
    train_filter = (processed_data.trade_date <= train_end) & (processed_data.trade_date >= train_start)
    test_filter = (processed_data.trade_date <= test_end) & (processed_data.trade_date >= test_start)
    
    #Handle errors if training or testing data is empty
    if sum(train_filter) == 0:
        report_mae_error('mae')
        report_mae_error('mae_with_exclusions')
        print(f'No train data for range {train_start} to {train_end}, exiting function.')
        exit()
    
    if sum(test_filter) == 0:
        report_mae_error('mae')
        report_mae_error('mae_with_exclusions')
        print(f'No train data for range {test_start} to {test_end}, exiting function.')
        exit()
        
    #Split training and testing data
    train_dataframe = processed_data[train_filter].sort_values(by='trade_date', ascending=True).reset_index(drop=True)
    test_dataframe = processed_data[test_filter].sort_values(by='trade_date', ascending=True).reset_index(drop=True)
    train_dataframe['last_seconds_ago'] = train_dataframe['last_seconds_ago'].fillna(0)
    train_dataframe['last_yield_spread'] = train_dataframe['last_yield_spread'].fillna(0)
    test_dataframe['last_seconds_ago'] = test_dataframe['last_seconds_ago'].fillna(0)
    test_dataframe['last_yield_spread'] = test_dataframe['last_yield_spread'].fillna(0)
    print('Test data start: {}, end: {}'.format(test_dataframe.trade_date.min(),test_dataframe.trade_date.max()))
    

    #Prepare data for model inputs 
    normalizers, x_train, y_train, x_val, y_val, x_test, y_test, val_idx = create_dataset(train_dataframe, test_dataframe, trade_history_col)

    if custom_experiment:
        history, model = train_model_custom(model, x_train, y_train, x_val, y_val, True, shuffle_buffer = .75)
    else:
        history, model = train_model_default(normalizers, x_train, y_train, x_val, y_val, True, shuffle_buffer = .75)
        
    pred = model.predict(x_test, batch_size=10000)
    
    if model_to_use == 'ensemble':
        pred = np.concatenate(pred, axis=1).mean(axis=1)
    
    test_dataframe['new_ys_prediction'] = pred
    
    #Stop gap measure in case we get nonsensical predictions that skew mae. Happened in July backtesting. 
    pred_filter = (-5000 < test_dataframe['new_ys_prediction']) & (test_dataframe['new_ys_prediction'] < 5000)
    if len(test_dataframe) - sum(pred_filter):
        print(f'There are {len(test_dataframe) - sum(pred_filter)} trades with extreme outlier predictions beyond +-5000bps. Removing them for MAE calculation. Check saved predictions for those specific trades.')
    
    test_dataframe = addcol(test_dataframe, 'cases', mkcases(test_dataframe))
    short_filter = ((test_dataframe.days_to_call == 0) | (test_dataframe.days_to_call > np.log10(400))) & \
                ((test_dataframe.days_to_maturity == 0) | (test_dataframe.days_to_maturity > np.log10(400))) & \
                ((test_dataframe.days_to_refund == 0) | (test_dataframe.days_to_refund > np.log10(400))) & \
                (test_dataframe.days_to_maturity < np.log10(30000))  
    
    #Evaluate model results 
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

    print(args)
    
    #Upload model and predictions
    base_path = f'gs://ficc-model-experiments/{experiment_name}/run_{experiment_number}'
    model_path = os.path.join(base_path, 'model')
    model.save(model_path)
    print(f'Model saved to {model_path}.')
    args_path =  os.path.join(base_path, 'args.json')
    with fs.open(args_path, 'w') as gf:
        json.dump(args, gf)
        print(f'Model args saved to {args_path}.')
    
    save_cols = ['rtrs_control_number', 'new_ys_prediction', 'prediction_datetime']
    test_dataframe['prediction_datetime'] = pd.to_datetime(timestamp)
    
    predictions_path = os.path.join(base_path, 'predictions.pkl')
    test_dataframe[save_cols].to_pickle(predictions_path)
    print(f'Predictions saved to {predictions_path}')
    
    print('Experiment completed. Exiting.')
    
    
    
    