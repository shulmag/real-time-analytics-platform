# -*- coding: utf-8 -*-
# @Date:   2021-10-08 13:25:52

import pandas as pd
import numpy as np
from google.cloud import bigquery
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from sklearn import preprocessing
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
# from lightgbm import LGBMRegressor
# import lightgbm
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from IPython.display import display, HTML

from data_preparation import process_data
from yield_value import yield_curve_level
from ficc_calc_end_date import calc_end_date

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="eng-reactor-287421-112eb767e1b3.json"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

layer_initializer = initializers.RandomNormal(mean=0.0, stddev=0.1, seed=10)
bq_client = bigquery.Client()
pd.options.mode.chained_assignment = None


encoders=None
fmax=None
TRAIN_TEST_SPLIT = 0.85
LEARNING_RATE = 0.001
BATCH_SIZE = 1000
NUM_EPOCHS = 100
SEQUENCE_LENGTH = 5
NUM_FEATURES = 5
PROCESSED_FILE = 'tuning_data.pkl'

nelson_params = None
scalar_params = None


DATA_QUERY = """ SELECT
  *
FROM
  `eng-reactor-287421.primary_views.trade_history_for_training`
WHERE
  yield IS NOT NULL
  AND yield > 0 
  AND yield <= 3 
  AND par_traded IS NOT NULL
  AND sp_long IS NOT NULL
  AND trade_date >= '2021-07-01' 
  AND trade_date <= '2021-10-01'
  AND msrb_valid_to_date > current_date -- condition to remove cancelled trades
ORDER BY
  trade_date DESC
            """

COUPON_FREQUENCY_DICT = {0:"Unknown",
                        1:"Semiannually",
                        2:"Monthly",
                        3:"Annually",
                        4:"Weekly",
                        5:"Quarterly",
                        6:"Every 2 years",
                        7:"Every 3 years",
                        8:"Every 4 years",
                        9:"Every 5 years",
                        10:"Every 7 years",
                        11:"Every 8 years",
                        12:"Biweekly",
                        13:"Changeable",
                        14:"Daily",
                        15:"Term mode",
                        16:"Interest at maturity",
                        17:"Bimonthly",
                        18:"Every 13 weeks",
                        19:"Irregular",
                        20:"Every 28 days",
                        21:"Every 35 days",
                        22:"Every 26 weeks",
                        23:"Not Applicable",
                        24:"Tied to prime",
                        25:"One time",
                        26:"Every 10 years",
                        27:"Frequency to be determined",
                        28:"Mandatory put",
                        29:"Every 52 weeks",
                        30:"When interest adjusts-commercial paper",
                        31:"Zero coupon",
                        32:"Certain years only",
                        33:"Under certain circumstances",
                        34:"Every 15 years",
                        35:"Custom",
                        36:"Single Interest Payment"
                        }

IDENTIFIERS = ['rtrs_control_number', 'cusip']


BINARY = ['callable',
          'sinking',
          'zerocoupon',
          'is_non_transaction_based_compensation',
          'is_general_obligation',
          'callable_at_cav',           
          'extraordinary_make_whole_call', 
          'make_whole_call',
          'has_unexpired_lines_of_credit',
          'escrow_exists',
          ]

CATEGORICAL_FEATURES = ['rating',
                        'incorporated_state_code',
                        'trade_type',
                        'transaction_type',
                        'maturity_description_code',
                        'purpose_class']

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
                    'max_amount_outstanding']

TRADE_HISTORY = ['trade_history']
TARGET = ['yield_spread']

PREDICTORS = BINARY + CATEGORICAL_FEATURES + NON_CAT_FEATURES + TARGET + TRADE_HISTORY

def sqltodf(sql,bq_client):
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()

def yield_curve_params(client):
    global nelson_params
    global scalar_params
    #The following fetches Nelson-Siegel coefficient and standard scalar parameters from BigQuery and sends them to a dataframe. 
    nelson_params = sqltodf("select * from `eng-reactor-287421.yield_curves.nelson_siegel_coef_daily` order by date desc",client)
    scalar_params = sqltodf("select * from`eng-reactor-287421.yield_curves.standardscaler_parameters_daily` order by date desc",client)


    #The below sets the index of both dataframes to date column and converts the data type to datetime. 
    nelson_params.set_index("date",drop=True,inplace=True)
    scalar_params.set_index("date",drop=True,inplace=True)
    scalar_params.index = pd.to_datetime(scalar_params.index)
    nelson_params.index = pd.to_datetime(nelson_params.index)

def get_ficc_ycl(trade):
    global nelson_params
    global scalar_params
    target_date = None

    if trade.trade_date < datetime(2021,7,27).date():
        target_date = datetime(2021,7,27).date()
    else:
        target_date = trade.trade_date
    duration = (trade.calc_date - target_date).days/365.25
    ficc_yl = yield_curve_level(duration,
                                target_date.strftime('%Y-%m-%d'),
                                nelson_params, 
                                scalar_params)
    return ficc_yl

def get_latest_trade_feature(x, feature):
    recent_trade = x[0]
    if feature == 'yield_spread':
        return recent_trade[0]
    elif feature == 'seconds_ago':
        return recent_trade[-1]
    elif feature == 'par_traded':
        return recent_trade[1]

def drop_extra_columns(df):
    df.drop(columns=[
                 'sp_stand_alone',
                 'sp_icr_school',
                 'sp_icr_school',
                 'sp_icr_school',
                 'sp_watch_long',
                 'sp_outlook_long',
                 'sp_prelim_long',
                 'MSRB_maturity_date',
                 'MSRB_INST_ORDR_DESC',
                 'MSRB_valid_from_date',
                 'MSRB_valid_to_date',
                 'upload_date',
                 'sequence_number',
                 'security_description',
                 'ref_valid_from_date',
                 'ref_valid_to_date',
                 'additional_next_sink_date',
                 'first_coupon_date',
                 'last_period_accrues_from_date',
                 'primary_market_settlement_date',
                 'assumed_settlement_date',
                 'sale_date','q','d'],
                  inplace=True)
    
    
    return df

def convert_dates(df):
    date_cols = [col for col in list(df.columns) if 'DATE' in col.upper()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    
    return df

def process_ratings(df):
    df = df[df.sp_long.isin(['A-','A','A+','AA-','AA','AA+','AAA','NR'])] 
    df['rating'] = df['sp_long']
    return df

                    
def get_latest_trade_feature(x, feature):
    recent_trade = x[0]
    if feature == 'yield_spread':
        return recent_trade[0]
    elif feature == 'seconds_ago':
        return recent_trade[-1]
    elif feature == 'par_traded':
        return recent_trade[1]   

def fill_missing_values(df):
    df.dropna(subset=['instrument_primary_name'], inplace=True)
    df.purpose_sub_class.fillna(1,inplace=True)
    df.call_timing.fillna(0, inplace=True) #Unknown
    df.call_timing_in_part.fillna(0, inplace=True) #Unknown
    df.sink_frequency.fillna(10, inplace=True) #Under special circumstances
    df.sink_amount_type.fillna(0, inplace=True)
    df.issue_text.fillna('No issue text', inplace=True)
    df.state_tax_status.fillna(0, inplace=True)
    df.series_name.fillna('No series name', inplace=True)

    df.next_call_price.fillna(100, inplace=True)
    df.par_call_price.fillna(100, inplace=True)
    df.min_amount_outstanding.fillna(0, inplace=True)
    df.max_amount_outstanding.fillna(0, inplace=True)
    # df.call_to_maturity.fillna(0, inplace=True)
    df.days_to_par.fillna(0, inplace=True)
    df.maturity_amount.fillna(0, inplace=True)
    df.issue_price.fillna(df.issue_price.mean(), inplace=True)
    df.orig_principal_amount.fillna(df.orig_principal_amount.mean(), inplace=True)
    df.original_yield.fillna(0, inplace=True)
    df.par_price.fillna(100, inplace=True)
    df.called_redemption_type.fillna(0, inplace=True)

    df.extraordinary_make_whole_call.fillna(False, inplace=True)
    df.make_whole_call.fillna(False, inplace=True)
    df.default_indicator.fillna(False, inplace=True)
    df.called_redemption_type.fillna(0, inplace=True)
    
    return df

        
def process_features(df):
    global COUPON_FREQUENCY_DICT
    df.interest_payment_frequency.fillna(0, inplace=True)
    df.loc[:,'interest_payment_frequency'] = df.interest_payment_frequency.apply(lambda x: COUPON_FREQUENCY_DICT[x])
    
    df.loc[:,'quantity'] = np.log10(df.par_traded.astype(np.float32))
    df.coupon = df.coupon.astype(np.float32)
    df.issue_amount = np.log10(df.issue_amount.astype(np.float32))
    #df['yield_spread'] = df['yield_spread'] * 100
    
    # Creating Binary features
    df.loc[:,'callable'] = df.is_callable  
    df.loc[:,'called'] = df.is_called 
    df.loc[:,'zerocoupon'] = df.coupon == 0
    df.loc[:,'whenissued'] = df.delivery_date >= df.trade_date
    df.loc[:,'sinking'] = ~df.next_sink_date.isnull()
    df.loc[:,'deferred'] = (df.interest_payment_frequency == 'Unknown') | df.zerocoupon
    
    # Converting the dates to a number of days from the settlement date. 
    # We only consider trades to be reportedly correctly if the trades are settled within one month of the trade date. 
    df.loc[:,'days_to_settle'] = (df.settlement_date - df.trade_date).dt.days.fillna(0)
    df = df[df.days_to_settle <= 31]

    df.loc[:, 'days_to_maturity'] =  np.log10(1 + (df.maturity_date - df.settlement_date).dt.days.fillna(0))
    df.loc[:, 'days_to_call'] = np.log10(1 + (df.next_call_date - df.settlement_date).dt.days.fillna(0))
    df.loc[:, 'days_to_refund'] = np.log10(1 + (df.refund_date - df.settlement_date).dt.days.fillna(0))
    df.loc[:, 'days_to_par'] = np.log10(1 + (df.par_call_date - df.settlement_date).dt.days.fillna(0))
    df.loc[:, 'call_to_maturity'] = np.log10(1 + (df.maturity_date - df.next_call_date).dt.days.fillna(0))
    
    # Removing bonds from Puerto Rico
    df = df[df.incorporated_state_code != 'PR']
    
    df.loc[:, 'last_seconds_ago'] = df.trade_history.apply(get_latest_trade_feature, args=["seconds_ago"])
    df.loc[:, 'last_yield_spread'] = df.trade_history.apply(get_latest_trade_feature, args=["yield_spread"])
    df.loc[:, 'last_size'] = df.trade_history.apply(get_latest_trade_feature, args=["par_traded"])


    df.maturity_amount = np.log10(1.0 + df.maturity_amount.astype(float))
    df.orig_principal_amount = np.log10(1.0 + df.orig_principal_amount.astype(float))
    df.max_amount_outstanding = np.log10(1.0 + df.max_amount_outstanding.astype(float))
    
    return df                 

def process_reference_data(df):
    df['ficc_ycl'] = df.parallel_apply(get_ficc_ycl,axis=1)
    df['yield_spread'] = df['yield'] * 100 - df['ficc_ycl']

    #Dropping extra columns from the data frame
    df = drop_extra_columns(df)

    #Converting dates to correct datatype
    df = convert_dates(df)

    #Processing the ratings of the bonds
    df = process_ratings(df)

    df = process_features(df)

    df = fill_missing_values(df)

    processed_data = df[IDENTIFIERS + PREDICTORS]
    processed_data.dropna(inplace=True)

    return processed_data

def create_input(df):
    global encoders
    datalist = []
    datalist.append(np.stack(df['trade_history'].to_numpy()))
    for f in NON_CAT_FEATURES + BINARY:
        datalist.append(df[f].to_numpy().astype('float32'))
        
    for f in CATEGORICAL_FEATURES:
        encoded = encoders[f].transform(df[f])
        datalist.append(encoded.astype('float32'))
    return datalist

def build_model(hp):
    inputs = []
    layer = []

    ############## INPUT BLOCK ###################
    trade_history_input = layers.Input(name="trade_history_input", 
                                       shape=(SEQUENCE_LENGTH,NUM_FEATURES), 
                                       dtype = tf.float32) 

    inputs.append(trade_history_input)

    for i in NON_CAT_FEATURES + BINARY:
        inputs.append(layers.Input(shape=(1,), name = f"{i}"))

    for i in inputs[1:]:
        layer.append(i)
    ####################################################


    ############## TRADE HISTORY MODEL #################

    # Adding the time2vec encoding to the input to transformer
    lstm_layer = layers.LSTM(hp.Int("lstm_layer_1_units", min_value=100, max_value=900, step=50), 
                             activation='tanh',
                             input_shape=(SEQUENCE_LENGTH,NUM_FEATURES),
                             kernel_initializer = layer_initializer,
                             return_sequences = True,
                             name='LSTM')

    lstm_layer_2 = layers.LSTM(hp.Int("lstm_layer_2_units", min_value=100, max_value=900, step=50), 
                               activation='tanh',
                               input_shape=(SEQUENCE_LENGTH,50),
                               kernel_initializer = layer_initializer,
                               return_sequences = False,
                               name='LSTM_2')

    features = lstm_layer(inputs[0])
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(hp.Choice("droupout_1", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]))(features)

    features = lstm_layer_2(features)
    features = layers.Dropout(hp.Choice("droupout_2", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]))(features)
    features = layers.BatchNormalization()(features)


    trade_history_output = layers.Dense(hp.Int("trade_history_output_layer", min_value=100, max_value=900, step=50), 
                                        activation='relu',
                                        kernel_initializer=layer_initializer)(features)

    ####################################################

    ############## REFERENCE DATA MODEL ################
    global encoders
    global fmax
    
    
    for f in CATEGORICAL_FEATURES:
        fin = layers.Input(shape=(1,), name = f)
        inputs.append(fin)
        embedded = layers.Flatten(name = f + "_flat")( layers.Embedding(input_dim = fmax[f]+1,
                                                                        output_dim = hp.Int("embedding_dim", min_value=1, max_value=100, step=5),
                                                                        input_length= 1,
                                                                        name = f + "_embed",
                                                                        embeddings_initializer=layer_initializer)(fin))
        layer.append(embedded)

    reference_hidden = layers.Dense(hp.Int("reference_hidden_1_units", min_value=100, max_value=900, step=50), 
                                    activation='relu',
                                    kernel_initializer=layer_initializer,
                                    name='reference_hidden_1')(layers.concatenate(layer))
    reference_hidden = layers.BatchNormalization()(reference_hidden)
    reference_hidden = layers.Dropout(hp.Choice("droupout_3", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]))(reference_hidden)
    
    reference_hidden2 = layers.Dense(hp.Int("reference_hidden_2_units", min_value=100, max_value=900, step=50), 
                                     activation='relu',
                                     kernel_initializer=layer_initializer,
                                     name='reference_hidden_2')(reference_hidden)
    reference_hidden2 = layers.BatchNormalization()(reference_hidden2)
    reference_hidden2 = layers.Dropout(hp.Choice("droupout_4", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]))(reference_hidden2)


    referenece_output = layers.Dense(hp.Int("reference_hidden_3_units", min_value=100, max_value=900, step=50), 
                                     activation='tanh',
                                     kernel_initializer=layer_initializer,
                                     name='reference_hidden_3')(reference_hidden2)

    ####################################################


    feed_forward_input = layers.concatenate([referenece_output, trade_history_output])

    hidden = layers.Dense(hp.Int("output_block_1_units", min_value=100, max_value=900, step=50), 
                          activation='relu',
                          kernel_initializer=layer_initializer)(feed_forward_input)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(hp.Choice("droupout_5", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]))(hidden)
    
    hidden2 = layers.Dense(hp.Int("output_block_2_units", min_value=100, max_value=900, step=50), 
                           activation='tanh',
                           kernel_initializer=layer_initializer)(hidden)
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dropout(hp.Choice("droupout_6", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]))(hidden2)

    final = layers.Dense(1,
                         kernel_initializer=layer_initializer)(hidden2)


    model = keras.Model(inputs=inputs, outputs=final)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.MeanAbsoluteError()])
    
    return model

if __name__ == '__main__':
    if not os.path.isfile(PROCESSED_FILE):
        reference_data = process_data(DATA_QUERY, 
                                bq_client,
                                SEQUENCE_LENGTH,
                                NUM_FEATURES,
                                'data.pkl')
        reference_data.to_pickle(PROCESSED_FILE)
    else:
        print('Reading from processed file')
        reference_data = pd.read_pickle(PROCESSED_FILE)
    
    yield_curve_params(bq_client)
    processed_data = process_reference_data(reference_data.copy())
    train_index = int(len(processed_data) * (1-TRAIN_TEST_SPLIT))
    train_dataframe = processed_data[train_index:]
    test_dataframe = processed_data[:train_index]
    print(f'Print length of training data frame {len(train_dataframe)}')
    print(f'Pring length of testing data frame {len(test_dataframe)}')
    
    print('Creating encoders for categorical features')
    
    encoders = {}
    fmax = {}
    for f in CATEGORICAL_FEATURES:
        fprep = preprocessing.LabelEncoder().fit(processed_data[f].drop_duplicates())
        fmax[f] = np.max(fprep.transform(fprep.classes_))
        encoders[f] = fprep
    
    print(encoders)
    

    tuner = RandomSearch(
        build_model,
        objective="val_mean_absolute_error",
        max_trials=25,
        overwrite=True,
        directory="model_tuning",
        project_name="yield_spread_model",
        distribution_strategy=tf.distribute.MirroredStrategy()
    )
    
    print(tuner.search_space_summary())

    val_index = int(len(train_dataframe) * (1-0.9))
    train_dataframe = train_dataframe[val_index:]
    val_dataframe = train_dataframe[:val_index]
    
    x_train = create_input(train_dataframe)
    y_train = train_dataframe.yield_spread
    
    x_val = create_input(val_dataframe)
    y_val = val_dataframe.yield_spread

    tuner.search(x_train, y_train, epochs=100, validation_data=(x_val, y_val))
