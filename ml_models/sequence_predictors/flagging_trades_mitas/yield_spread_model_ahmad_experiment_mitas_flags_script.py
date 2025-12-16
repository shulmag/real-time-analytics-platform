import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers.experimental.preprocessing import Normalization

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

import lightgbm
from lightgbm import LGBMRegressor


from ficc.utils.auxiliary_variables import PREDICTORS, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES, IDENTIFIERS, IS_BOOKKEEPING, IS_SAME_DAY, IS_REPLICA, NTBC_PRECURSOR
from ficc.utils.adding_flags import add_replica_flag, add_bookkeeping_flag, add_same_day_flag, add_ntbc_precursor_flag


TRAIN_TEST_SPLIT = 0.85
LEARNING_RATE = 0.0001
BATCH_SIZE = 1000
NUM_EPOCHS = 100

DROPOUT = 0.01
SEQUENCE_LENGTH = 5
NUM_FEATURES = 5

DATE_SPLIT = '12-01-2021'

def load_file():
    filename = '../data/processed_data_ficc_ycl_2021-12-31-23-59.pkl'
    filename_with_exclusions_and_flags = '../data/processed_data_ficc_ycl_2021-12-31-23-59_with_exclusions_is_same_day_flag_ntbc_precursor_flag.pkl'

    add_exclusions_and_flags = False
    if os.path.isfile(filename_with_exclusions_and_flags):
        data = pd.read_pickle(filename_with_exclusions_and_flags)
    else:
        data = pd.read_pickle(filename)
        add_exclusions_and_flags = True

    # putting in exclusions
    if add_exclusions_and_flags:
        data = data[(data.days_to_call == 0) | (data.days_to_call > np.log10(400))]
        data = data[(data.days_to_refund == 0) | (data.days_to_refund > np.log10(400))]
        data = data[data.days_to_maturity < np.log10(30000)]
        data = data[data.sinking == False]
        data = data[data.incorporated_state_code != 'VI']
        data = data[data.incorporated_state_code != 'GU']
        data = data[(data.coupon_type == 8)]
        data = data[data.is_called == False]
        data = data[~data.purpose_sub_class.isin([6, 20, 22, 44, 57, 90])]
        data = data[~data.called_redemption_type.isin([18, 19])]

    # adding flags
    if add_exclusions_and_flags:
        # data = add_replica_flag(data, IS_REPLICA)
        # data = add_bookkeeping_flag(data, IS_BOOKKEEPING)
        data = add_same_day_flag(data, IS_SAME_DAY)
        data = add_ntbc_precursor_flag(data, NTBC_PRECURSOR)
        data.to_pickle(filename_with_exclusions_and_flags)

    return data


PREDICTORS.append('target_attention_features')
CATEGORICAL_FEATURES.append('moodys_long')
PREDICTORS.append('moodys_long')

assert IS_SAME_DAY not in BINARY and IS_SAME_DAY not in PREDICTORS, 'Flags should not be in `BINARY` or `PREDICTORS` for these experiments to run, since the experiments test whether adding the flags help model performance'
assert IS_REPLICA not in BINARY and IS_REPLICA not in PREDICTORS, 'Flags should not be in `BINARY` or `PREDICTORS` for these experiments to run, since the experiments test whether adding the flags help model performance'
assert IS_BOOKKEEPING not in BINARY and IS_BOOKKEEPING not in PREDICTORS, 'Flags should not be in `BINARY` or `PREDICTORS` for these experiments to run, since the experiments test whether adding the flags help model performance'
assert NTBC_PRECURSOR not in BINARY and NTBC_PRECURSOR not in PREDICTORS, 'Flags should not be in `BINARY` or `PREDICTORS` for these experiments to run, since the experiments test whether adding the flags help model performance'

BINARY_IS_SAME_DAY = BINARY + [IS_SAME_DAY]
PREDICTORS_IS_SAME_DAY = PREDICTORS + [IS_SAME_DAY]

BINARY_ALL_FLAGS = BINARY + [IS_BOOKKEEPING, IS_SAME_DAY, IS_REPLICA]
PREDICTORS_ALL_FLAGS = PREDICTORS + [IS_BOOKKEEPING, IS_SAME_DAY, IS_REPLICA]

BINARY_NTBC_PRECURSOR = BINARY + [NTBC_PRECURSOR]
PREDICTORS_NTBC_PRECURSOR = PREDICTORS + [NTBC_PRECURSOR]

BINARY_IS_SAME_DAY_NTBC_PRECURSOR = BINARY + [IS_SAME_DAY, NTBC_PRECURSOR]
PREDICTORS_IS_SAME_DAY_NTBC_PRECURSOR = PREDICTORS + [IS_SAME_DAY, NTBC_PRECURSOR]


def adding_target_trade_features_to_calculate_attention(data):
    trade_mapping = {'D':[0,0], 'S':[0,1], 'P':[1,0]}
    def target_trade_processing_for_attention(row):
        target_trade_features = []
        target_trade_features.append(row['quantity'])
        target_trade_features = target_trade_features + trade_mapping[row['trade_type']]
        #target_trade_features.append(row['coupon'])
        return np.tile(target_trade_features, (5,1))

    data['target_attention_features'] = data.apply(target_trade_processing_for_attention, axis = 1)
    data.purpose_sub_class.fillna(0, inplace=True)

    data.loc[data.sp_stand_alone.isna(), 'sp_stand_alone'] = 'NR'
    data.moodys_long = data.moodys_long.astype(str)
    data.loc[data.moodys_long == 'nan' ,'moodys_long'] = 'NR'
    data.loc[data.moodys_long == '#Aaa' ,'moodys_long'] = 'Aaa'
    data.rating = data.rating.astype('str')
    data.sp_stand_alone = data.sp_stand_alone.astype('str')
    data.loc[(data.sp_stand_alone != 'NR'),'rating'] = data[(data.sp_stand_alone != 'NR')]['sp_stand_alone'].loc[:]
    data[(data.sp_stand_alone != 'NR')][['rating','sp_stand_alone','sp_long']]

    return data


def encode(data):
    encoders = {}
    fmax = {}
    for f in CATEGORICAL_FEATURES:
        fprep = preprocessing.LabelEncoder().fit(data[f].drop_duplicates())
        fmax[f] = np.max(fprep.transform(fprep.classes_))
        encoders[f] = fprep
    return encoders, fmax


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


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()


def train_model(data, binary, encoders, fmax, seed):
    train_dataframe = data[(data.trade_date < DATE_SPLIT)]
    test_dataframe = data[(data.trade_date >= DATE_SPLIT)]

    def create_input(df):
        datalist = []
        datalist.append(np.stack(df['trade_history'].to_numpy()))
        datalist.append(np.stack(df['target_attention_features'].to_numpy()))

        noncat_and_binary = []
        for f in NON_CAT_FEATURES + binary:
            noncat_and_binary.append(np.expand_dims(df[f].to_numpy().astype('float32'), axis=1))
        datalist.append(np.concatenate(noncat_and_binary, axis=-1))
        
        for f in CATEGORICAL_FEATURES:
            encoded = encoders[f].transform(df[f])
            datalist.append(encoded.astype('float32'))
        
        return datalist

    x_train = create_input(train_dataframe)
    y_train = train_dataframe.yield_spread
    x_test = create_input(test_dataframe)
    y_test = test_dataframe.yield_spread

    # Normalization layer for the trade history
    trade_history_normalizer = Normalization(name='Trade_history_normalizer')
    trade_history_normalizer.adapt(x_train[0])

    # Normalization layer for the non-categorical and binary features
    noncat_binary_normalizer = Normalization(name='Numerical_binary_normalizer')
    noncat_binary_normalizer.adapt(x_train[2])

    tf.keras.utils.set_random_seed(seed)

    inputs = []
    layer = []

    ############## INPUT BLOCK ###################
    trade_history_input = layers.Input(name="trade_history_input", 
                                    shape=(SEQUENCE_LENGTH,NUM_FEATURES), 
                                    dtype = tf.float32) 

    target_attention_input = layers.Input(name="target_attention_input", 
                                    shape=(SEQUENCE_LENGTH, 3), 
                                    dtype = tf.float32) 


    inputs.append(trade_history_input)
    inputs.append(target_attention_input)

    inputs.append(layers.Input(
        name="NON_CAT_AND_BINARY_FEATURES",
        shape=(len(NON_CAT_FEATURES + binary),)
    ))


    layer.append(noncat_binary_normalizer(inputs[2]))
    ####################################################


    ############## TRADE HISTORY MODEL #################

    # Adding the time2vec encoding to the input to transformer
    lstm_layer = layers.LSTM(50, 
                            activation='tanh',
                            input_shape=(SEQUENCE_LENGTH,NUM_FEATURES),
                            return_sequences = True,
                            name='LSTM')

    # lstm_attention_layer = layers.Attention(use_scale=True, name='attention_layer_1')
    lstm_attention_layer = CustomAttention(50)

    lstm_layer_2 = layers.LSTM(100, 
                            activation='tanh',
                            input_shape=(SEQUENCE_LENGTH,50),
                            return_sequences = False,
                            name='LSTM_2')


    features = lstm_layer(trade_history_normalizer(inputs[0]))
    # features = lstm_attention_layer([features, features])
    features = lstm_attention_layer(features, features, inputs[1])
    features = layers.BatchNormalization()(features)
    # features = layers.Dropout(DROPOUT)(features)

    features = lstm_layer_2(features)
    features = layers.BatchNormalization()(features)
    # features = layers.Dropout(DROPOUT)(features)

    trade_history_output = layers.Dense(100, 
                                        activation='relu')(features)
    ####################################################

    ############## REFERENCE DATA MODEL ################
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

    feed_forward_input = layers.concatenate([reference_output, trade_history_output])

    hidden = layers.Dense(300,activation='relu')(feed_forward_input)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(DROPOUT)(hidden)

    hidden2 = layers.Dense(100,activation='tanh')(hidden)
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dropout(DROPOUT)(hidden2)

    final = layers.Dense(1)(hidden2)

    model = keras.Model(inputs=inputs, outputs=final)

    # print(model.summary())

    tf.keras.utils.plot_model(
        model,
        show_shapes=False,
        show_layer_names=True,
        rankdir="LR",
        expand_nested=False,
        dpi=96,
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=keras.losses.MeanAbsoluteError(),
            metrics=[keras.metrics.MeanAbsoluteError()])

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

    history = model.fit(x_train, 
                        y_train, 
                        epochs=NUM_EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        verbose=1, 
                        validation_split=0.1, 
                        callbacks=fit_callbacks,
                        use_multiprocessing=True,
                        workers=8)

    # plt.plot(range(len(history.history['val_loss'])),history.history['val_loss'], label='val_loss')
    # plt.plot(range(len(history.history['loss'])),history.history['loss'], label='loss')
    # plt.title('Validation loss and training Loss per epoch')
    # plt.legend(loc="upper right")
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()

    _, mae = model.evaluate(x_test, y_test, verbose=1)
    print(f"NN Test loss: {round(mae, 3)}")

    return model


def train_lightgbm(data, binary, predictors, name):
    def gbmprep(df):
        df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype('category')
        df[NON_CAT_FEATURES + binary] = df[NON_CAT_FEATURES + binary].astype('float64')
        df = df.drop(columns = ['trade_history','target_attention_features','yield_spread','calc_day_cat'])
        return df

    train_dataframe = data[(data.trade_date < DATE_SPLIT)]
    test_dataframe = data[(data.trade_date >= DATE_SPLIT)]
    trainlabel = train_dataframe.yield_spread
    testlabel = test_dataframe.yield_spread

    gbt_td = gbmprep(train_dataframe[predictors])
    gbtmodel = LGBMRegressor(num_iterations=300, max_depth=12, num_leaves=300, objective='mae', verbosity=-1)
    gbtmodel.fit(gbt_td, trainlabel)

    gbt_pred = gbtmodel.predict( gbmprep(test_dataframe[predictors]) )
    delta = testlabel - gbt_pred
    # print( delta.mean(), delta.abs().mean() )
    print(f"LightGBM Test loss: {round(delta.abs().mean(), 3)}")

    lightgbm.plot_importance(gbtmodel).figure.savefig(f'{name}_lightgbm_feature_importance.pdf')    # https://stackoverflow.com/questions/56151815/how-to-save-feature-importance-plot-of-xgboost-to-a-file-from-jupyter-notebook

    return gbtmodel


if __name__ == '__main__':
    data = load_file()
    data = adding_target_trade_features_to_calculate_attention(data)

    binary_predictors_name_groups = [
            (BINARY, PREDICTORS, 'no_flags'), 
            # (BINARY_ALL_FLAGS, PREDICTORS_ALL_FLAGS, 'all'), 
            # (BINARY_IS_SAME_DAY, PREDICTORS_IS_SAME_DAY, 'is_same_day'), 
            (BINARY_NTBC_PRECURSOR, PREDICTORS_NTBC_PRECURSOR, 'ntbc_precursor'), 
            (BINARY_IS_SAME_DAY_NTBC_PRECURSOR, PREDICTORS_IS_SAME_DAY_NTBC_PRECURSOR, 'is_same_day_and_ntbc_precursor'),
        ]

    for binary, predictors, name in binary_predictors_name_groups:
        processed_data = data[IDENTIFIERS + predictors + ['dollar_price','calc_date', 'trade_date','trade_datetime', 'purpose_sub_class', 'called_redemption_type', 'calc_day_cat', 'muni_issue_type', 'yield', 'ficc_ycl']]
        encoders, fmax = encode(processed_data)
        for seed in [10, 11, 12]:
            print(f'data: {name}. seed: {seed}')
            model = train_model(processed_data, binary, encoders, fmax, seed)
        print(f'data: {name}. lightgbm')
        gbtmodel = train_lightgbm(processed_data, binary, predictors, name)
