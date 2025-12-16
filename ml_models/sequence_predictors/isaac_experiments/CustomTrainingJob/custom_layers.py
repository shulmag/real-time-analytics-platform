import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
import regex as r
import time

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

