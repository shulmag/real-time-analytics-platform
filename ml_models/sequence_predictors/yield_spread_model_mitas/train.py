import time    # used for gigaflop computation of training procedure (i.e., `trainer.fit(...)`)
import warnings

import numpy as np
import pandas as pd

from datetime import datetime

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import wandb

import sys
sys.path.insert(0,'../')

from rating_model_mitas.train import load_model, save_model


TRAIN_TEST_SPLIT = 0.8
TRAIN_TEST_TRADE_DATE = datetime(2022, 4, 1)    # April 1 2022
BOUNDARY = 50    # used to format the print output of the training by separating different training runs


'''Checks whether a CUDA gpu is available to be used for training.'''
def is_gpu_available():
    return torch.cuda.is_available()


'''Checks whether mps is available to be used for training. `mps` corresponds  
to the GPU on the Mac M1. If run on the regular (non-nightly) version of PyTorch, 
then this raises the following error:
AttributeError: module 'torch.backends' has no attribute 'mps'.'''
def is_mps_available():
    try:
        return torch.backends.mps.is_available()
    except AttributeError:
        return False


def _get_device():
    if is_gpu_available(): return 'gpu'
    # elif is_mps_available(): return 'mps'
    else: return 'cpu'    # update the condition with `or is_mps_available()` when PyTorch lightning trainer supports the mps accelerator: https://github.com/PyTorchLightning/pytorch-lightning/issues/13102


'''Get the predicted label from the model and inputs.'''
def get_predictions(model, *inputs):
    return model(*inputs)


'''Returns the l1 loss of the `labels` versus the `predictions` of the model.'''
l1_loss_func = lambda predictions, labels: F.l1_loss(torch.squeeze(predictions), labels, reduction='none')    # torch.squeeze removes the unnecessary dimension of 1
l1_loss_func_numpy = lambda predictions, labels: np.absolute(predictions - labels)


'''Returns the l1 loss of the `labels` versus the `predictions` of the model.'''
l2_loss_func = lambda predictions, labels: F.mse_loss(torch.squeeze(predictions), labels, reduction='none')    # torch.squeeze removes the unnecessary dimension of 1
l2_loss_func_numpy = lambda predictions, labels: (predictions - labels) ** 2


'''This function returns two different loss measures as described in the 
comments of the functions `_l1_loss(...)`, and `_l2_loss(...)`.'''
def get_loss(dataloader, model):
    total_l1_loss = 0
    total_l2_loss = 0
    total = 0
    for inputs_and_labels in dataloader:
        labels = inputs_and_labels.pop()
        inputs = inputs_and_labels
        predictions = model(*inputs)

        total += labels.shape[0]
        total_l1_loss += l1_loss_func(predictions, labels).sum()
        total_l2_loss += l2_loss_func(predictions, labels).sum()

    return total_l1_loss / total, total_l2_loss / total


'''Prints the losses for the train, validation, and test sets respectively.'''
def get_all_losses(model, num_decimal_places=3):
    print('#' * BOUNDARY)
    print(f'Model: {model}')
    printer_string = lambda losses, label: f'{label} L1 loss: {round(losses[0].item(), num_decimal_places)}, {label} L2 loss: {round(losses[1].item(), num_decimal_places)}'
    train_losses = get_loss(model.train_dataloader(), model)
    print(printer_string(train_losses, 'Train'))
    val_losses = get_loss(model.val_dataloader(), model)
    print(printer_string(val_losses, 'Validation'))

    test_data_exists = model.test_dataloader() is not None
    if test_data_exists:
        test_losses = get_loss(model.test_dataloader(), model)
        print(printer_string(test_losses, 'Test'))
    print('#' * BOUNDARY)

    return (train_losses, val_losses, test_losses) if test_data_exists else (train_losses, val_losses)     # return losses


'''This function trains a `model` for a number of epochs specified in `epochs`. If the model 
already exists in `model_filename`, just loads the model and returns it, otherwise saves the 
trained model to this `model_filename` if the `model_filename` is not `None`. The flag 
`print_losses_before_training` determines whether the losses of the model (with random weight 
initializations) are outputted to the user to be seen as a sanity check. `wandb_logging_name` 
is the name which the run is stored as in Weights & Biases, and if this value is `None`, then 
the run is not stored. `model_hyperparameters` is a dictionary of hyperparameters used to 
create the model that will be stored in Weights & Biases. The flag `keep_best_validation_model` 
determines whether the final returned model is the the model that had the lowest validation 
loss, or is the model that was the end result of the training process.'''
def train(model, 
          epochs=100, 
          model_filename=None, 
          save=False, 
          print_losses_before_training=False, 
          print_losses_after_training=True, 
          wandb_logging_name=None, 
          wandb_project='mitas_yield_spread', 
          keep_best_validation_model=True):
    model_at_filename = load_model(model_filename, model)
    if model_at_filename is not None: model = model_at_filename
    else:
        if print_losses_before_training:
            get_all_losses(model)
        else:
            print('#' * BOUNDARY)
            print(f'Model: {model}')

        if wandb_logging_name is not None:
            wandb_logger = WandbLogger(project=wandb_project, entity='ficc-ai', name=wandb_logging_name)
            wandb_logger.experiment.config.update(model.hyperparameters)
        else:
            wandb_logger = False
        # if model_hyperparameters is not None and wandb_logger is not None:
        #     wandb_logger.experiment.config.update(model_hyperparameters)
        checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', dirpath='checkpoints/')
        trainer = pl.Trainer(max_epochs=epochs, 
                             accelerator=_get_device(), 
                             logger=wandb_logger, 
                             callbacks=[checkpoint] if keep_best_validation_model else None)

        # train the model
        start_time = time.time()    # used for gigaflop computation
        train_dataloader = model.train_dataloader()
        trainer.fit(model, train_dataloader, model.val_dataloader())
        elapsed_time = time.time() - start_time    # used for gigaflop computation
        num_flops = 2 * model.num_parameters * len(train_dataloader.dataset)    # 1 flop per trainable parameter for forward pass, 1 per backward pass, all multiplied by the number of examples
        print(f'The training procedure took {round(elapsed_time, 3)} seconds, so the average performance was: {(num_flops / 1e9) / elapsed_time} gigaflops / second')

        # Reload the checkpoint of the best model, to this point
        if keep_best_validation_model:
            best_checkpoint = torch.load(checkpoint.best_model_path)
            model.load_state_dict(best_checkpoint['state_dict'])
        test_results_dict = trainer.test(model, model.test_dataloader())[0]    # trainer.test returns a list of dictionaries where each item in the list is a dictionary of the test results for each dataloader passed in (currently passing in a single dataloader)
        if wandb_logging_name is not None:
            wandb.finish()
        if save:
            if type(model_filename) != str: warnings.warn(f'The filename passed in is {model_filename}, which is not of type `str`, so it cannot be used to create a file. Model will not be saved.')
            else: save_model(model_filename, model)
    
    if print_losses_after_training:    # sometimes running the get_all_losses procedure causes the kernel to crash
        print('Returning model and ALL losses')
        return model, get_all_losses(model)
    else:
        print('Returning model and test loss')
        return model, test_results_dict['test_loss']


'''Trains a PyTorch model using `train_data`, which should be encoded, and each 
combination of hyperparameters from the candidates in `num_hidden_layers_candidates` 
and `num_nodes_hidden_layer_candidates`. To determine model performance, the metric 
used is the l1 loss on `test_data`. Returns the best model found, the parameters as 
a dictionary, the train and test loss as a dictionary, and the dataframe storing the 
experiment information.'''
def train_hyperparameter_search(model, 
                                batch_size, 
                                num_workers, 
                                train_data, 
                                test_data, 
                                label_encoders, 
                                binary_features, 
                                categorical_features, 
                                num_hidden_layers_candidates, 
                                num_nodes_hidden_layer_candidates, 
                                num_epochs, 
                                save_progress_for_every_run=False):
    num_hidden_layers_in_result_df = []
    num_nodes_hidden_layer_in_result_df = []
    train_l1_loss_in_result_df = []
    test_l1_loss_in_result_df = []
    test_l1_loss_for_best_model = float('inf')
    best_model = None

    for num_hidden_layers in num_hidden_layers_candidates:
        for num_nodes_hidden_layer in num_nodes_hidden_layer_candidates:
            num_hidden_layers_in_result_df.append(num_hidden_layers)
            num_nodes_hidden_layer_in_result_df.append(num_nodes_hidden_layer)
            nn, losses = train(model(batch_size, 
                                     num_workers, 
                                     train_data, 
                                     test_data, 
                                     label_encoders, 
                                     binary_features, 
                                     categorical_features, 
                                     num_nodes_hidden_layer, 
                                     num_hidden_layers), 
                               num_epochs)
            train_l1_loss = losses[0][0].item()    # .item() extracts the raw value from the PyTorch tensor
            train_l1_loss_in_result_df.append(train_l1_loss)
            test_l1_loss = losses[2][0].item()    # .item() extracts the raw value from the PyTorch tensor
            test_l1_loss_in_result_df.append(test_l1_loss)
            if test_l1_loss < test_l1_loss_for_best_model:
                best_model = nn

            if save_progress_for_every_run:
                file = open('progress.txt', 'a')
                file.write(f'num_hidden_layers: {num_hidden_layers}, num_nodes_hidden_layer: {num_nodes_hidden_layer}, train_l1_loss: {train_l1_loss}, test_l1_loss: {test_l1_loss}\n')
                file.close()
            
            if num_hidden_layers == 0:    # no need to try multiple num_nodes_hidden_layer if num_hidden_layers is 0
                break

    result_df = pd.DataFrame({'num_hidden_layers': num_hidden_layers_in_result_df, 
                              'num_nodes_hidden_layer': num_nodes_hidden_layer_in_result_df, 
                              'train_l1_loss': train_l1_loss_in_result_df, 
                              'test_l1_loss': test_l1_loss_in_result_df})
    row_for_min_test_l1_loss = result_df[result_df['test_l1_loss'] == min(result_df['test_l1_loss'])].iloc[0]    # .iloc[0] extracts the row from the dataframe
    parameters = ('num_hidden_layers', 'num_nodes_hidden_layer')
    parameters_for_best_model = {parameter: row_for_min_test_l1_loss[parameter] for parameter in parameters}
    losses = ('train_l1_loss', 'test_l1_loss')
    losses_for_best_model = {loss: row_for_min_test_l1_loss[loss] for loss in losses}

    return best_model, parameters_for_best_model, losses_for_best_model, result_df


'''This function loads the model if found at `model_filename`, and otherwise trains it, 
and if `save` is True, then saves it to `model_filename`.'''
def load_model_or_train_model(model_filename, model, epochs=100, save=False):
    if load_model(model_filename, model) is None:
        losses = train(model, epochs)
        if model_filename is not None and save:
            save_model(model_filename, model)
        return losses, model
    else:
        print(f'Returning model')
        return model


'''Return train and test data based on having a fraction of the data in the train 
data, specified in `train_test_split`, and the remaining fraction in the test data. 
Assumes that the data is sorted in descending order of `trade_date` to ensure that 
the most recent data is in the test data.'''
def get_train_test_data_frac(data, train_test_split=TRAIN_TEST_SPLIT):
    num_data = len(data)
    num_train = int(train_test_split * num_data)
    num_test = num_data - num_train
    return data.tail(num_train), data.head(num_test)    # the tail contains the oldest data so should be used for training, the head contains the most recent data so should be used for testing


'''Return train and test data based on splitting at `trade_datetime`.'''
def get_train_test_data_trade_datetime(data, trade_datetime=TRAIN_TEST_TRADE_DATE):
    return get_train_test_data_from_start_end_trade_datetimes(data, train_end_trade_datetime=trade_datetime, test_start_trade_datetime=trade_datetime)


'''Return train and test data. Train data has all trades between 
`train_start_trade_datetime` and `train_end_trade_datetime`. Test data has 
all trades between `test_start_trade_datetime` and `test_end_trade_datetime`.'''
def get_train_test_data_from_start_end_trade_datetimes(data, train_start_trade_datetime=None, train_end_trade_datetime=None, test_start_trade_datetime=None, test_end_trade_datetime=None):
    assert 'trade_datetime' in data.columns

    def get_data_between_datetimes(df, start_datetime, end_datetime):
        return df[(start_datetime <= df['trade_datetime']) & (df['trade_datetime'] < end_datetime)]
    
    earliest_datetime_possible = pd.to_datetime(pd.Timestamp.min)    # cannot use `datetime.min`, otherwise will get OutOfBoundsDatetime Error
    latest_datetime_possible = pd.to_datetime(pd.Timestamp.max)    # cannot use `datetime.max`, otherwise will get OutOfBoundsDatetime Error
    train_start_trade_datetime = earliest_datetime_possible if train_start_trade_datetime is None else train_start_trade_datetime
    train_end_trade_datetime = latest_datetime_possible if train_end_trade_datetime is None else train_end_trade_datetime
    test_start_trade_datetime = earliest_datetime_possible if test_start_trade_datetime is None else test_start_trade_datetime
    test_end_trade_datetime = latest_datetime_possible if test_end_trade_datetime is None else test_end_trade_datetime

    train_data = get_data_between_datetimes(data, train_start_trade_datetime, train_end_trade_datetime)
    test_data = get_data_between_datetimes(data, test_start_trade_datetime, test_end_trade_datetime)
    assert len(train_data) + len(test_data) == len(data), f'Issue with split since train_data has size {len(train_data)}, test_data has size {len(test_data)}, but the original data has size {len(data)} which is NOT the sum of the splits.'
    return train_data, test_data
           

'''Return train and test data based on the `train_indices`. More specifically, 
the train data will be the datapoints from `data` with indices that appear in 
`train_indices` and the test data will be datapoints from `data` with indices 
that appear in `test_indices`. If `test_indices` is `None`, then the test data 
becomes all of the datapoints in `data` that are not in the training data.'''
def get_train_test_data_index(data, train_indices, test_indices=None):
    if test_indices is None: test_indices = ~data.index.isin(train_indices)    # select indices not in a list: https://stackoverflow.com/questions/29134635/slice-pandas-dataframe-by-index-values-that-are-not-in-a-list
    return data.loc[train_indices], data.loc[test_indices]
