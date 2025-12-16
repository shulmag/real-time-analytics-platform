import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split as train_val_split_func

from ficc.utils.auxiliary_variables import TRADE_HISTORY

import sys
sys.path.insert(0,'../')

from yield_spread_model_mitas.models import SUPPORTED_RNN_ARCHITECTURES, _YieldSpreadPredictor, _L1Loss, _CheckNumParameters, _YieldSpreadPredictorEmbeddings, _HiddenLayers
from yield_spread_model_mitas.datasets_pytorch import EmbeddingsDataset, TradeHistoryDataset

from rating_model_mitas.models import TRAIN_VAL_SPLIT
from rating_model_mitas.datasets_pytorch import DatasetFromData


'''An l1 loss recurrent model which processes multiple sources of trade history. This implementation 
is almost identical to `RecurrentL1Loss` in yield_spread_model_mitas/models.py, except for the fact that 
it generalizes the code to allow for multiple trade histories to be passed in.'''
class MultipleRecurrentL1Loss(_YieldSpreadPredictor, _L1Loss, _CheckNumParameters):
    def __init__(self, 
                 batch_size, 
                 num_workers, 
                 train_trade_history, 
                 test_trade_history, 
                 num_layers=1, 
                 hidden_size=1, 
                 recurrent_architecture='lstm', 
                 trade_history_columns=TRADE_HISTORY, 
                 train_val_split=TRAIN_VAL_SPLIT):
        assert recurrent_architecture in SUPPORTED_RNN_ARCHITECTURES, f'{recurrent_architecture} is not supported by this model. This model only supports: {SUPPORTED_RNN_ARCHITECTURES.keys()}.'
        super().__init__(batch_size, num_workers)
        self.hyperparameters['recurrent_architecture'] = recurrent_architecture    # used for wandb hyperparameter logging

        self.num_layers = num_layers    # used in __str__
        self.hyperparameters['num_layers_recurrent_architecture'] = num_layers    # used for wandb hyperparameter logging
        self.hidden_size = hidden_size    # used in __str__
        self.hyperparameters['hidden_size_recurrent_architecture'] = hidden_size    # used for wandb hyperparameter logging
        self.hyperparameters['num_rnns'] = len(trade_history_columns)    # used for wand hyperparamter logging
        self.setup_dataloaders(train_trade_history, test_trade_history, train_val_split, trade_history_columns)
        
        self.recurrent_architecture = recurrent_architecture    # used for __str__
        self.recurrents = nn.ModuleList([SUPPORTED_RNN_ARCHITECTURES[recurrent_architecture](self.num_features_per_trade, hidden_size, num_layers) for _ in range(len(trade_history_columns))])
        self.linear = nn.Linear(len(trade_history_columns) * hidden_size, 1)
        self.check_num_parameters()

    def setup_dataloaders(self, train_trade_history, test_trade_history, train_val_split, trade_history_columns):
        self.size = len(train_trade_history)
        self.hyperparameters['num_train_datapoints'] = self.size    # used for wandb hyperparameter logging
        train_val_dataset = TradeHistoryDataset(train_trade_history)
        num_train_datapoints = int(self.size * train_val_split)
        num_val_datapoints = self.size - num_train_datapoints

        self.num_features_per_trade = -1    # initialization
        for column in trade_history_columns:    # check whether each trade history column has the same number of features per trade
            num_features_per_trade = len(train_trade_history[column].iloc[0][0])
            if self.num_features_per_trade == -1: self.num_features_per_trade = num_features_per_trade
            else: assert num_features_per_trade == self.num_features_per_trade, f'All trade history columns must have the same number of features per trade. Check column: {column}'
        
        if len(trade_history_columns) == 1:    # special case of just a single history column
            val_inputs, train_inputs = train_val_split_func(train_val_dataset.inputs, train_size=num_val_datapoints, shuffle=False)
            val_inputs, train_inputs = [val_inputs], [train_inputs]    # wrap with list brackets to unwrap later with the same syntax as the case where there are multiple trade history columns
        else:
            val_inputs, train_inputs = [], []
            for idx in range(len(trade_history_columns)):
                val_inputs_idx, train_inputs_idx = train_val_split_func(train_val_dataset.inputs[idx], train_size=num_val_datapoints, shuffle=False)
                val_inputs.append(val_inputs_idx)
                train_inputs.append(train_inputs_idx)
        
        val_labels, train_labels = train_val_split_func(train_val_dataset.labels, train_size=num_val_datapoints, shuffle=False)

        self.train_dataloader_object = self.create_dataloader_object(DatasetFromData(*train_inputs, train_labels), True)
        self.val_dataloader_object = self.create_dataloader_object(DatasetFromData(*val_inputs, val_labels), False)
        
        self.test_dataset = TradeHistoryDataset(test_trade_history) if test_trade_history is not None else None    # enter else clause if test data is unlabeled

    def forward(self, *x):
        x_list = [recurrent(x[idx])[0][:, -1, :] for idx, recurrent in enumerate(self.recurrents)]    # the first index being `:` means we want the entire batch, the second index being `-1` means we want the output of the last layer, the third index being `:` means we want the entire vector output of the layer
        x = torch.cat(x_list, dim=1)
        return self.linear(x)

    def step(self, batch, batch_idx, name):
        inputs, labels = batch[:-1], batch[-1]
        outputs = self(*inputs)
        loss = self.loss_func(outputs, labels)
        self.log(name, loss)
        return {name: loss}

    def __str__(self):
        return f'Multiple {self.recurrent_architecture} L1 Loss with {self.num_layers} layers and {self.hidden_size} hidden size'


'''An l1 loss NN with embedding layers on each of the categorical features combined with 
multiple recurrent models which processes multiple sources of trade history. This implementation 
is a combination of `NNL1LossEmbeddingsWithRecurrence` in yield_spread_model_mitas/models.py and 
`MultipleRecurrentL1Loss` in this file. The parameters for the recurrent model are given by 
`num_layers_recurrent` and `hidden_size_recurrent`, while the architecture is given as a string 
in `recurrent_architecture`. All other parameters are for the feed forward network.'''
class NNL1LossEmbeddingsWithMultipleRecurrence(_YieldSpreadPredictorEmbeddings, _HiddenLayers, _L1Loss, _CheckNumParameters):
    def __init__(self, 
                 batch_size, 
                 num_workers, 
                 train_data_with_trade_history, 
                 test_data_with_trade_history, 
                 label_encoders, 
                 categorical_features, 
                 num_nodes_hidden_layer=0, 
                 num_hidden_layers=0, 
                 num_layers_recurrent=1, 
                 hidden_size_recurrent=1, 
                 recurrent_architecture='lstm', 
                 trade_history_columns=TRADE_HISTORY, 
                 train_val_split=TRAIN_VAL_SPLIT, 
                 power=0.5, 
                 batch_normalization=False):
        assert recurrent_architecture in SUPPORTED_RNN_ARCHITECTURES, f'{recurrent_architecture} is not supported by this model. This model only supports: {SUPPORTED_RNN_ARCHITECTURES.keys()}.'
        super().__init__(batch_size, num_workers)
        self.power = power
        self.hyperparameters['power'] = power    # used for wandb hyperparameter logging
        self.setup_dataloaders(train_data_with_trade_history, test_data_with_trade_history, categorical_features, train_val_split, trade_history_columns)
        self.setup_embeddings(label_encoders)
        first_layer_dimension = self.total_embeddings_dimension + len(self.train_val_dataset.column_names_for_binary_and_continuous_features) + hidden_size_recurrent * len(trade_history_columns)
        self.layer_initializer(num_nodes_hidden_layer, num_hidden_layers, first_layer_dimension, batch_normalization)

        self.recurrent_architecture = recurrent_architecture    # used for __str__
        self.hyperparameters['recurrent_architecture'] = recurrent_architecture    # used for wandb hyperparameter logging
        self.num_recurrents = len(trade_history_columns)    # used for __str__
        self.recurrents = nn.ModuleList([SUPPORTED_RNN_ARCHITECTURES[recurrent_architecture](self.num_features_per_trade, hidden_size_recurrent, num_layers_recurrent) for _ in range(self.num_recurrents)])
        self.hyperparameters['num_layers_recurrent_architecture'] = num_layers_recurrent    # used for wandb hyperparameter logging
        self.hyperparameters['hidden_size_recurrent_architecture'] = hidden_size_recurrent    # used for wandb hyperparameter logging

        self.check_num_parameters()

    def setup_dataloaders(self, train_data_with_trade_history, test_data_with_trade_history, categorical_features, train_val_split, trade_history_columns):
        self.size = len(train_data_with_trade_history)
        self.hyperparameters['num_train_datapoints'] = self.size    # used for wandb hyperparameter logging
        num_train_datapoints = int(self.size * train_val_split)
        num_val_datapoints = self.size - num_train_datapoints

        train_data_without_trade_history = train_data_with_trade_history.drop(columns=trade_history_columns)
        test_data_without_trade_history = test_data_with_trade_history.drop(columns=trade_history_columns)
        trade_history_train = train_data_with_trade_history[trade_history_columns + ['yield_spread']]
        trade_history_test = test_data_with_trade_history[trade_history_columns + ['yield_spread']]
        
        self.num_features_per_trade = -1    # initialization
        for column in trade_history_columns:    # check whether each trade history column has the same number of features per trade
            num_features_per_trade = len(train_data_with_trade_history[column].iloc[0][0])
            if self.num_features_per_trade == -1: self.num_features_per_trade = num_features_per_trade
            else: assert num_features_per_trade == self.num_features_per_trade, f'All trade history columns must have the same number of features per trade. Check column: {column}'
        
        self.train_val_dataset = EmbeddingsDataset(train_data_without_trade_history, categorical_features, label_feature='yield_spread')

        # train and val are flipped because the order of the data is is in descending order of trade_datetime (see query) and we want the training set to be earlier in time than the validation set
        train_val_split_func_caller = lambda array: train_val_split_func(array, train_size=num_val_datapoints, shuffle=False)
        val_inputs_categorical, train_inputs_categorical = train_val_split_func_caller(self.train_val_dataset.inputs_categorical)
        val_inputs_binary_and_continuous, train_inputs_binary_and_continuous = train_val_split_func_caller(self.train_val_dataset.inputs_binary_and_continuous)
        val_labels, train_labels = train_val_split_func_caller(self.train_val_dataset.labels)

        train_val_dataset_trade_history = TradeHistoryDataset(trade_history_train)
        if len(trade_history_columns) == 1:    # special case of just a single history column
            val_inputs_trade_history, train_inputs_trade_history = train_val_split_func_caller(train_val_dataset_trade_history.inputs)
            val_inputs_trade_history, train_inputs_trade_history = [val_inputs_trade_history], [train_inputs_trade_history]    # wrap with list brackets to unwrap later with the same syntax as the case where there are multiple trade history columns
        else:
            val_inputs_trade_history, train_inputs_trade_history = [], []
            for idx in range(len(trade_history_columns)):
                val_inputs_trade_history_idx, train_inputs_trade_history_idx = train_val_split_func_caller(train_val_dataset_trade_history.inputs[idx])
                val_inputs_trade_history.append(val_inputs_trade_history_idx)
                train_inputs_trade_history.append(train_inputs_trade_history_idx)

        self.train_dataloader_object = self.create_dataloader_object(DatasetFromData(train_inputs_categorical, 
                                                                                     train_inputs_binary_and_continuous, 
                                                                                     *train_inputs_trade_history, 
                                                                                     train_labels),
                                                                     True)
        self.val_dataloader_object = self.create_dataloader_object(DatasetFromData(val_inputs_categorical, 
                                                                                   val_inputs_binary_and_continuous, 
                                                                                   *val_inputs_trade_history, 
                                                                                   val_labels), 
                                                                   False)

        test_dataset = EmbeddingsDataset(test_data_without_trade_history, categorical_features, label_feature='yield_spread')
        test_dataset_trade_history = TradeHistoryDataset(trade_history_test)
        self.test_dataset = DatasetFromData(test_dataset.inputs_categorical, 
                                            test_dataset.inputs_binary_and_continuous, 
                                            *test_dataset_trade_history.inputs, 
                                            test_dataset.labels)

    def forward(self, x_categorical, x_binary_and_continuous, *x_trade_history):
        x_embedded = self.apply_embeddings(x_categorical, x_binary_and_continuous)
        x_trade_history_processed = [recurrent(x_trade_history[idx])[0][:, -1, :] for idx, recurrent in enumerate(self.recurrents)]    # the first index being `:` means we want the entire batch, the second index being `-1` means we want the output of the last layer, the third index being `:` means we want the entire vector output of the layer
        x = torch.cat([x_embedded, *x_trade_history_processed], dim=1)
        return self.apply_layers(x)

    def step(self, batch, batch_idx, name):
        return MultipleRecurrentL1Loss.step(self, batch, batch_idx, name)

    def __str__(self):
        return f'NN L1 Loss with embeddings (power={self.power}) and hidden layers {self.num_nodes_hidden_layer} and {self.num_recurrents} {self.recurrent_architecture}'
