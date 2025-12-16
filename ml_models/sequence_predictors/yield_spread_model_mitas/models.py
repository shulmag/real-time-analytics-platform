import warnings

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split as train_val_split_func

from ficc.utils.auxiliary_variables import TRADE_HISTORY
from ficc.utils.auxiliary_functions import flatten

# importing from parent directory: https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import sys
sys.path.insert(0,'../')

from yield_spread_model_mitas.train import l1_loss_func, l2_loss_func, l1_loss_func_numpy, l2_loss_func_numpy
from yield_spread_model_mitas.datasets_pytorch import EmbeddingsDataset, TradeHistoryDataset

from rating_model_mitas.datasets_pytorch import data_to_inputs_and_labels, OneHotDataset, DatasetFromData
from rating_model_mitas.models import TRAIN_VAL_SPLIT


NUM_PARAMETERS_THRESHOLD_PERCENT = 10    # raises a warning suggesting regularization if the number of model parameters are greater than NUM_PARAMETERS_THRESHOLD_PERCENT percent of the number of training data samples

# recurrent architectures used for processing trade history
SUPPORTED_RNN_ARCHITECTURES = {'rnn_tanh': lambda input_dimension, hidden_size, num_layers: nn.RNN(input_dimension, hidden_size, num_layers, nonlinearity='tanh', batch_first=True), 
                               'rnn_relu': lambda input_dimension, hidden_size, num_layers: nn.RNN(input_dimension, hidden_size, num_layers, nonlinearity='relu', batch_first=True), 
                               'lstm': lambda input_dimension, hidden_size, num_layers: nn.LSTM(input_dimension, hidden_size, num_layers, batch_first=True), 
                               'gru': lambda input_dimension, hidden_size, num_layers: nn.GRU(input_dimension, hidden_size, num_layers, batch_first=True)}


# below function takes in a list of outputs of the tied layers, where each output is a matrix where the first dimension is the batch size and the second dimension is the output of the tied layer
stack_tied_groups = lambda tied_groups, num_output_nodes: torch.cat(tied_groups, dim=0).view(len(tied_groups), -1, num_output_nodes)    # in view(...): first argument is number of past trades, second argument is batch size, third argument is num of nodes in output of tied layer
SUPPORTED_POOLING_FUNCTIONS = {'none': lambda tied_groups, _: torch.cat(tied_groups, dim=1), 
                               'max': lambda tied_groups, num_output_nodes: torch.max(stack_tied_groups(tied_groups, num_output_nodes), dim=0)[0],    # final indexing of 0 returns the actual max values instead of the argmax
                               'min': lambda tied_groups, num_output_nodes: torch.min(stack_tied_groups(tied_groups, num_output_nodes), dim=0)[0], 
                               'mean': lambda tied_groups, num_output_nodes: torch.mean(stack_tied_groups(tied_groups, num_output_nodes), dim=0, dtype=torch.float)}    # need to convert dtype to float following mean computation


'''Returns the value of `feature` as the prediction. No machine learning.'''
def single_feature_model(all_data, feature):
    inputs, labels = data_to_inputs_and_labels(all_data, 'yield_spread')
    l1_loss = l1_loss_func_numpy(inputs[feature].to_numpy(), labels.to_numpy()).mean()    # this line causes the kernel to restart locally with entire dataset; maybe due to dataset being too large
    l2_loss = l2_loss_func_numpy(inputs[feature].to_numpy(), labels.to_numpy()).mean()    # this line causes the kernel to restart locally with entire dataset; maybe due to dataset being too large
    print(f'L1 Loss for `{feature}` model: {l1_loss}. L2 loss for `{feature}` model: {l2_loss}')
    return l1_loss, l2_loss


'''Base class for a regressor trying to predict yield spread. Configures the optimizer to be ADAM and 
initializes the batch size and the number of workers. Assumes that the train and validation 
dataloaders will be attributes, and the test_dataset will be an attribute in the superclass.'''
class _YieldSpreadPredictor(pl.LightningModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.hyperparameters = dict()    # used for wandb hyperparameter logging
        self.batch_size = batch_size
        self.hyperparameters['batch_size'] = batch_size    # used for wandb hyperparameter logging
        self.num_workers = num_workers
        self.hyperparameters['num_workers'] = num_workers    # used for wandb hyperparameter logging

    def configure_optimizers(self):
        return optim.Adam(self.parameters())

    # step that occurs for a single batch
    def step(self, batch, batch_idx, name):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_func(outputs, labels)
        self.log(name, loss)
        return {name: loss}

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'loss')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val_loss')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test_loss')

    def create_dataloader_object(self, dataset, shuffle):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, persistent_workers=False)

    def train_dataloader(self):
        return self.train_dataloader_object

    def val_dataloader(self):
        return self.val_dataloader_object

    def test_dataloader(self):
        return self.create_dataloader_object(self.test_dataset, False) if self.test_dataset is not None else None


'''Base class for a NN regressor trying to predict yield spread using one hot encoding. Assumes that 
the categorical features will be the ones that will be one hot encoded.'''
class _YieldSpreadPredictorOneHot(_YieldSpreadPredictor):
    def __init__(self, batch_size, num_workers, train_data, test_data, label_encoders, categorical_features, train_val_split=TRAIN_VAL_SPLIT):
        super().__init__(batch_size, num_workers)
        self.hyperparameters['encoding'] = 'one_hot'    # used for wandb hyperparameter logging
        features_to_one_hot_encode = categorical_features
        self.setup_dataloaders(train_data, test_data, label_encoders, features_to_one_hot_encode, train_val_split)

        single_datapoint_input, _ = self.get_single_datapoint()
        self.first_layer_dimension = len(single_datapoint_input)

    def setup_dataloaders(self, train_data, test_data, label_encoders, features_to_one_hot_encode, train_val_split):
        self.size = len(train_data)
        self.hyperparameters['num_train_datapoints'] = self.size    # used for wandb hyperparameter logging
        train_val_dataset = OneHotDataset(train_data, label_encoders, features_to_one_hot_encode, 'yield_spread')
        num_train_datapoints = int(self.size * train_val_split)
        num_val_datapoints = self.size - num_train_datapoints

        # train and val are flipped because the order of the data is is in descending order of trade_datetime (see query) and we want the training set to be earlier in time than the validation set
        val_inputs, train_inputs, val_labels, train_labels = train_val_split_func(train_val_dataset.inputs, train_val_dataset.labels, train_size=num_val_datapoints, shuffle=False)
        self.train_dataloader_object = self.create_dataloader_object(DatasetFromData(train_inputs, train_labels), True)
        self.val_dataloader_object = self.create_dataloader_object(DatasetFromData(val_inputs, val_labels), False)    # shuffle=False due to PyTorch recommendation that validation set should not be shuffled

        self.test_dataset = OneHotDataset(test_data, label_encoders, features_to_one_hot_encode, label_feature='yield_spread') if test_data is not None else None    # enter else clause if test data is unlabeled

    # gets the first datapoint from the first batch and uses it to get the dimension of the datapoint to be used in constructing the layers
    def get_single_datapoint(self):
        batch_datapoints_one_hot_input, batch_datapoints_one_hot_label = next(iter(self.train_dataloader()))
        return batch_datapoints_one_hot_input[0], batch_datapoints_one_hot_label[0]


class _SimpleRegressionOneHot(_YieldSpreadPredictorOneHot):
    def __init__(self, batch_size, num_workers, train_data, test_data, label_encoders, categorical_features,train_val_split=TRAIN_VAL_SPLIT):
        super().__init__(batch_size, num_workers, train_data, test_data, label_encoders, categorical_features, train_val_split)
        self.linear = nn.Linear(self.first_layer_dimension, 1)

    def forward(self, x):
        x = x.view(-1, self.first_layer_dimension)
        x = self.linear(x)
        return x.double()    # return x.double() due to RuntimeError fixed by: https://stackoverflow.com/questions/67456368/pytorch-getting-runtimeerror-found-dtype-double-but-expected-float

    def get_model_name(self, loss_name):
        return f'{loss_name} Regression (One Hot) data size {self.size} and batch size {self.batch_size}'


class _L1Loss():
    def __init__(self):
        raise NotImplementedError('The __init__ method of _L1Loss should never be called.')

    def loss_func(self, predictions, labels):
        return l1_loss_func(predictions, labels).mean()


class _L2Loss():
    def __init__(self):
        raise NotImplementedError('The __init__ method of _L2Loss should never be called.')
        
    def loss_func(self, predictions, labels):
        return l2_loss_func(predictions, labels).mean()


class _CheckNumParameters():
    def __init__(self):
        raise NotImplementedError('The __init__ method of _CheckNumParameters should never be called.')

    '''Returns the number of trainable parameters in a model. From
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9?page=2.'''
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def check_num_parameters(self, threshold_percent=NUM_PARAMETERS_THRESHOLD_PERCENT):
        self.num_parameters = self.count_parameters()
        print(f'Number of parameters: {self.num_parameters}, Number of datapoints: {self.size}')
        if self.num_parameters >= threshold_percent / 100 * self.size:
            warnings.warn(f'Consider adding regularization as the number of trainable parameters is large. The number of parameters is about {round(self.num_parameters / self.size * 100)} % of the training data.')


'''An l1 regression model using one hot encoding implemented in PyTorch Lightning.'''
class L1RegressionOneHot(_SimpleRegressionOneHot, _L1Loss, _CheckNumParameters):
    def __init__(self, *args):
        super().__init__(*args)
        self.hyperparameters['loss_func'] = 'l1'    # used for wandb hyperparameter logging
        self.check_num_parameters()
    
    def __str__(self):
        return self.get_model_name('L1')


'''An l2 regression model using one hot encoding implemented in PyTorch Lightning.'''
class L2RegressionOneHot(_SimpleRegressionOneHot, _L2Loss, _CheckNumParameters):
    def __init__(self, *args):
        super().__init__(*args)
        self.hyperparameters['loss_func'] = 'l2'    # used for wandb hyperparameter logging
        self.check_num_parameters()

    def __str__(self):
        return self.get_model_name('L2')


'''Base class for models with embeddings on the categorical features.'''
class _YieldSpreadPredictorEmbeddings(_YieldSpreadPredictor):
    def __init__(self, *args):
        super().__init__(*args)
        self.hyperparameters['encoding'] = 'embeddings'    # used for wandb hyperparameter logging

    def setup_dataloaders(self, train_data, test_data, categorical_features, train_val_split):
        self.size = len(train_data)
        self.hyperparameters['num_train_datapoints'] = self.size    # used for wandb hyperparameter logging
        self.train_val_dataset = EmbeddingsDataset(train_data, categorical_features, label_feature='yield_spread')

        num_train_datapoints = int(self.size * train_val_split)
        num_val_datapoints = self.size - num_train_datapoints

        # train and val are flipped because the order of the data is is in descending order of trade_datetime (see query) and we want the training set to be earlier in time than the validation set
        train_val_split_func_caller = lambda array: train_val_split_func(array, train_size=num_val_datapoints, shuffle=False)
        val_inputs_categorical, train_inputs_categorical = train_val_split_func_caller(self.train_val_dataset.inputs_categorical)
        val_inputs_binary_and_continuous, train_inputs_binary_and_continuous = train_val_split_func_caller(self.train_val_dataset.inputs_binary_and_continuous)
        val_labels, train_labels = train_val_split_func_caller(self.train_val_dataset.labels)

        self.train_dataloader_object = self.create_dataloader_object(DatasetFromData(train_inputs_categorical, train_inputs_binary_and_continuous, train_labels), True)
        self.val_dataloader_object = self.create_dataloader_object(DatasetFromData(val_inputs_categorical, val_inputs_binary_and_continuous, val_labels), False)

        self.test_dataset = EmbeddingsDataset(test_data, categorical_features, label_feature='yield_spread') if test_data is not None else None    # enter else clause if test data is unlabeled

    def setup_embeddings(self, label_encoders):
        self.embeddings = nn.ModuleList()
        for column_name in self.train_val_dataset.column_names_for_embedding:
            self.embeddings.add_module(f'{column_name}_embedding', nn.Embedding(len(label_encoders[column_name].classes_), round(len(label_encoders[column_name].classes_) ** self.power)))    # the `add_module` procedure allows naming of the embedding layers which makes it easier to load the state dict
        self.total_embeddings_dimension = sum([embedding.embedding_dim for embedding in self.embeddings])

    def apply_embeddings(self, x_categorical, x_binary_and_continuous):
        x = [embedding(x_categorical[:, idx].long()) for idx, embedding in enumerate(self.embeddings)]    # .long() solves RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.DoubleTensor instead (while checking arguments for embedding)
        x = torch.cat(x, dim=1)
        x = torch.cat([x, x_binary_and_continuous], dim=1)
        return x.float()    # convert the tensor to have float values

    def step(self, batch, batch_idx, name):
        inputs_categorical, inputs_binary_and_continuous, labels = batch
        outputs = self(inputs_categorical, inputs_binary_and_continuous)
        loss = self.loss_func(outputs, labels)
        self.log(name, loss)
        return {name: loss}


'''Base class for models with hidden layers (linear layers with relu activation).'''
class _HiddenLayers():
    def __init__(self):
        raise NotImplementedError('The __init__ method of _HiddenLayers should never be called.')

    def layer_initializer(self, num_nodes_hidden_layer, num_hidden_layers, first_layer_dimension, batch_normalization):
        assert type(num_hidden_layers) == int and num_hidden_layers >= 0, f'The number of hidden layers should be a non-negative integer, but was provided as {num_hidden_layers}'
        if hasattr(self, 'first_layer_dimensions_lost'):    # this is to accommodate tied layers
            first_layer_dimension -= self.first_layer_dimensions_lost

        if type(num_nodes_hidden_layer) == int:
            num_nodes_hidden_layer = [num_nodes_hidden_layer] * num_hidden_layers
        self.num_nodes_hidden_layer = num_nodes_hidden_layer    # used in __str__
        self.hyperparameters['num_nodes_in_each_hidden_layer'] = self.num_nodes_hidden_layer    # used for wandb hyperparameter logging
        assert len(self.num_nodes_hidden_layer) == num_hidden_layers, f'There should be {num_hidden_layers} hidden layers, but the number of nodes has been provided for {len(self.num_nodes_hidden_layer)} of them'

        layers = OrderedDict()
        if num_hidden_layers == 0:
            layers['first'] = nn.Linear(first_layer_dimension, 1)
        else:
            layers['first'] = nn.Linear(first_layer_dimension, num_nodes_hidden_layer[0])
            layers['relu_0'] = nn.ReLU()
            for idx in range(num_hidden_layers - 1):
                if batch_normalization:
                    layers[f'batch_norm_{idx + 1}'] = nn.BatchNorm1d(num_nodes_hidden_layer[idx + 1], affine=False, track_running_stats=False)
                layers[f'hidden_{idx + 1}'] = nn.Linear(num_nodes_hidden_layer[idx], num_nodes_hidden_layer[idx + 1])
                layers[f'relu_{idx + 1}'] = nn.ReLU()
            layers['last'] = nn.Linear(num_nodes_hidden_layer[-1], 1)
        self.layers = nn.Sequential(layers)

    def apply_layers(self, x):
        return self.layers(x)


'''Base class for models that have a layer that has tied weights.'''
class _TiedLayer():
    def __init__(self):
        raise NotImplementedError('The __init__ method of _TiedLayer should never be called.')

    def tied_layer_initializer(self, binary_and_continuous_columns, groups_of_tied_features, num_output_nodes_tied_layer, pool):
        # e.g., groups = ((last_yield_spread, last_quantity, last_seconds_ago), (last_last_yield_spread, last_last_quantity, last_last_seconds_ago))
        assert pool in SUPPORTED_POOLING_FUNCTIONS, f'pool value must be in {SUPPORTED_POOLING_FUNCTIONS.keys()}'
        self.pool = pool
        self.hyperparameteres['pooling_architecture_of_tied_layer'] = pool
        self.hyperparameters['groups_of_tied_features'] = groups_of_tied_features    # used for wandb hyperparameter logging
        num_groups_tied_features = len(groups_of_tied_features)
        assert num_groups_tied_features > 0, f'`groups` needs to be a collection, but was instead passed in as {groups_of_tied_features}'
        assert num_groups_tied_features > 1, f'`groups` needs to have more than one group'
        num_features_in_group = len(groups_of_tied_features[0])
        for group in groups_of_tied_features:
            assert num_features_in_group == len(group), f'Number of features in each group should equal {num_features_in_group}, but one of the groups had the following features: {group}'
        
        self.tied_layer = nn.Linear(num_features_in_group, num_output_nodes_tied_layer, bias=False)
        self.num_output_nodes_tied_layer = num_output_nodes_tied_layer
        self.hyperparameters['output_dimension_of_tied_layer'] = num_output_nodes_tied_layer    # used for wandb hyperparameter logging
        column_to_index = {column: idx for idx, column in enumerate(binary_and_continuous_columns)}
        convert_columns_to_indices = lambda columns: [column_to_index[column] for column in columns]
        self.group_indices_list = [convert_columns_to_indices(group) for group in groups_of_tied_features]
        
        grouped_features = flatten(groups_of_tied_features)
        ungrouped_features = list(set(binary_and_continuous_columns) - set(grouped_features))
        self.ungrouped_features_indices = convert_columns_to_indices(ungrouped_features)

    def apply_embeddings(self, x_categorical, x_binary_and_continuous):    # when processing just trade history, the categorical features are empty since the categorical features come from the reference data
        x_tied_groups = [self.tied_layer(x_binary_and_continuous[:, group_indices].float()) for group_indices in self.group_indices_list]
        x_tied = SUPPORTED_POOLING_FUNCTIONS[self.pool](x_tied_groups, self.num_output_nodes_tied_layer)
        if x_categorical.nelement() != 0:    # makes sure there exists categorical features to perform embeddings on (no categorical features probably means that we are processing just trade history, but in any case, we do not need to perform embeddings)
            x_embedded = torch.cat([embedding(x_categorical[:, idx]) for idx, embedding in enumerate(self.embeddings)], dim=1)
            x = torch.cat([x_embedded, x_tied, x_binary_and_continuous[:, self.ungrouped_features_indices]], dim=1)
        return x.float() if x_categorical.nelement() != 0 else x_tied.float()    # convert the value types to float

    @staticmethod    # this is created as a static method because of circular dependence (in order to make this an instance method and use self.pool, we need to call tied_layer_initializer which needs to be called after layer_initializer which needs this method)
    def compute_first_layer_dimensions_lost(groups_of_tied_features, num_output_nodes_tied_layer, outputs_pooled):
        if not outputs_pooled:
            return len(groups_of_tied_features) * (len(groups_of_tied_features[0]) - num_output_nodes_tied_layer)
        else:
            return len(groups_of_tied_features) * len(groups_of_tied_features[0]) - num_output_nodes_tied_layer


'''An l1 loss NN with embedding layers on each of the categorical features.'''
class NNL1LossEmbeddings(_YieldSpreadPredictorEmbeddings, _HiddenLayers, _L1Loss, _CheckNumParameters):
    def __init__(self, 
                 batch_size, 
                 num_workers, 
                 train_data, 
                 test_data, 
                 label_encoders, 
                 categorical_features, 
                 num_nodes_hidden_layer=0, 
                 num_hidden_layers=0, 
                 train_val_split=TRAIN_VAL_SPLIT, 
                 power=0.5, 
                 batch_normalization=False):
        super().__init__(batch_size, num_workers)
        self.power = power
        self.hyperparameters['power'] = power    # used for wandb hyperparameter logging
        self.setup_dataloaders(train_data, test_data, categorical_features, train_val_split)
        self.setup_embeddings(label_encoders)
        first_layer_dimension = self.total_embeddings_dimension + len(self.train_val_dataset.column_names_for_binary_and_continuous_features)
        self.layer_initializer(num_nodes_hidden_layer, num_hidden_layers, first_layer_dimension, batch_normalization)
        self.check_num_parameters()

    def forward(self, x_categorical, x_binary_and_continuous):
        x = self.apply_embeddings(x_categorical, x_binary_and_continuous)
        return self.apply_layers(x)

    def __str__(self):
        return f'NN L1 Loss with embeddings (power={self.power}) with hidden layers {self.num_nodes_hidden_layer}'


'''An l1 loss NN with a single tied layer and pooling layer. This model should be used 
to process only the trade history.'''
class NNL1LossTied(_YieldSpreadPredictorEmbeddings, _HiddenLayers, _TiedLayer, _L1Loss, _CheckNumParameters):
    def __init__(self, 
                 batch_size, 
                 num_workers, 
                 train_trade_history_flattened, 
                 test_trade_history_flattened, 
                 groups_of_tied_features, 
                 num_output_nodes_tied_layer, 
                 num_nodes_hidden_layer=0, 
                 num_hidden_layers=0, 
                 pool='none', 
                 train_val_split=TRAIN_VAL_SPLIT, 
                 batch_normalization=False):
        super().__init__(batch_size, num_workers)
        self.setup_dataloaders(train_trade_history_flattened, test_trade_history_flattened, [], train_val_split)
        self.tied_layer_initializer(self.train_val_dataset.column_names_for_binary_and_continuous_features, groups_of_tied_features, num_output_nodes_tied_layer, pool)
        first_layer_dimension = len(groups_of_tied_features) * len(groups_of_tied_features[0])
        self.first_layer_dimensions_lost = _TiedLayer.compute_first_layer_dimensions_lost(groups_of_tied_features, num_output_nodes_tied_layer, pool != 'none')
        self.layer_initializer(num_nodes_hidden_layer, num_hidden_layers, first_layer_dimension, batch_normalization)
        self.check_num_parameters()

    def forward(self, x_categorical, x_binary_and_continuous):    # there are no categorical features and just specified here to match the EmbeddingsDataset
        x = _TiedLayer.apply_embeddings(self, x_categorical, x_binary_and_continuous)
        return self.apply_layers(x)

    def __str__(self):
        return f'NN L1 Loss with flattened trade history in a tied layer (pool={self.pool}) with hidden layers {self.num_nodes_hidden_layer}'
        

'''An l1 loss NN with embedding layers on each of the categorical features, along with 
a linear layer (w/o bias) with tied weights for the group of features representing 
each of the past trades.'''
class NNL1LossEmbeddingsTied(NNL1LossEmbeddings, _TiedLayer, _CheckNumParameters):
    def __init__(self, 
                 batch_size, 
                 num_workers, 
                 train_data, 
                 test_data, 
                 label_encoders, 
                 categorical_features, 
                 groups_of_tied_features, 
                 num_output_nodes_tied_layer, 
                 num_nodes_hidden_layer=0, 
                 num_hidden_layers=0, 
                 pool='none', 
                 train_val_split=TRAIN_VAL_SPLIT, 
                 power=0.5, 
                 batch_normalization=False):
        self.first_layer_dimensions_lost = _TiedLayer.compute_first_layer_dimensions_lost(groups_of_tied_features, num_output_nodes_tied_layer, pool != 'none')
        super().__init__(batch_size, 
                         num_workers, 
                         train_data, 
                         test_data, 
                         label_encoders, 
                         categorical_features, 
                         num_nodes_hidden_layer, 
                         num_hidden_layers, 
                         train_val_split, 
                         power, 
                         batch_normalization)
        self.tied_layer_initializer(self.train_val_dataset.column_names_for_binary_and_continuous_features, groups_of_tied_features, num_output_nodes_tied_layer, pool)
        self.check_num_parameters()
    
    def apply_embeddings(self, x_categorical, x_binary_and_continuous):
        return _TiedLayer.apply_embeddings(self, x_categorical, x_binary_and_continuous)

    def __str__(self):
        return f'NN L1 Loss with embeddings (power={self.power}) with flattened trade history in a tied layer (pool={self.pool}) and with hidden layers {self.num_nodes_hidden_layer}'


'''An l1 loss recurrent model which processes just the trade history.'''
class RecurrentL1Loss(_YieldSpreadPredictor, _L1Loss, _CheckNumParameters):
    def __init__(self, 
                 batch_size, 
                 num_workers, 
                 train_trade_history, 
                 test_trade_history, 
                 num_layers=1, 
                 hidden_size=1, 
                 recurrent_architecture='lstm', 
                 trade_history_column=TRADE_HISTORY, 
                 train_val_split=TRAIN_VAL_SPLIT):
        super().__init__(batch_size, num_workers)
        assert recurrent_architecture in SUPPORTED_RNN_ARCHITECTURES, f'{recurrent_architecture} is not supported by this model. This model only supports: {SUPPORTED_RNN_ARCHITECTURES.keys()}.'
        self.hyperparameters['recurrent_architecture'] = recurrent_architecture    # used for wandb hyperparameter logging

        self.num_layers = num_layers    # used in __str__
        self.hyperparameters['num_layers_recurrent_architecture'] = num_layers    # used for wandb hyperparameter logging
        self.hidden_size = hidden_size    # used in __str__
        self.hyperparameters['hidden_size_recurrent_architecture'] = hidden_size    # used for wandb hyperparameter logging
        self.setup_dataloaders(train_trade_history, test_trade_history, train_val_split, trade_history_column)

        self.recurrent_architecture = recurrent_architecture    # used for __str__
        self.recurrent = SUPPORTED_RNN_ARCHITECTURES[recurrent_architecture](self.num_features_per_trade, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, 1)
        self.check_num_parameters()

    def setup_dataloaders(self, train_trade_history, test_trade_history, train_val_split, trade_history_column):
        self.size = len(train_trade_history)
        self.hyperparameters['num_train_datapoints'] = self.size    # used for wandb hyperparameter logging
        train_val_dataset = TradeHistoryDataset(train_trade_history)
        num_train_datapoints = int(self.size * train_val_split)
        num_val_datapoints = self.size - num_train_datapoints
        self.num_features_per_trade = len(train_trade_history[trade_history_column[0]].iloc[0][0])
        
        val_inputs, train_inputs, val_labels, train_labels = train_val_split_func(train_val_dataset.inputs, train_val_dataset.labels, train_size=num_val_datapoints, shuffle=False)
        self.train_dataloader_object = self.create_dataloader_object(DatasetFromData(train_inputs, train_labels), True)
        self.val_dataloader_object = self.create_dataloader_object(DatasetFromData(val_inputs, val_labels), False)
        
        self.test_dataset = TradeHistoryDataset(test_trade_history) if test_trade_history is not None else None    # enter else clause if test data is unlabeled

    def forward(self, x):
        x = self.recurrent(x)[0][:, -1, :]    # the first index being `:` means we want the entire batch, the second index being `-1` means we want the output of the last layer, the third index being `:` means we want the entire vector output of the layer
        return self.linear(x)

    def __str__(self):
        return f'{self.recurrent_architecture} L1 Loss with {self.num_layers} layers and {self.hidden_size} hidden size'


'''An l1 loss NN with embedding layers on each of the categorical features combined with 
a recurrent model that processes the trade history. The parameters for the recurrent model 
are given by `num_layers_recurrent` and `hidden_size_recurrent`, while the architecture is 
given as a string in `recurrent_architecture`. All other parameters are for the feed 
forward network.'''
class NNL1LossEmbeddingsWithRecurrence(_YieldSpreadPredictorEmbeddings, _HiddenLayers, _L1Loss, _CheckNumParameters):
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
                 trade_history_column=TRADE_HISTORY, 
                 train_val_split=TRAIN_VAL_SPLIT, 
                 power=0.5, 
                 batch_normalization=False):
        assert recurrent_architecture in SUPPORTED_RNN_ARCHITECTURES, f'{recurrent_architecture} is not supported by this model. This model only supports: {SUPPORTED_RNN_ARCHITECTURES.keys()}.'
        super().__init__(batch_size, num_workers)
        self.recurrent_architecture = recurrent_architecture    # used for __str__
        self.hyperparameters['recurrent_architecture'] = recurrent_architecture    # used for wandb hyperparameter logging
        self.power = power
        self.hyperparameters['power'] = power    # used for wandb hyperparameter logging
        self.setup_dataloaders(train_data_with_trade_history, test_data_with_trade_history, categorical_features, train_val_split, trade_history_column)
        self.setup_embeddings(label_encoders)
        first_layer_dimension = self.total_embeddings_dimension + len(self.train_val_dataset.column_names_for_binary_and_continuous_features) + hidden_size_recurrent
        self.layer_initializer(num_nodes_hidden_layer, num_hidden_layers, first_layer_dimension, batch_normalization)
        self.recurrent = SUPPORTED_RNN_ARCHITECTURES[recurrent_architecture](self.num_features_per_trade, hidden_size_recurrent, num_layers_recurrent)
        self.hyperparameters['num_layers_recurrent_architecture'] = num_layers_recurrent    # used for wandb hyperparameter logging
        self.hyperparameters['hidden_size_recurrent_architecture'] = hidden_size_recurrent    # used for wandb hyperparameter logging

        self.check_num_parameters()

    def setup_dataloaders(self, train_data_with_trade_history, test_data_with_trade_history, categorical_features, train_val_split, trade_history_column):
        self.size = len(train_data_with_trade_history)
        self.hyperparameters['num_train_datapoints'] = self.size    # used for wandb hyperparameter logging
        num_train_datapoints = int(self.size * train_val_split)
        num_val_datapoints = self.size - num_train_datapoints

        train_data_without_trade_history = train_data_with_trade_history.drop(columns=trade_history_column)
        test_data_without_trade_history = test_data_with_trade_history.drop(columns=trade_history_column)
        trade_history_train = train_data_with_trade_history[trade_history_column + ['yield_spread']]
        trade_history_test = test_data_with_trade_history[trade_history_column + ['yield_spread']]

        self.num_features_per_trade = len(trade_history_train[trade_history_column[0]].iloc[0][0])
        self.train_val_dataset = EmbeddingsDataset(train_data_without_trade_history, categorical_features, label_feature='yield_spread')

        # train and val are flipped because the order of the data is is in descending order of trade_datetime (see query) and we want the training set to be earlier in time than the validation set
        train_val_split_func_caller = lambda array: train_val_split_func(array, train_size=num_val_datapoints, shuffle=False)
        val_inputs_categorical, train_inputs_categorical = train_val_split_func_caller(self.train_val_dataset.inputs_categorical)
        val_inputs_binary_and_continuous, train_inputs_binary_and_continuous = train_val_split_func_caller(self.train_val_dataset.inputs_binary_and_continuous)
        val_labels, train_labels = train_val_split_func_caller(self.train_val_dataset.labels)

        train_val_dataset_trade_history = TradeHistoryDataset(trade_history_train)
        val_inputs_trade_history, train_inputs_trade_history, = train_val_split_func_caller(train_val_dataset_trade_history.inputs)
        self.train_dataloader_object = self.create_dataloader_object(DatasetFromData(train_inputs_categorical, 
                                                                                     train_inputs_binary_and_continuous, 
                                                                                     train_inputs_trade_history, 
                                                                                     train_labels),
                                                                     True)
        self.val_dataloader_object = self.create_dataloader_object(DatasetFromData(val_inputs_categorical, 
                                                                                   val_inputs_binary_and_continuous, 
                                                                                   val_inputs_trade_history, 
                                                                                   val_labels), 
                                                                   False)

        test_dataset = EmbeddingsDataset(test_data_without_trade_history, categorical_features, label_feature='yield_spread')
        test_dataset_trade_history = TradeHistoryDataset(trade_history_test)
        self.test_dataset = DatasetFromData(test_dataset.inputs_categorical, 
                                            test_dataset.inputs_binary_and_continuous, 
                                            test_dataset_trade_history.inputs, 
                                            test_dataset.labels)

    def forward(self, x_categorical, x_binary_and_continuous, x_trade_history):
        x_embedded = self.apply_embeddings(x_categorical, x_binary_and_continuous)
        x_trade_history_processed = self.recurrent(x_trade_history)[0][:, -1, :]    # the first index being `:` means we want the entire batch, the second index being `-1` means we want the output of the last layer, the third index being `:` means we want the entire vector output of the layer
        x = torch.cat([x_embedded, x_trade_history_processed], dim=1)
        return self.apply_layers(x)

    def step(self, batch, batch_idx, name):
        inputs_categorical, inputs_binary_and_continuous, inputs_trade_history, labels = batch
        outputs = self(inputs_categorical, inputs_binary_and_continuous, inputs_trade_history)
        loss = self.loss_func(outputs, labels)
        self.log(name, loss)
        return {name: loss}

    def __str__(self):
        return f'NN L1 Loss with embeddings (power={self.power}) and hidden layers {self.num_nodes_hidden_layer} and recurrent architecture {self.recurrent_architecture}'


'''An l1 loss NN with embedding layers on each of the categorical features along with a 
linear layer (w/o bias) with tied weights for the group of features representing each of 
the past trades, combined with a recurrent model that processes the trade history. The 
parameters for the recurrent model are given by `num_layers_recurrent` and 
`hidden_size_recurrent`, while the architecture is given as a string in 
`recurrent_architecture`. All other parameters are for the feed forward network.'''
class NNL1LossEmbeddingsTiedWithRecurrence(NNL1LossEmbeddingsWithRecurrence, _TiedLayer):
    def __init__(self, 
                 batch_size, 
                 num_workers, 
                 train_data_with_flattened_trade_history_with_trade_history, 
                 test_data_with_flattened_trade_history_with_trade_history, 
                 label_encoders, 
                 categorical_features, 
                 groups_of_tied_features, 
                 num_output_nodes_tied_layer, 
                 num_nodes_hidden_layer=0, 
                 num_hidden_layers=0, 
                 num_layers_recurrent=1, 
                 hidden_size_recurrent=1, 
                 recurrent_architecture='lstm', 
                 trade_history_column=TRADE_HISTORY,  
                 pool='none', 
                 train_val_split=TRAIN_VAL_SPLIT, 
                 power=0.5, 
                 batch_normalization=False):
        self.first_layer_dimensions_lost = _TiedLayer.compute_first_layer_dimensions_lost(groups_of_tied_features, num_output_nodes_tied_layer, pool != 'none')
        super().__init__(batch_size, 
                         num_workers, 
                         train_data_with_flattened_trade_history_with_trade_history, 
                         test_data_with_flattened_trade_history_with_trade_history, 
                         label_encoders, 
                         categorical_features, 
                         num_nodes_hidden_layer, 
                         num_hidden_layers, 
                         num_layers_recurrent, 
                         hidden_size_recurrent, 
                         recurrent_architecture, 
                         trade_history_column, 
                         train_val_split, 
                         power, 
                         batch_normalization)
        self.tied_layer_initializer(self.train_val_dataset.column_names_for_binary_and_continuous_features, groups_of_tied_features, num_output_nodes_tied_layer, pool)
        self.check_num_parameters()

    def apply_embeddings(self, x_categorical, x_binary_and_continuous):
        return _TiedLayer.apply_embeddings(self, x_categorical, x_binary_and_continuous)

    def __str__(self):
        return f'NN L1 Loss with embeddings (power={self.power}) with flattened trade history in a tied layer (pool={self.pool}) and hidden layers {self.num_nodes_hidden_layer} and recurrent architecture {self.recurrent_architecture}'