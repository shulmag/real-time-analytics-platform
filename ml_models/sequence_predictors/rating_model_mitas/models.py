import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split as train_test_split_func

import sys
sys.path.insert(0,'../')

from rating_model_mitas.datasets_pytorch import OneHotDataset, EmbeddingsDataset, DatasetFromData


TRAIN_VAL_SPLIT = 0.9


'''Base class for a classifier trying to predict ratings. Configures the optimizer to be ADAM and 
initializes the batch size and the number of workers. Assumes that the train and validation 
dataloaders will be attributes, and the test_dataset will be an attribute in the superclass.'''
class _RatingClassifier(pl.LightningModule):
    def __init__(self, batch_size, num_workers, num_ratings, loss_func_weights):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_ratings = num_ratings
        self.loss_func_weights = loss_func_weights

    def configure_optimizers(self):
        return optim.Adam(self.parameters())

    def loss_func(self, outputs, labels):
        return F.cross_entropy(outputs, labels) if self.loss_func_weights is None else self._loss_func_with_weights(outputs, labels)

    def _loss_func_with_weights(self, outputs, labels):
        num_weights = len(self.loss_func_weights)
        assert num_weights == 3
        loss_centered = F.cross_entropy(outputs, labels, reduction='none')
        
        labels_plus1 = torch.minimum(labels + 1, torch.ones(self.batch_size, device=self.device) * (self.num_ratings - 1)).long()    # torch.minimum(...) ensures that the indices do not go out of bounds
        loss_plus1 = F.cross_entropy(outputs, labels_plus1, reduction='none')

        labels_minus1 = torch.maximum(labels - 1, torch.zeros(self.batch_size, device=self.device)).long()    # torch.maximum(...) ensures that the indices do not go out of bounds
        loss_minus1 = F.cross_entropy(outputs, labels_minus1, reduction='none')
        
        losses = [loss_minus1, loss_centered, loss_plus1]
        return torch.mean(sum([losses[i] * self.loss_func_weights[i] for i in range(num_weights)]))

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)['loss']
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)['loss']
        self.log('test_loss', loss)
        return {'test_loss': loss}

    def train_dataloader(self):
        return self.train_dataloader_object

    def val_dataloader(self):
        return self.val_dataloader_object

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers) if self.test_dataset is not None else None

    def get_weights_addon_string(self):
        weights_addon_string = ''
        if self.loss_func_weights is not None:
            weights_addon_string = f'with loss function weights {self.loss_func_weights}'
        return weights_addon_string


'''Base class for a NN classifer trying to predict ratings using one hot encoding. Assumes that 
the categorical features will be the ones that will be one hot encoded.'''
class _RatingClassifierOneHot(_RatingClassifier):
    def __init__(self, batch_size, num_workers, train_data, test_data, encoders, binary_features, categorical_features, num_ratings, loss_func_weights=None, train_val_split=TRAIN_VAL_SPLIT):
        super().__init__(batch_size, num_workers, num_ratings, loss_func_weights)
        features_to_one_hot_encode = categorical_features
        self.setup_dataloaders(train_data, test_data, encoders, features_to_one_hot_encode, train_val_split)

        single_datapoint_input, _ = self.get_single_datapoint()
        self.first_layer_dimension = len(single_datapoint_input)

    def setup_dataloaders(self, train_data, test_data, encoders, features_to_one_hot_encode, train_val_split):
        self.size = len(train_data)
        train_val_dataset = OneHotDataset(train_data, encoders, features_to_one_hot_encode, 'rating')
        num_train_datapoints = int(self.size * train_val_split)
        num_val_datapoints = self.size - num_train_datapoints

        # train and val are flipped because the order of the data is is in descending order of trade date (see query) and we want the training set to be earlier in time than the validation set
        val_inputs, train_inputs, val_labels, train_labels = train_test_split_func(train_val_dataset.inputs, train_val_dataset.labels, train_size=num_val_datapoints, shuffle=False)

        self.train_dataloader_object = torch.utils.data.DataLoader(DatasetFromData(train_inputs, train_labels), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_dataloader_object = torch.utils.data.DataLoader(DatasetFromData(val_inputs, val_labels), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        self.test_dataset = OneHotDataset(test_data, encoders, features_to_one_hot_encode, 'rating') if test_data is not None else None    # enter else clause if test data is unlabeled

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_func(outputs, labels)
        self.log('loss', loss)
        return {'loss': loss}

    def get_single_datapoint(self):
        batch_datapoints_one_hot_input, batch_datapoints_one_hot_label = next(iter(self.train_dataloader()))
        return batch_datapoints_one_hot_input[0], batch_datapoints_one_hot_label[0]


'''A logistic regression model using one hot encoding implemented in PyTorch Lightning.'''
class LogisticRegressionOneHot(_RatingClassifierOneHot):
    def __init__(self, batch_size, num_workers, train_data, test_data, encoders, binary_features, categorical_features, num_ratings, loss_func_weights=None, train_val_split=TRAIN_VAL_SPLIT):
        super().__init__(batch_size, num_workers, train_data, test_data, encoders, binary_features, categorical_features, num_ratings, loss_func_weights, train_val_split)
        self.linear = nn.Linear(self.first_layer_dimension, num_ratings)

    def forward(self, x):
        x = x.view(-1, self.first_layer_dimension)
        x = self.linear(x)    # do not use softmax activation in the final layer since cross entropy loss automatically applies a softmax activation
        return x

    def __str__(self):
        return f'Logistic Regression (One Hot) data size {self.size} and batch size {self.batch_size} {self.get_weights_addon_string()}'


'''Base class for models with hidden layers (linear layers with relu activation).'''
class _HiddenLayers():
    def __init__(self):
        raise NotImplementedError('The __init__ method of _HiddenLayers should never be called.')

    def layer_initializer(self, num_nodes_hidden_layer, num_hidden_layers, first_layer_dimension, last_layer_dimension):
        self.num_nodes_hidden_layer = num_nodes_hidden_layer
        self.num_hidden_layers = num_hidden_layers

        self.layers = [nn.Linear(first_layer_dimension, num_nodes_hidden_layer), nn.ReLU()] \
                      + [nn.Linear(num_nodes_hidden_layer, num_nodes_hidden_layer), nn.ReLU()] * (num_hidden_layers - 1) \
                      + [nn.Linear(num_nodes_hidden_layer, last_layer_dimension)]
        self.hidden_layers = nn.Sequential(*self.layers)

    def apply_layers(self, x):
        return self.hidden_layers(x)


'''Class for a NN classifer with hidden layers trying to predict ratings using one hot encoding.'''
class NNCrossEntropyOneHotHiddenLayers(_RatingClassifierOneHot, _HiddenLayers):
    def __init__(self, batch_size, num_workers, train_data, test_data, encoders, binary_features, categorical_features, num_ratings, num_nodes_hidden_layer, num_hidden_layers, loss_func_weights=None, train_val_split=TRAIN_VAL_SPLIT):
        super().__init__(batch_size, num_workers, train_data, test_data, encoders, binary_features, categorical_features, num_ratings, loss_func_weights, train_val_split)
        self.layer_initializer(num_nodes_hidden_layer, num_hidden_layers, self.first_layer_dimension, num_ratings)

    def forward(self, x):
        x = x.view(-1, self.first_layer_dimension)
        return self.apply_layers(x)

    def __str__(self):
        return f'NN with Cross Entropy Loss (One Hot): {self.num_hidden_layers} hidden layer(s) with {self.num_nodes_hidden_layer} nodes in each hidden layer, data size {self.size} and batch size {self.batch_size} {self.get_weights_addon_string()}'


'''Base class for a NN classifer trying to predict ratings using embeddings. Assumes that 
the categorical features will be the ones that will embedded.'''
class _RatingClassifierEmbedding(_RatingClassifier):
    def __init__(self, batch_size, num_workers, train_data, test_data, encoders, binary_features, categorical_features, num_ratings, loss_func_weights, train_val_split, power):
        super().__init__(batch_size, num_workers, num_ratings, loss_func_weights)
        self.power = power
        self.setup_dataloaders(train_data, test_data, encoders, binary_features, categorical_features, train_val_split)

    def setup_dataloaders(self, train_data, test_data, encoders, binary_features, categorical_features, train_val_split):
        self.size = len(train_data)
        train_val_dataset = EmbeddingsDataset(train_data, binary_features, categorical_features)

        self.embeddings = nn.ModuleList([nn.Embedding(len(encoders[column_name].classes_), round(len(encoders[column_name].classes_) ** self.power)) for column_name in train_val_dataset.column_names_for_embedding])
        total_embeddings_dimension = sum([embedding.embedding_dim for embedding in self.embeddings])
        self.first_layer_dimension = total_embeddings_dimension + len(train_val_dataset.column_names_for_binary_features) + len(train_val_dataset.column_names_for_continuous_features)

        num_train_datapoints = int(self.size * train_val_split)
        num_val_datapoints = self.size - num_train_datapoints
        
        # train and val are flipped because the order of the data is is in descending order of trade date (see query) and we want the training set to be earlier in time than the validation set
        train_test_split_func_caller = lambda array: train_test_split_func(array, train_size=num_val_datapoints, shuffle=False)
        val_inputs_categorical, train_inputs_categorical = train_test_split_func_caller(train_val_dataset.inputs_categorical)
        val_inputs_binary, train_inputs_binary = train_test_split_func_caller(train_val_dataset.inputs_binary)
        val_inputs_continuous, train_inputs_continuous = train_test_split_func_caller(train_val_dataset.inputs_continuous)
        val_labels, train_labels = train_test_split_func_caller(train_val_dataset.labels)
        
        self.train_dataloader_object = torch.utils.data.DataLoader(DatasetFromData(train_inputs_categorical, train_inputs_binary, train_inputs_continuous, train_labels), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_dataloader_object = torch.utils.data.DataLoader(DatasetFromData(val_inputs_categorical, val_inputs_binary, val_inputs_continuous, val_labels), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        self.test_dataset = EmbeddingsDataset(test_data, binary_features, categorical_features) if test_data is not None else None    # enter else clause if test data is unlabeled
    
    def apply_embeddings(self, x_categorical, x_binary, x_continuous):
        x = [embedding(x_categorical[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        x = torch.cat(x, dim=1)
        x = torch.cat([x, x_binary, x_continuous], dim=1)
        return x.float()    # convert the tensor to float
    
    def training_step(self, batch, batch_idx):
        inputs_categorical, inputs_binary, inputs_continuous, labels = batch
        outputs = self(inputs_categorical, inputs_binary, inputs_continuous)
        loss = self.loss_func(outputs, labels)
        self.log('loss', loss)
        return {'loss': loss}


'''A logistic regression model using embeddings implemented in PyTorch Lightning.'''
class LogisticRegressionEmbedding(_RatingClassifierEmbedding):
    def __init__(self, batch_size, num_workers, train_data, test_data, encoders, binary_features, categorical_features, num_ratings, loss_func_weights=None, train_val_split=TRAIN_VAL_SPLIT, power=0.5):
        super().__init__(batch_size, num_workers, train_data, test_data, encoders, binary_features, categorical_features, num_ratings, loss_func_weights, train_val_split, power)
        self.linear = nn.Linear(self.first_layer_dimension, num_ratings)

    def forward(self, x_categorical, x_binary, x_continuous):
        x = self.apply_embeddings(x_categorical, x_binary, x_continuous)
        x = self.linear(x)
        return x

    def __str__(self):
        return f'Logistic Regression (Embeddings: power={self.power}) data size {self.size} and batch size {self.batch_size} {self.get_weights_addon_string()}'


'''Class for a NN classifer with hidden layers trying to predict ratings using embeddings.'''
class NNCrossEntropyEmbeddingsHiddenLayers(_RatingClassifierEmbedding, _HiddenLayers):
    def __init__(self, batch_size, num_workers, train_data, test_data, encoders, binary_features, categorical_features, num_ratings, num_nodes_hidden_layer, num_hidden_layers, loss_func_weights=None, train_val_split=TRAIN_VAL_SPLIT, power=0.5):
        super().__init__(batch_size, num_workers, train_data, test_data, encoders, binary_features, categorical_features, num_ratings, loss_func_weights, train_val_split, power)
        self.layer_initializer(num_nodes_hidden_layer, num_hidden_layers, self.first_layer_dimension, num_ratings)

    def forward(self, x_categorical, x_binary, x_continuous):
        x = self.apply_embeddings(x_categorical, x_binary, x_continuous)
        return self.apply_layers(x)

    def __str__(self):
        return f'NN with Cross Entropy Loss (Embeddings: power={self.power}): {self.num_hidden_layers} hidden layer(s) with {self.num_nodes_hidden_layer} nodes in each hidden layer, data size {self.size} and batch size {self.batch_size} {self.get_weights_addon_string()}'
