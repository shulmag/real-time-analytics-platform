import numpy as np

import torch
from torch.utils.data import Dataset

# importing from parent directory: https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import sys
sys.path.insert(0,'../')

from rating_model_mitas.datasets_pytorch import data_to_inputs_and_labels


'''The `EmbeddingsDataset` class is used to create a DataLoader object that allows 
for PyTorch to iterate through the data when training the model, where the 
categorical features have been encoded with a dimension of $N_i^{1/2}$, where 
$N_i$ is the alphabet size for feature $i$. The difference between this class and 
the one of the same name in rating_model/datasets_pytorch.py is that this one 
does not distinguish between binary and continuous features.'''
class EmbeddingsDataset(Dataset):
    def __init__(self, data, categorical_features, label_feature='yield_spread'):
        inputs, labels = data_to_inputs_and_labels(data, label_feature)

        categorical_features_set = set(categorical_features)

        self.column_names_for_embedding = []
        self.column_names_for_binary_and_continuous_features = []
        for column in inputs.columns:
            if column in categorical_features_set:
                self.column_names_for_embedding.append(column)
            else:
                self.column_names_for_binary_and_continuous_features.append(column)

        self.inputs_categorical = torch.as_tensor(inputs[self.column_names_for_embedding].to_numpy())
        self.inputs_binary_and_continuous = torch.as_tensor(inputs[self.column_names_for_binary_and_continuous_features].to_numpy())
        self.labels = torch.as_tensor(labels.to_numpy())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs_categorical[idx], self.inputs_binary_and_continuous[idx], self.labels[idx]


'''The `TradeHistoryDataset` class is used to create a DataLoader object that allows for 
PyTorch to iterate through the trade history data when training the model, where the label 
is the yield spread contained in the array `yield_spreads`.'''
class TradeHistoryDataset(Dataset):
    def __init__(self, trade_history_and_yield_spreads):
        trade_history, yield_spreads = data_to_inputs_and_labels(trade_history_and_yield_spreads, 'yield_spread')
        num_trade_histories = trade_history.shape[1]    # trade_history.shape[1] indicates how many trade histories there are, index 0 is the number of datapoints
        inputs = [torch.as_tensor(np.stack(trade_history.iloc[:, idx].to_numpy(), axis=0)).float() for idx in range(num_trade_histories)]    # .iloc[:, idx] converts each column in the DataFrame into a series, then call np.stack to convert list of numpy arrays into a single numpy array; use .float() to avoid RuntimeError: expected scalar type Double but found Float
        self.inputs = inputs[0] if len(inputs) == 1 else inputs    # if there is a single item in the list, then remove it from the list
        self.labels = yield_spreads if torch.is_tensor(yield_spreads) else torch.as_tensor(yield_spreads.to_numpy())    # if yield_spreads is already in torch tensor form, then keep it that way, otherwise convert the pandas series to a torch tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if type(self.inputs) == list:
            trade_history_inputs_and_labels = [self.inputs[trade_history_idx][idx] for trade_history_idx in range(len(self.inputs))] + [self.labels[idx]]
            return trade_history_inputs_and_labels
        else:
            return self.inputs[idx], self.labels[idx]
