import numpy as np

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor

import sys
sys.path.insert(0,'../')

from yield_spread_modle_mitas.train import l1_loss_func, l2_loss_func

from rating_model_mitas.datasets_pytorch import OneHotDataset


# Loss functions
mae_loss_pytorch = lambda predictions, labels: l1_loss_func(predictions, labels).mean()
mse_loss_pytorch = lambda predictions, labels: l2_loss_func(predictions, labels).mean()
mae_loss_pandas = lambda predictions, labels: np.mean(np.abs(labels - predictions))
mse_loss_pandas = lambda predictions, labels: np.mean((labels - predictions) ** 2)


'''This regression model has an l2 loss on the objective.'''
def l2_regression(label_encoders, features_to_one_hot_encode, train_data_encoded, test_data_encoded=None):
    one_hot_train_data = OneHotDataset(train_data_encoded, label_encoders, features_to_one_hot_encode, 'yield_spread')
    inputs = one_hot_train_data.inputs.numpy()
    labels = one_hot_train_data.labels.numpy()
    linreg = SklearnLinearRegression().fit(inputs, labels)
    if test_data_encoded is None:
        predictions = linreg.predict(inputs)
        mae_loss = mae_loss_pandas(predictions, labels)
        mse_loss = mse_loss_pandas(predictions, labels)
    else:
        one_hot_test_data = OneHotDataset(test_data_encoded, label_encoders, features_to_one_hot_encode, 'yield_spread')
        predictions = linreg.predict(one_hot_test_data.inputs.numpy())
        mae_loss = mae_loss_pandas(predictions, one_hot_test_data.labels.numpy())
        mse_loss = mse_loss_pandas(predictions, one_hot_test_data.labels.numpy())
    print(f'MAE for linear regression model: {mae_loss}')
    print(f'MSE for linear regression model: {mse_loss}')
    return mae_loss


'''This regression model has an l1 loss on the objective.'''
def l1_regression(label_encoders, features_to_one_hot_encode, train_data_encoded, test_data_encoded=None):
    one_hot_train_data = OneHotDataset(train_data_encoded, label_encoders, features_to_one_hot_encode, 'yield_spread')
    inputs = one_hot_train_data.inputs.numpy()
    labels = one_hot_train_data.labels.numpy()
    linreg = SklearnSGDRegressor('epsilon_insensitive', epsilon=0, max_iter=int(1e6), alpha=0).fit(inputs, labels)    # this model minimizes the absolute error: https://stackoverflow.com/questions/50392783/training-linear-models-with-mae-using-sklearn-in-python
    if test_data_encoded is None:
        predictions = linreg.predict(inputs)
        mae_loss = mae_loss_pandas(predictions, labels)
        mse_loss = mse_loss_pandas(predictions, labels)
    else:
        one_hot_test_data = OneHotDataset(test_data_encoded, label_encoders, features_to_one_hot_encode, 'yield_spread')
        predictions = linreg.predict(one_hot_test_data.inputs.numpy())
        test_labels = one_hot_test_data.labels.numpy()
        mae_loss = mae_loss_pandas(predictions, test_labels)
        mse_loss = mse_loss_pandas(predictions, test_labels)
    print(f'MAE for linear regression model: {mae_loss}')
    print(f'MSE for linear regression model: {mse_loss}')
    return mae_loss