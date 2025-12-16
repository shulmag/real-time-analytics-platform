import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split as train_test_split_func

import lightgbm as lgb
import catboost as cb

import wandb
from wandb.lightgbm import wandb_callback

import torch

import sys
sys.path.insert(0,'../')

from yield_spread_model_mitas.models import TRAIN_VAL_SPLIT
from yield_spread_model_mitas.train import l1_loss_func_numpy, l2_loss_func_numpy

from rating_model_mitas.datasets_pytorch import data_to_inputs_and_labels


LABEL_FEATURE = 'yield_spread'


'''Return predictions of `model` for a single `dataset` as a numpy array.'''
def get_predictions_for_single_dataset(model, dataset: pd.core.frame.DataFrame, label_feature=LABEL_FEATURE):
    return model.predict(dataset.drop(columns=[label_feature]))


'''Prints the l1 and l2 losses for the `dataset` using the predictions from `model`. Returns 
the l1 and l2 loss respectively.'''
def get_all_losses_for_single_dataset(model, dataset: pd.core.frame.DataFrame, print_label=None, num_decimal_places=3, verbose=True, label_feature=LABEL_FEATURE):
    if verbose: assert print_label != None, 'Since `verbose=True`, there must be a `print_label`'
    predictions = get_predictions_for_single_dataset(model, dataset, label_feature)
    labels = dataset[label_feature]
    l1_loss, l2_loss = np.mean(l1_loss_func_numpy(predictions, labels)), np.mean(l2_loss_func_numpy(predictions, labels))
    if verbose: print(f'{print_label} L1 loss: {round(l1_loss, num_decimal_places)}, L2 loss: {round(l2_loss, num_decimal_places)}')
    return l1_loss, l2_loss


'''Prints the l1 and l2 losses for each dataset in `datasets_and_print_labels` using the 
predictions from `model`. `datasets_and_print_labels` needs to be a collection of pairs 
where the first item in the pair is a pandas dataframe and the second item is the print 
label for that dataframe. Returns the l1 and l2 losses respectively as a dictionary where 
the key is the print label for a dataset and the value is the l1 and l2 loss of that 
dataset when `model` is used to make predictions for it.'''
def get_all_losses(model, datasets_and_print_labels, num_decimal_places=3, verbose=True, label_feature=LABEL_FEATURE):
    print_labels_and_losses = dict()
    BOUNDARY = 50
    if verbose: print('#' * BOUNDARY)
    for dataset, print_label in datasets_and_print_labels:
        losses = get_all_losses_for_single_dataset(model, dataset, print_label, num_decimal_places, verbose, label_feature)
        print_labels_and_losses[print_label] = losses
    if verbose: print('#' * BOUNDARY)
    return print_labels_and_losses


'''Get train and validation datasets from `train_data`.'''
def _get_train_and_validation_data(train_data, label_feature=LABEL_FEATURE, train_val_split=TRAIN_VAL_SPLIT):
    num_train_and_val = len(train_data)
    train_inputs, train_labels = data_to_inputs_and_labels(train_data, label_feature)
    num_train = int(num_train_and_val * train_val_split)
    num_val = num_train_and_val - num_train
    val_inputs, train_inputs, val_labels, train_labels = train_test_split_func(train_inputs, train_labels, train_size=num_val, shuffle=False)
    return train_inputs, train_labels, val_inputs, val_labels


'''Convert `categorial_features` in `df` with dtype object to dtype category to be used in 
the LightGBM model, and return the resulting dataframe.'''
def convert_columns_with_dtype_object_to_category(df, categorical_features, verbose=True):
    categorical_features_with_dtype_object = [column for column in categorical_features if df[column].dtype == 'object']
    if categorical_features_with_dtype_object:
        if verbose: print(f'Converting the following features to dtype category: {categorical_features_with_dtype_object}')
        df[categorical_features_with_dtype_object] = df[categorical_features_with_dtype_object].astype('category')    # converting multiple columns to `category` dtype in one line: https://stackoverflow.com/questions/28910851/python-pandas-changing-some-column-types-to-categories
    return df


'''Trains a lightgbm model using `train_data`, the names of the `categorical_features` and 
the hyperparameters of `num_leaves`, `max_depth`, `min_data_in_leaf`, and `feature_fraction`. 
The training procedure uses an early stopping criterion based on increasing validation loss. 
Returns the model trained and both the train and test losses. If there is no test data, 
then `test_data` should be `None`.'''
def train_lightgbm_model(train_data, 
                         test_data, 
                         categorical_features=None, 
                         num_leaves=100, 
                         num_trees=100, 
                         max_depth=17, 
                         min_data_in_leaf=20, 
                         feature_fraction=1.0, 
                         seed=1, 
                         wandb_logger_addon_string='', 
                         wandb_project='mitas_yield_spread'):
    if categorical_features is None:
        get_categorical_features = lambda df: [feature for feature, dtype in df.dtypes.to_dict().items() if dtype.name == 'category']
        train_categorical_features = get_categorical_features(train_data)
        test_categorical_features = get_categorical_features(test_data)
        assert set(train_categorical_features) == set(test_categorical_features)    # make sure that the categorical features from the train and test dataframes are the same
        categorical_features = train_categorical_features
    else:
        train_data = convert_columns_with_dtype_object_to_category(train_data, categorical_features)
        test_data = convert_columns_with_dtype_object_to_category(test_data, categorical_features, verbose=False)
    print(f'Categorical features used in the LightGBM model: {categorical_features}')

    train_inputs, train_labels, val_inputs, val_labels = _get_train_and_validation_data(train_data)
    categorical_features_from_data = list(set(categorical_features).intersection(set(train_inputs.columns)))
    train_data_lgb = lgb.Dataset(train_inputs, label=train_labels, categorical_feature=categorical_features_from_data)
    val_data_lgb = lgb.Dataset(val_inputs, label=val_labels, categorical_feature=categorical_features_from_data)

    wandb_callback_list = []    # have `wandb_callback()` inside here if `wandb_project` is not `None`
    if wandb_project is not None:
        wandb.init(project=wandb_project,
                entity='ficc-ai', 
                name=f'lgb{wandb_logger_addon_string}_num_leaves={num_leaves}_num_trees={num_trees}_max_depth={max_depth}_min_data_in_leaf={min_data_in_leaf}_feature_fraction={feature_fraction}')
        wandb_callback_list = [wandb_callback()]
    
    model = lgb.train({'objective': 'l1', 
                       'device_type': 'cpu',    # when functionality is fixed, add the following code before assigning 'cpu': 'gpu' if torch.cuda.is_available() else 
                       'seed': seed, 
                       'num_leaves': num_leaves, 
                       'num_trees': num_trees, 
                       'max_depth': max_depth, 
                       'min_data_in_leaf': min_data_in_leaf, 
                       'feature_fraction': feature_fraction}, 
                      train_data_lgb, 
                      valid_sets=[val_data_lgb], 
                      categorical_feature=categorical_features_from_data, 
                      callbacks=[lgb.early_stopping(stopping_rounds=5)] + wandb_callback_list)    # stopping_rounds value of 5 chosen from example in quick start guide
    
    datasets_and_print_labels = ((train_data, 'Train'), (test_data, 'Test')) if test_data is not None else ((train_data, 'Train'),)
    losses = get_all_losses(model, datasets_and_print_labels)
    
    if wandb_project is not None:
        if test_data is not None:
            wandb.log({'test_l1': losses['Test'][0]})
        wandb.finish()

    return model, losses


'''Trains a lightgbm model using `train_data`, the names of the `categorical_features` and 
each combination of hyperparameters from the candidates in `num_leaves_candidates`, `num_trees_candidates`, 
`max_depth_candidates`, `min_data_in_leaf_candidates`, and `feature_fraction_candidates`. 
The training procedure uses an early stopping criterion based on increasing validation loss. 
To determine model performance, the metric used is the l1 loss on `test_data`. Returns the 
best model found, the parameters as a dictionary, the train and test loss as a dictionary, 
and the dataframe storing the experiment information.'''
def train_lightgbm_model_hyperparameter_search(train_data, 
                                               test_data, 
                                               categorical_features, 
                                               num_leaves_candidates, 
                                               num_trees_candidates, 
                                               max_depth_candidates, 
                                               min_data_in_leaf_candidates, 
                                               feature_fraction_candidates, 
                                               seed=1, 
                                               print_results=False, 
                                               num_decimal_places=3, 
                                               wandb_logger_addon_string='', 
                                               wandb_project='mitas_yield_spread'):
    assert test_data is not None
    
    train_inputs, train_labels, val_inputs, val_labels = _get_train_and_validation_data(train_data)
    categorical_features_from_data = list(set(categorical_features).intersection(set(train_inputs.columns)))
    train_data_lgb = lgb.Dataset(train_inputs, label=train_labels, categorical_feature=categorical_features_from_data, free_raw_data=False)    # need to set free_raw_data = False to solve dataset constructor error: https://stackoverflow.com/questions/53904948/how-to-change-lightgbm-parameters-when-it-is-running
    val_data_lgb = lgb.Dataset(val_inputs, label=val_labels, categorical_feature=categorical_features_from_data, free_raw_data=False)    # need to set free_raw_data = False to solve dataset constructor error: https://stackoverflow.com/questions/53904948/how-to-change-lightgbm-parameters-when-it-is-running
    
    best_model = None
    num_leaves_for_best_model = None
    num_trees_for_best_model = None
    max_depth_for_best_model = None
    min_data_in_leaf_for_best_model = None
    feature_fraction_for_best_model = None
    train_l1_loss_for_best_model = float('inf')
    test_l1_loss_for_best_model = float('inf')
    
    num_leaves_in_result_df = []
    num_trees_in_result_df = []
    max_depth_in_result_df = []
    min_data_in_leaf_in_result_df = []
    feature_fraction_in_result_df = []
    train_l1_loss_in_result_df = []
    test_l1_loss_in_result_df = []

    for num_leaves in num_leaves_candidates:
        for num_trees in num_trees_candidates: 
            for max_depth in max_depth_candidates:
                for min_data_in_leaf in min_data_in_leaf_candidates:
                    for feature_fraction in feature_fraction_candidates:
                        num_leaves_in_result_df.append(num_leaves)
                        num_trees_in_result_df.append(num_trees)
                        max_depth_in_result_df.append(max_depth)
                        min_data_in_leaf_in_result_df.append(min_data_in_leaf)
                        feature_fraction_in_result_df.append(feature_fraction)

                        wandb.init(project=wandb_project, 
                                   entity='ficc-ai', 
                                   name=f'lgb{wandb_logger_addon_string}_num_leaves={num_leaves}_num_trees={num_trees}_max_depth={max_depth}_min_data_in_leaf={min_data_in_leaf}_feature_fraction={feature_fraction}')
                        model = lgb.train({'objective': 'l1', 
                                        'device_type': 'gpu' if torch.cuda.is_available() else 'cpu', 
                                        'seed': seed, 
                                        'num_leaves': num_leaves, 
                                        'num_trees': num_trees, 
                                        'max_depth': max_depth, 
                                        'min_data_in_leaf': min_data_in_leaf, 
                                        'feature_fraction': feature_fraction}, 
                                        train_data_lgb, 
                                        valid_sets=[val_data_lgb], 
                                        categorical_feature=categorical_features_from_data, 
                                        callbacks=[lgb.early_stopping(stopping_rounds=5),    # stopping_rounds value of 5 chosen from example in quick start guide
                                                   wandb_callback()])   
                        losses = get_all_losses(model, ((train_data, 'Train'), (test_data, 'Test')))
                        train_l1_loss, test_l1_loss = losses['Train'][0], losses['Test'][0]
                        wandb.log({'test_l1': test_l1_loss})
                        wandb.finish()

                        train_l1_loss_in_result_df.append(train_l1_loss)
                        test_l1_loss_in_result_df.append(test_l1_loss)

                        if test_l1_loss < test_l1_loss_for_best_model:
                            best_model = model
                            num_leaves_for_best_model = num_leaves
                            num_trees_for_best_model = num_trees
                            max_depth_for_best_model = max_depth
                            min_data_in_leaf_for_best_model = min_data_in_leaf
                            feature_fraction_for_best_model = feature_fraction
                            train_l1_loss_for_best_model = train_l1_loss
                            test_l1_loss_for_best_model = test_l1_loss

    result_df = pd.DataFrame({'num_leaves': num_leaves_in_result_df, 
                              'num_trees': num_trees_in_result_df, 
                              'max_depth': max_depth_in_result_df, 
                              'min_data_in_leaf': min_data_in_leaf_in_result_df, 
                              'feature_fraction': feature_fraction_in_result_df, 
                              'train_l1_loss': train_l1_loss_in_result_df, 
                              'test_l1_loss': test_l1_loss_in_result_df})
    if print_results:
        print(result_df)
    print(f'Best LightGBM model has `num_leaves`: {num_leaves_for_best_model}, `num_trees`: {num_trees_for_best_model}, `max_depth`: {max_depth_for_best_model}, `min_data_in_leaf`: {min_data_in_leaf_for_best_model}, `feature_fraction`: {feature_fraction_for_best_model}, Train L1 loss: {round(train_l1_loss_for_best_model, num_decimal_places)}, Test L1 loss: {round(test_l1_loss_for_best_model, num_decimal_places)}')
    return best_model, \
           {'num_leaves': num_leaves_for_best_model, 'num_trees': num_trees_for_best_model, 'max_depth': max_depth_for_best_model, 'min_data_in_leaf': min_data_in_leaf_for_best_model, 'feature_fraction': feature_fraction_for_best_model}, \
           {'Train': train_l1_loss_for_best_model, 'Test': test_l1_loss_for_best_model}, \
           result_df


'''Trains a CatBoost model using `train_data`, the names of the `categorical_features` and 
the hyperparameters of `max_depth`, `min_data_in_leaf`, and `num_trees`. The training 
procedure returns the best model at any point in training due to validation loss. Returning 
the best model is handled when `eval_set` is passed into `model.fit(...)`. Returns the 
model trained and both the train and test losses. If there is no test data, then `test_data` 
should be `None`.

We do not have a hyperparameter for `num_leaves` as the VM version of CatBoost does not 
support it since the tree growing strategy cannot have this constraint.
We do not have a hyperparameter for `feature_fraction` since CatBoost does not support it.'''
def train_catboost_model(train_data, 
                         test_data, 
                         categorical_features, 
                         max_depth=14, 
                         min_data_in_leaf=10, 
                         num_trees=1000, 
                         seed=1):
    categorical_features_from_data = list(set(categorical_features).intersection(set(train_data.columns)))
    train_inputs, train_labels, val_inputs, val_labels = _get_train_and_validation_data(train_data)
    train_data_cb = cb.Pool(train_inputs, train_labels, cat_features=categorical_features_from_data)
    val_data_cb = cb.Pool(val_inputs, val_labels, cat_features=categorical_features_from_data)
    model = cb.CatBoostRegressor(loss_function='MAE', 
                                 task_type='GPU' if torch.cuda.is_available() else 'CPU', 
                                 random_seed=seed, 
                                 max_depth=max_depth, 
                                 min_data_in_leaf=min_data_in_leaf, 
                                 num_trees=num_trees, 
                                 early_stopping_rounds=5)    # chosen to be the same as the LightGBM model
    model.fit(train_data_cb, eval_set=val_data_cb)
    datasets_and_print_labels = ((train_data, 'Train'), (test_data, 'Test')) if test_data is not None else ((train_data, 'Train'),)
    losses = get_all_losses(model, datasets_and_print_labels)
    return model, losses


'''Trains a catboost model using `train_data`, the names of the `categorical_features` and 
each combination of hyperparameters from the candidates in `max_depth_candidates`, 
`min_data_in_leaf_candidates`, and `num_trees_candidates`. The training procedure returns 
the best model at any point in training due to validation loss. Returning the best model 
is handled when `eval_set` is passed into `model.fit(...)`. To determine model performance, 
the metric used is the l1 loss on `test_data`. Returns the best model found, the parameters 
as a dictionary, the train and test loss as a dictionary, and the dataframe storing the 
experiment information.

We do not have a hyperparameter for `num_leaves` as the VM version of CatBoost does not 
support it since the tree growing strategy cannot have this constraint.
We do not have a hyperparameter for `feature_fraction` since CatBoost does not support it.'''
def train_catboost_model_hyperparameter_search(train_data, 
                                               test_data, 
                                               categorical_features, 
                                               max_depth_candidates, 
                                               min_data_in_leaf_candidates, 
                                               num_trees_candidates, 
                                               seed=1, 
                                               print_results=False, 
                                               num_decimal_places=3):
    assert test_data is not None
    
    categorical_features_from_data = list(set(categorical_features).intersection(set(train_data.columns)))
    train_inputs, train_labels, val_inputs, val_labels = _get_train_and_validation_data(train_data)
    train_data_cb = cb.Pool(train_inputs, train_labels, cat_features=categorical_features_from_data)
    val_data_cb = cb.Pool(val_inputs, val_labels, cat_features=categorical_features_from_data)

    best_model = None
    num_trees_for_best_model = None
    max_depth_for_best_model = None
    min_data_in_leaf_for_best_model = None
    train_l1_loss_for_best_model = float('inf')
    test_l1_loss_for_best_model = float('inf')
    
    num_trees_in_result_df = []
    max_depth_in_result_df = []
    min_data_in_leaf_in_result_df = []
    train_l1_loss_in_result_df = []
    test_l1_loss_in_result_df = []

    for num_trees in num_trees_candidates:
        for max_depth in max_depth_candidates:
            for min_data_in_leaf in min_data_in_leaf_candidates:
                max_depth_in_result_df.append(max_depth)
                min_data_in_leaf_in_result_df.append(min_data_in_leaf)
                num_trees_in_result_df.append(num_trees)

                model = cb.CatBoostRegressor(loss_function='MAE', 
                                             task_type='GPU' if torch.cuda.is_available() else 'CPU', 
                                             random_seed=seed, 
                                             max_depth=max_depth, 
                                             min_data_in_leaf=min_data_in_leaf, 
                                             num_trees=num_trees, 
                                             early_stopping_rounds=5)    # chosen to be the same as the LightGBM model
                model.fit(train_data_cb, eval_set=val_data_cb)

                losses = get_all_losses(model, ((train_data, 'Train'), (test_data, 'Test')))
                train_l1_loss, test_l1_loss = losses['Train'][0], losses['Test'][0]
                train_l1_loss_in_result_df.append(train_l1_loss)
                test_l1_loss_in_result_df.append(test_l1_loss)

                if test_l1_loss < test_l1_loss_for_best_model:
                    best_model = model
                    max_depth_for_best_model = max_depth
                    min_data_in_leaf_for_best_model = min_data_in_leaf
                    num_trees_for_best_model = num_trees
                    train_l1_loss_for_best_model = train_l1_loss
                    test_l1_loss_for_best_model = test_l1_loss

    result_df = pd.DataFrame({'max_depth': max_depth_in_result_df, 
                              'min_data_in_leaf': min_data_in_leaf_in_result_df, 
                              'num_trees': num_trees_in_result_df, 
                              'train_l1_loss': train_l1_loss_in_result_df, 
                              'test_l1_loss': test_l1_loss_in_result_df})
    if print_results:
        print(result_df)
    print(f'Best CatBoost model has `num_trees`: {num_trees_for_best_model}, `max_depth`: {max_depth_for_best_model}, `min_data_in_leaf`: {min_data_in_leaf_for_best_model}, Train L1 loss: {round(train_l1_loss_for_best_model, num_decimal_places)}, Test L1 loss: {round(test_l1_loss_for_best_model, num_decimal_places)}')
    return best_model, \
           {'max_depth': max_depth_for_best_model, 'min_data_in_leaf': min_data_in_leaf_for_best_model, 'num_trees': num_trees_for_best_model}, \
           {'Train': train_l1_loss_for_best_model, 'Test': test_l1_loss_for_best_model}, \
           result_df


'''Plots the feature importance based on a trained CatBoost model.'''
def plot_feature_importance_catboost(model, data, figsize):
    sorted_feature_importance = model.feature_importances_.argsort()
    fig, ax = plt.subplots(figsize=figsize)    # setting the figsize: https://stackoverflow.com/questions/14770735/how-do-i-change-the-figure-size-with-subplots
    bars = ax.barh(data.columns[sorted_feature_importance], model.feature_importances_[sorted_feature_importance])    # extracting the feature importances: https://towardsdatascience.com/catboost-regression-in-6-minutes-3487f3e5b329
    ax.bar_label(bars)    # putting labels on the bars to indicate their values: https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
    ax.set_xlabel('Feature importance')