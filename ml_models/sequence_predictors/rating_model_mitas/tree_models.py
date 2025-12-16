import numpy as np
import pandas as pd

from datasets_pytorch import data_to_inputs_and_labels
from train import _correct_exact, _correct_pm1, _correct_for_threshold

import lightgbm as lgb

import torch

import sys
sys.path.insert(0,'../')

from yield_spread_model_mitas.tree_models import _get_train_and_validation_data


'''Prints the accuracies for the `dataset` using the predictions from `model`. Returns 
the exact accuracy, +/- 1 accuracy, and the threshold accuracy respectively.'''
def get_all_accuracies_for_single_dataset(model, dataset, print_label, threshold=1/3, num_decimal_places=3):
    inputs, labels = data_to_inputs_and_labels(dataset, 'rating')
    num_datapoints = len(inputs)
    outputs = model.predict(inputs)
    _correct_for_threshold_func = _correct_for_threshold(threshold, outputs_are_probabilities=True)
    correct_exact_count, correct_pm1_count, correct_for_threshold_count = _correct_exact(labels, outputs).sum(), _correct_pm1(labels, outputs).sum(), _correct_for_threshold_func(labels, outputs).sum()
    correct_exact_percent, correct_pm1_percent, correct_for_threshold_percent = correct_exact_count / num_datapoints * 100, correct_pm1_count / num_datapoints * 100, correct_for_threshold_count / num_datapoints * 100
    print(f'{print_label} Accuracy: {round(correct_exact_percent, num_decimal_places)} %, Accuracy +/- 1: {round(correct_pm1_percent, num_decimal_places)} %, Accuracy for at least {round(threshold, 2) * 100} percent confidence: {round(correct_for_threshold_percent, num_decimal_places)} %')
    return correct_exact_percent, correct_pm1_percent, correct_for_threshold_percent


'''Prints the accuracies for each dataset in `datasets_and_print_labels` using the 
predictions from `model`. `datasets_and_print_labels` needs to be a collection of pairs 
where the first item in the pair is a pandas dataframe and the second item is the print 
label for that dataframe. Returns the accuracies respectively as a dictionary where 
the key is the print label for a dataset and the value is the exact accuracy, +/- 1 
accuracy, and the threshold accuracy of that dataset when `model` is used to make 
predictions for it.'''
def get_all_accuracies(model, datasets_and_print_labels, threshold=1/3, num_decimal_places=3):
    print_labels_and_accuracies = dict()
    BOUNDARY = 50
    print('#' * BOUNDARY)
    for dataset, print_label in datasets_and_print_labels:
        accuracies = get_all_accuracies_for_single_dataset(model, dataset, print_label, threshold, num_decimal_places)
        print_labels_and_accuracies[print_label] = accuracies
    print('#' * BOUNDARY)
    return print_labels_and_accuracies


'''Trains a lightgbm model using `train_data`, the names of the `categorical_features` and 
the hyperparameters of `num_leaves`, `max_depth`, `min_data_in_leaf`, and `feature_fraction`. 
The training procedure uses an early stopping criterion based on increasing validation loss. 
Returns the model trained and both the train and test losses. If there is no test data, 
then `test_data` should be `None`.'''
def train_lightgbm_model(train_data, 
                         test_data, 
                         categorical_features, 
                         num_ratings, 
                         num_leaves=100, 
                         num_trees=100, 
                         max_depth=17, 
                         min_data_in_leaf=20, 
                         feature_fraction=1.0, 
                         seed=1):
    train_inputs, train_labels, val_inputs, val_labels = _get_train_and_validation_data(train_data, 'rating')
    categorical_features_from_data = list(set(categorical_features).intersection(set(train_inputs.columns)))
    train_data_lgb = lgb.Dataset(train_inputs, label=train_labels, categorical_feature=categorical_features_from_data)
    val_data_lgb = lgb.Dataset(val_inputs, label=val_labels, categorical_feature=categorical_features_from_data)
    model = lgb.train({'objective': 'softmax',    # https://www.quora.com/Is-the-softmax-loss-the-same-as-the-cross-entropy-loss
                       'device_type': 'gpu' if torch.cuda.is_available() else 'cpu', 
                       'seed': seed, 
                       'num_class': num_ratings, 
                       'num_leaves': num_leaves, 
                       'num_trees': num_trees, 
                       'max_depth': max_depth, 
                       'min_data_in_leaf': min_data_in_leaf, 
                       'feature_fraction': feature_fraction}, 
                      train_data_lgb, 
                      valid_sets=[val_data_lgb], 
                      categorical_feature=categorical_features_from_data, 
                      callbacks=[lgb.early_stopping(stopping_rounds=5)])    # hyperparameters from previous notebook, stopping_rounds value of 5 chosen from example in quick start guide
    datasets_and_print_labels = ((train_data, 'Train'), (test_data, 'Test')) if test_data is not None else ((train_data, 'Train'),)
    accuracies = get_all_accuracies(model, datasets_and_print_labels)
    return model, accuracies


'''Compute cross entropy loss for `model` evaluated on `dataset`.'''
def get_cross_entropy_loss(model, dataset):
    TOLERANCE = 1e-12    # need to add this tolerance value to ensure that we do not take log of 0
    inputs, labels = data_to_inputs_and_labels(dataset, 'rating')
    probabilities = model.predict(inputs)
    probability_for_each_correct_label = probabilities[np.arange(len(probabilities)), labels]
    return -1 * np.sum(np.log(probability_for_each_correct_label + TOLERANCE))


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
                                               num_ratings, 
                                               num_leaves_candidates, 
                                               num_trees_candidates, 
                                               max_depth_candidates, 
                                               min_data_in_leaf_candidates, 
                                               feature_fraction_candidates, 
                                               seed=1, 
                                               print_results=False, 
                                               num_decimal_places=3):
    assert test_data is not None
    
    train_inputs, train_labels, val_inputs, val_labels = _get_train_and_validation_data(train_data, 'rating')
    categorical_features_from_data = list(set(categorical_features).intersection(set(train_inputs.columns)))
    train_data_lgb = lgb.Dataset(train_inputs, label=train_labels, categorical_feature=categorical_features_from_data, free_raw_data=False)    # need to set free_raw_data = False to solve dataset constructor error: https://stackoverflow.com/questions/53904948/how-to-change-lightgbm-parameters-when-it-is-running
    val_data_lgb = lgb.Dataset(val_inputs, label=val_labels, categorical_feature=categorical_features_from_data, free_raw_data=False)    # need to set free_raw_data = False to solve dataset constructor error: https://stackoverflow.com/questions/53904948/how-to-change-lightgbm-parameters-when-it-is-running
    
    best_model = None
    num_leaves_for_best_model = None
    num_trees_for_best_model = None
    max_depth_for_best_model = None
    min_data_in_leaf_for_best_model = None
    feature_fraction_for_best_model = None
    train_exact_accuracy_for_best_model = None
    train_cross_entropy_loss_for_best_model = None
    test_exact_accuracy_for_best_model = None
    test_cross_entropy_loss_for_best_model = float('inf')
    
    num_leaves_in_result_df = []
    num_trees_in_result_df = []
    max_depth_in_result_df = []
    min_data_in_leaf_in_result_df = []
    feature_fraction_in_result_df = []
    train_exact_accuracy_in_result_df = []
    test_exact_accuracy_in_result_df = []

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

                        model = lgb.train({'objective': 'softmax',    # https://www.quora.com/Is-the-softmax-loss-the-same-as-the-cross-entropy-loss
                                        'device_type': 'gpu' if torch.cuda.is_available() else 'cpu', 
                                        'seed': seed, 
                                        'num_class': num_ratings, 
                                        'num_leaves': num_leaves, 
                                        'num_trees': num_trees, 
                                        'max_depth': max_depth, 
                                        'min_data_in_leaf': min_data_in_leaf, 
                                        'feature_fraction': feature_fraction}, 
                                        train_data_lgb, 
                                        valid_sets=[val_data_lgb], 
                                        categorical_feature=categorical_features_from_data, 
                                        callbacks=[lgb.early_stopping(stopping_rounds=5)])    # stopping_rounds value of 5 chosen from example in quick start guide

                        train_cross_entropy_loss = get_cross_entropy_loss(model, train_data)
                        test_cross_entropy_loss = get_cross_entropy_loss(model, test_data)
                        accuracies = get_all_accuracies(model, ((train_data, 'Train'), (test_data, 'Test')))
                        train_exact_accuracy, test_exact_accuracy = accuracies['Train'][0], accuracies['Test'][0]
                        train_exact_accuracy_in_result_df.append(train_exact_accuracy)
                        test_exact_accuracy_in_result_df.append(test_exact_accuracy)
                        if test_cross_entropy_loss < test_cross_entropy_loss_for_best_model:
                            best_model = model
                            num_leaves_for_best_model = num_leaves
                            num_trees_for_best_model = num_trees
                            max_depth_for_best_model = max_depth
                            min_data_in_leaf_for_best_model = min_data_in_leaf
                            feature_fraction_for_best_model = feature_fraction
                            train_exact_accuracy_for_best_model = train_exact_accuracy
                            train_cross_entropy_loss_for_best_model = train_cross_entropy_loss
                            test_exact_accuracy_for_best_model = test_exact_accuracy
                            test_cross_entropy_loss_for_best_model = test_cross_entropy_loss

    result_df = pd.DataFrame({'num_leaves': num_leaves_in_result_df, 
                              'num_trees': num_trees_in_result_df, 
                              'max_depth': max_depth_in_result_df, 
                              'min_data_in_leaf': min_data_in_leaf_in_result_df, 
                              'feature_fraction': feature_fraction_in_result_df, 
                              'train_exact_accuracy': train_exact_accuracy_in_result_df, 
                              'test_exact_accuracy': test_exact_accuracy_in_result_df})

    print(f'train_exact_accuracy_for_best_model: {train_exact_accuracy_for_best_model}, test_exact_accuracy_for_best_model: {test_exact_accuracy_for_best_model}')

    if print_results:
        print(result_df)
    print(f'Best LightGBM model has `num_leaves`: {num_leaves_for_best_model}, `num_trees`: {num_trees_for_best_model}, `max_depth`: {max_depth_for_best_model}, `min_data_in_leaf`: {min_data_in_leaf_for_best_model}, `feature_fraction`: {feature_fraction_for_best_model}, Train Accuracy: {round(train_exact_accuracy_for_best_model, num_decimal_places)}, Test Accuracy: {round(test_exact_accuracy_for_best_model, num_decimal_places)}')
    return best_model, \
           {'num_leaves': num_leaves_for_best_model, 'num_trees': num_trees_for_best_model, 'max_depth': max_depth_for_best_model, 'min_data_in_leaf': min_data_in_leaf_for_best_model, 'feature_fraction': feature_fraction_for_best_model}, \
           {'Train': train_exact_accuracy_for_best_model, 'Test': test_exact_accuracy_for_best_model}, \
           result_df