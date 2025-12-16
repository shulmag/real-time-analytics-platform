import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

import sys
sys.path.insert(0,'../')

from rating_model_mitas.train import _correct_exact, _correct_pm1, _correct_for_threshold


def _get_accuracy_sklearn(model, inputs, labels, threshold):
    # predictions = model.predict(inputs)
    softmax_probabilities = model.predict_proba(inputs)
    total = labels.shape[0]
    correct_exact_count = _correct_exact(labels, softmax_probabilities).sum()
    correct_pm1_count = _correct_pm1(labels, softmax_probabilities).sum()    # if the prediction is +/- 1 from the correct label, then it counts as correct, e.g., if the correct label is AA, then any of the following predictions would count as correct: AA, AA+, AA-
    correct_for_threshold_func = _correct_for_threshold(threshold, outputs_are_probabilities=True)
    correct_for_threshold_count = correct_for_threshold_func(labels, softmax_probabilities).sum()    # if the probability of predicting the correct rating is at least at the threshold value, e.g., setting the threshold at 33%, if the correct label is AA, and the prediction has a 60% of BBB+ and 40% of AA, then it would count as correct

    return 100 * correct_exact_count / total, \
           100 * correct_pm1_count / total, \
           100 * correct_for_threshold_count / total


def _get_all_accuracies_sklearn(model, inputs, labels, test_inputs, test_labels, size, threshold=1/3, num_decimal_places=3):
    boundary = 50
    print('#' * boundary)
    print(f'Model (size = {size}): {model}')
    printer_string = lambda accuracies, label: f'{label} Accuracy: {round(accuracies[0].item(), num_decimal_places)} %, {label} Accuracy +/- 1: {round(accuracies[1].item(), num_decimal_places)} %, {label} Accuracy for at least {round(threshold, 2) * 100} percent confidence: {round(accuracies[2].item(), num_decimal_places)} %'
    print(printer_string(_get_accuracy_sklearn(model, inputs, labels, threshold), 'Train'))
    print(printer_string(_get_accuracy_sklearn(model, test_inputs, test_labels, threshold), 'Test'))
    print('#' * boundary)


def baseline(test_data):
    test_data_labels = test_data.labels.numpy()
    print(f'Test Baseline Accuracy (predicting the most popular label for all cases): {np.bincount(test_data_labels).max() / test_data_labels.size * 100} %')

def sklearn_logreg(data_list, test_data):
    baseline(test_data)
    test_data_inputs = test_data.inputs.numpy()
    test_data_labels = test_data.labels.numpy()
    for data in data_list:
        data_inputs = data.inputs.numpy()
        data_labels = data.labels.numpy()
        size = data_inputs.shape[0]
        logreg_sklearn = SklearnLogisticRegression(solver='liblinear', random_state=0, max_iter=int(1e6)).fit(data_inputs, data_labels)
        _get_all_accuracies_sklearn(logreg_sklearn, data_inputs, data_labels, test_data_inputs, test_data_labels, size)