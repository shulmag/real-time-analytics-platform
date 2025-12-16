import os
import time

from datetime import datetime

import pandas as pd

from ficc.utils.auxiliary_variables import IS_BOOKKEEPING, IS_SAME_DAY, NTBC_PRECURSOR, IS_REPLICA
from ficc.utils.adding_flags import add_bookkeeping_flag, add_same_day_flag, add_ntbc_precursor_flag, add_replica_flag

from adding_flags_v2 import add_all_flags, add_ntbc_precursor_flag_v4, add_all_flags_v2, add_all_flags_v3, add_bookkeeping_flag_v2, add_same_day_flag_v2, add_ntbc_precursor_flag_v2, add_replica_flag_v2, add_same_day_flag_v4, add_replica_flag_v4, add_bookkeeping_flag_v4, add_bookkeeping_flag_v5, add_same_day_flag_v5, add_ntbc_precursor_flag_v5, add_replica_flag_v5

import sys
sys.path.insert(0, '/Users/user/ficc/ficc/ml_models/sequence_predictors/')

from rating_model_mitas.data_prep import read_processed_file_pickle

FILENAME = '/Users/user/ficc/ficc/ml_models/sequence_predictors/data/processed_data_ficc_ycl_2021-12-31-23-59.pkl'
FLAGS = [IS_BOOKKEEPING, IS_SAME_DAY, NTBC_PRECURSOR, IS_REPLICA]


def load_file_and_create_datasets(filename=FILENAME):
    three_months_data = read_processed_file_pickle(filename)
    one_day_data = three_months_data[three_months_data['trade_datetime'] <= datetime(2021, 10, 2)]
    one_week_data = three_months_data[three_months_data['trade_datetime'] <= datetime(2021, 10, 8)]
    one_month_data = three_months_data[three_months_data['trade_datetime'] <= datetime(2021, 11, 1)]
    return one_day_data, one_week_data, one_month_data, three_months_data


def add_flags(data, save_filename=None):
    start_time = time.time()
    data = add_same_day_flag(data)
    data = add_ntbc_precursor_flag(data)
    data = add_replica_flag(data)
    data = add_bookkeeping_flag(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time} seconds')
    if save_filename != None:
        data.to_pickle(f'./files/{save_filename}_truth.pkl')
        data.to_csv(f'./files/{save_filename}_truth.csv')
    return data, elapsed_time


def add_flags_v2(data, save_filename=None, compare_filename=None):
    if compare_filename != None: assert os.path.exists(f'./files/{compare_filename}_truth.pkl'), 'No file to compare against'
    start_time = time.time()
    data = add_same_day_flag_v2(data)
    data = add_ntbc_precursor_flag_v2(data)
    data = add_replica_flag_v2(data)
    data = add_bookkeeping_flag_v2(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time} seconds')
    if save_filename != None:
        data.to_pickle(f'./files/{save_filename}.pkl')
        data.to_csv(f'./files/{save_filename}.csv')
    if compare_filename != None:
        truth = pd.read_pickle(f'./files/{compare_filename}_truth.pkl')
        for flag in FLAGS:
            assert data[flag].equals(truth[flag]), f'{flag} values are not equal'
        print(f'All flags match')
    return data, elapsed_time


def add_flags_v3(data, save_filename=None, compare_filename=None):
    if compare_filename != None: assert os.path.exists(f'./files/{compare_filename}_truth.pkl'), 'No file to compare against'
    start_time = time.time()
    data = add_all_flags_v2(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time} seconds')
    if save_filename != None:
        data.to_pickle(f'./files/{save_filename}.pkl')
        data.to_csv(f'./files/{save_filename}.csv')
    if compare_filename != None:
        truth = pd.read_pickle(f'./files/{compare_filename}_truth.pkl')
        for flag in FLAGS:
            assert data[flag].equals(truth[flag]), f'{flag} values are not equal'
        print(f'All flags match')
    return data, elapsed_time


def add_flags_v4(data, save_filename=None, compare_filename=None):
    if compare_filename != None: assert os.path.exists(f'./files/{compare_filename}_truth.pkl'), 'No file to compare against'
    start_time = time.time()
    data = add_all_flags_v3(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time} seconds')
    if save_filename != None:
        data.to_pickle(f'./files/{save_filename}.pkl')
        data.to_csv(f'./files/{save_filename}.csv')
    if compare_filename != None:
        truth = pd.read_pickle(f'./files/{compare_filename}_truth.pkl')
        for flag in FLAGS:
            assert data[flag].equals(truth[flag]), f'{flag} values are not equal'
        print(f'All flags match')
    return data, elapsed_time


def _test_flag(flag_name, data, save_filename, compare_filename):
    assert flag_name in FLAGS, f'{flag_name} is not in FLAGS'
    if compare_filename != None: assert os.path.exists(f'./files/{compare_filename}_truth.pkl'), 'No file to compare against'
    start_time = time.time()
    add_flag_function = (add_bookkeeping_flag_v4, add_same_day_flag_v4, add_ntbc_precursor_flag_v4, add_replica_flag_v4)[FLAGS.index(flag_name)]
    data = add_flag_function(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time} seconds')
    if save_filename != None:
        data.to_pickle(f'./files/{save_filename}.pkl')
        data.to_csv(f'./files/{save_filename}.csv')
    if compare_filename != None:
        truth = pd.read_pickle(f'./files/{compare_filename}_truth.pkl')
        assert data[flag_name].equals(truth[flag_name]), f'{flag_name} values are not equal'
        print(f'{flag_name} flag matches')
    return data, elapsed_time


test_same_day_flag = lambda data, save_filename=None, compare_filename=None: _test_flag(IS_SAME_DAY, data, save_filename, compare_filename)
test_ntbc_precursor_flag = lambda data, save_filename=None, compare_filename=None: _test_flag(NTBC_PRECURSOR, data, save_filename, compare_filename)
test_replica_flag = lambda data, save_filename=None, compare_filename=None: _test_flag(IS_REPLICA, data, save_filename, compare_filename)
test_bookkeeping_flag = lambda data, save_filename=None, compare_filename=None: _test_flag(IS_BOOKKEEPING, data, save_filename, compare_filename)


def create_ground_truth_datasets():
    one_day_data, one_week_data, one_month_data, three_months_data = load_file_and_create_datasets()
    _, elapsed_time_one_day = add_flags(one_day_data, 'one_day')
    _, elapsed_time_one_week = add_flags(one_week_data, 'one_week')
    _, elapsed_time_one_month = add_flags(one_month_data, 'one_month')
    _, elapsed_time_three_months = add_flags(three_months_data, 'three_months')
    with open('times.txt', 'w') as f:
        f.write(f'One day: {elapsed_time_one_day}\nOne week: {elapsed_time_one_week}\nOne month: {elapsed_time_one_month}\nThree months: {elapsed_time_three_months}\n')


def _test_on_data_subset(time_length):
    valid_time_lengths = ['one_day', 'one_week', 'one_month', 'three_months']
    assert time_length in valid_time_lengths, f'Time length of {time_length} is not a valid time length'
    argument_number = valid_time_lengths.index(time_length)
    data = load_file_and_create_datasets()[argument_number]
    for test_flag_function in (test_same_day_flag, test_ntbc_precursor_flag, test_replica_flag, test_bookkeeping_flag):
        test_flag_function(data, compare_filename=time_length)


test_one_day = lambda: _test_on_data_subset('one_day')
test_one_week = lambda: _test_on_data_subset('one_week')
test_one_month = lambda: _test_on_data_subset('one_month')
test_three_months = lambda: _test_on_data_subset('three_months')


if __name__ == '__main__':
    # create_ground_truth_datasets()
    # test_one_day()
    test_one_week()
    # test_one_month()
    # test_three_months()


# ##### Results #####
# Adding `observed=True` flag to the groupby commands does not cause a speed up; leaving for now, since Charles suggested it and noticed speed ups with it in his code for other situations
# First grouping by day and then by CUSIP did not help in speeding up the is_same_day flag procedure. Trying this idea now with all the flags at once (to reduce grouping overhead)
# Grouping by day and then by other features in a unified procedure caused a significant slowdown. Seems like multiple group by's slow thing down tremendously. `add_all_flags_v2` is very slow (3x slower than original)
# Using this line before calling the `is_same_day` procedure does not create any speedup: df = df[['trade_datetime', 'cusip', 'par_traded', 'trade_type']].copy()