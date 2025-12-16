'''
'''
import time
import multiprocess as mp

import numpy as np

from modules.test.auxiliary_variables import QUANTITY, TRADE_TYPE, DIRECTORY
from modules.test.auxiliary_functions import response_from_individual_pricing, response_from_batch_pricing, check_that_batch_pricing_gives_output_for_all_cusips


NUM_TIMES_TEST_SHOULD_BE_REPEATED = 100


def call_repeatedly_in_parallel(func, args_list, num_calls=NUM_TIMES_TEST_SHOULD_BE_REPEATED):
    '''Call `func` with arguments in `args_list` for `num_calls` times and put all of these 
    results into a returned list.'''
    # return [func(*args_list) for _ in range(num_calls)]
    with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
        results = pool_object.map(lambda run_idx: func(*args_list, run_idx), range(num_calls))
    return results


def test_individual_pricing():
    '''Get speed of individually pricing a CUSIP.'''
    cusip = '64971XQM3'

    def time_for_single_test(cusip, run_idx):    # `run_idx` argument is to match that of `time_for_batch_pricing_first_k_cusips`
        start_time = time.time()
        response_dict = response_from_individual_pricing(cusip, TRADE_TYPE, QUANTITY)
        time_elapsed = time.time() - start_time
        assert 'error' not in response_dict, f'For CUSIP {cusip}, pricing should not have an error, but has an error: {response_dict["error"]}'
        return time_elapsed
    
    results = call_repeatedly_in_parallel(time_for_single_test, [cusip])
    print(f'For individual pricing, avg speed: {round(np.mean(results), 5)} seconds with std dev: {round(np.std(results), 5)} seconds, for {len(results)} runs')


def _get_num_cusips_list(max_num_cusips):
    '''Get a list of numbers going like 1, 3, 10, 30, 100, 300, ... up to `max_num_cusips`.'''
    powers_of_10_less_than_cusips = [10 ** i for i in range(int(np.log10(max_num_cusips + 1)) + 1)]    # add 1 to `max_num_cusips` prevent floating point issues where log(1000) becomes 2.9999...
    num_cusips_list = sorted(powers_of_10_less_than_cusips + [3 * item for item in powers_of_10_less_than_cusips])
    if num_cusips_list[-1] > max_num_cusips: num_cusips_list.pop()
    return num_cusips_list


def _test_batch_pricing_file(filename_wo_directory):
    '''Get speed of batch pricing on `filename_wo_directory`.'''
    filename = f'{DIRECTORY}/{filename_wo_directory}'
    all_cusips = np.loadtxt(filename, delimiter=',', dtype=str)

    def time_for_batch_pricing_first_k_cusips(num_cusips, run_idx):
        cusips = all_cusips[:num_cusips].tolist()    # need to use .tolist() so that cusips is a list type not a numpy array type due to the first `if` statement in `response_from_batch_pricing`
        start_time = time.time()
        request_obj = response_from_batch_pricing(f'speed_test_{run_idx}.csv', cusips)    # need to have `run_idx` appended to filename otherwise the files collide with different parallel runs of the function since they have the same name
        time_elapsed = time.time() - start_time
        try:
            check_that_batch_pricing_gives_output_for_all_cusips(request_obj, cusips)
        except AssertionError as e:    # enters this `except` statement if batch pricing returns the following: 'You have been logged out due to a period of inactivity. Refresh the page!'
            print(f'Retrying batch pricing because of the following error: {e}')
            time.sleep(1)    # have a one second delay to prevent overloading the server
            return time_for_batch_pricing_first_k_cusips(num_cusips, run_idx)
        return time_elapsed

    num_cusips_list = _get_num_cusips_list(len(all_cusips))
    speed_test_results_filename = f'{DIRECTORY}/speed_test.txt'
    open(speed_test_results_filename, 'w').close()    # clear the file
    for num_cusips in num_cusips_list:
        results = call_repeatedly_in_parallel(time_for_batch_pricing_first_k_cusips, [num_cusips])
        results_as_text = f'For batch pricing {num_cusips} CUSIPs, mean speed: {round(np.mean(results), 3)} seconds with std dev: {round(np.std(results), 3)} seconds, median speed: {round(np.median(results), 3)} seconds, for {len(results)} runs'
        with open(speed_test_results_filename, 'a') as file:
            file.write(results_as_text + '\n')
        print(results_as_text)


def test_batch_pricing5_25_10000():
    '''Get speed of batch pricing on 5-25_10000.csv.'''
    return _test_batch_pricing_file('5-25_10000.csv')


def test_batch_pricing8_3_10000():
    '''Get speed of batch pricing on 8-3_10k.csv.'''
    return _test_batch_pricing_file('8-3_10k.csv')


def test_batch_pricing_64971XQM3():
    '''Get speed of batch pricing with a file with only CUSIP 64971XQM3.'''
    return _test_batch_pricing_file('64971XQM3_10000.csv')


def test_batch_pricing_508642DQ5():
    '''Get speed of batch pricing with a file with only CUSIP 508642DQ5. 508642DQ5 is no longer outstanding.'''
    return _test_batch_pricing_file('508642DQ5_10000.csv')


def test_batch_pricing_181234J55():
    '''Get speed of batch pricing with a file with only CUSIP 181234J55. 181234J55 is not supported.'''
    return _test_batch_pricing_file('181234J55_10000.csv')


def test_batch_pricing_084145AF8():
    '''Get speed of batch pricing with a file with only CUSIP 084145AF8. 084145AF8 has defaulted.'''
    return _test_batch_pricing_file('084145AF8_10000.csv')


def test_batch_pricing_862369DC6():
    '''Get speed of batch pricing with a file with only CUSIP 862369DC6. 862369DC6 has missing or negative reported yields.'''
    return _test_batch_pricing_file('862369DC6_10000.csv')


def test_batch_pricing_CUSIP():
    '''Get speed of batch pricing with a file with only CUSIP CUSIP. CUSIP is invalid.'''
    return _test_batch_pricing_file('CUSIP_10000.csv')


def test_batch_pricing_79165TNT4():
    '''Get speed of batch pricing with a file with only CUSIP 79165TNT4. 79165TNT4 is maturing very soon or has already matured.'''
    return _test_batch_pricing_file('79165TNT4_10000.csv')


def test_batch_pricing_455054BH8():
    '''Get speed of batch pricing with a file with only CUSIP 455054BH8. 455054BH8 has an irregular/variable coupon rate or interest payment frequency.'''
    return _test_batch_pricing_file('455054BH8_10000.csv')


def test_batch_pricing_25483WAG7():
    '''Get speed of batch pricing with a file with only CUSIP 25483WAG7. 25483WAG7 has abnormally high (greater than 10%) reported yields.'''
    return _test_batch_pricing_file('25483WAG7_10000.csv')
