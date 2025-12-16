'''
'''
import time

import numpy as np

from modules.test.auxiliary_variables import DIRECTORY
from modules.test.auxiliary_functions import run_multiple_times_before_failing, response_from_batch_pricing, check_that_batch_pricing_gives_output_for_all_cusips, get_cusip_list_quantity_list_tradetype_list


def test_find_bad_cusips():
    '''If there is a CUSIP that is causing a batch pricing error for a large batch, this code can find it. 
    The FILENAME.csv file must be copied into the files/ directory.'''
    FILENAME = '6-12_50k'
    filename = f'{DIRECTORY}/{FILENAME}.csv'
    cusip_list, quantity_list, trade_type_list = get_cusip_list_quantity_list_tradetype_list(filename)

    check_that_batch_pricing_gives_output_for_all_cusips_run_multiple_times = run_multiple_times_before_failing(check_that_batch_pricing_gives_output_for_all_cusips)
    none_found = True    # boolean variable denoting whether a bad CUSIP was found; NOTE: once a bad CUSIP is found in the first pass, we know it exists, and thus, do not need to re-initialize `none_found`
    start, end = 0, len(cusip_list)
    chunk_size = 10 ** int(np.log10(len(cusip_list) - 1))    # largest power of 10 that is less than the (total number of CUSIPs - 1)
    while chunk_size >= 1:
        for start_idx in range(start, end, chunk_size):
            print(start_idx, min(start_idx + chunk_size, end))
            cusip_list_chunk = cusip_list[start_idx : start_idx + chunk_size]
            request_obj = response_from_batch_pricing(f'{DIRECTORY}/{FILENAME}_pricing.csv', cusip_list_chunk, quantity_list[start_idx : start_idx + chunk_size], trade_type_list[start_idx : start_idx + chunk_size])
            time.sleep(1)    # minor delay to not overload the server and get errors
            try:
                check_that_batch_pricing_gives_output_for_all_cusips_run_multiple_times(request_obj, cusip_list_chunk)
            except AssertionError:
                start = start_idx
                end = min(start_idx + chunk_size, len(cusip_list))
                none_found = False
                break
        if none_found:
            print('No bad CUSIPs were found')
            return None
        chunk_size //= 10
    print(cusip_list[start_idx])
    assert True

    
    ## binary search    # overloaded the server since each chunk was so large it created a cascade of API calls
    # start, end = 0, len(cusip_list)
    # while start <= end:
    #     mid = (start + end) // 2
    #     print(start, end, mid)
    #     cusip_list_first_half = cusip_list[start : mid]
    #     request_obj = response_from_batch_pricing(f'{DIRECTORY}/{FILENAME}_pricing.csv', cusip_list_first_half, quantity_list[start : mid], trade_type_list[start : mid])
    #     try:
    #         check_that_batch_pricing_gives_output_for_all_cusips_run_multiple_times(request_obj, cusip_list_first_half)
    #     except AssertionError:
    #         end = mid - 1
    #     else:
    #         start = mid
    # print(cusip_list[mid])
    # assert True
    

    ## one by one    # too slow
    # failed_cusips = []
    # for cusip, quantity in zip(cusip_list, quantity_list):
    #     print(cusip)
    #     # time.sleep(0.1)    # small delay to avoid requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')) https://stackoverflow.com/questions/52051989/requests-exceptions-connectionerror-connection-aborted-connectionreseterro
    #     filename_for_single_cusip = f'{DIRECTORY}/{FILENAME}_{cusip}.csv'
    #     request_obj = response_from_batch_pricing(filename_for_single_cusip, [cusip], [quantity])
    #     try:
    #         check_that_batch_pricing_gives_output_for_all_cusips_run_multiple_times(request_obj, [cusip])
    #     except Exception as e:
    #         failed_cusips.append(cusip)
    #         print(f'The CUSIP {cusip} had the following error: {e}')
    # assert len(failed_cusips) == 0, f'The following CUSIPs had failures: {failed_cusips}'
