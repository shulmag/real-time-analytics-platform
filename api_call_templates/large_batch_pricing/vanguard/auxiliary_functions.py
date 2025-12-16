'''
'''
import time
from datetime import timedelta
from functools import wraps

from auxiliary_variables import API_URL_LIST


def function_timer(function_to_time):
    '''This function is to be used as a decorator. It will print out the execution time of `function_to_time`. 
    It is very similar to the decorator by the same name in `app_engine/demo/server/modules/ficc/utils/auxiliary_functions.py`.'''
    @wraps(function_to_time)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
    def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
        print(f'BEGIN {function_to_time.__name__}')
        start_time = time.time()
        result = function_to_time(*args, **kwargs)
        end_time = time.time()
        print(f'END {function_to_time.__name__}. Execution time: {timedelta(seconds=end_time - start_time)}')
        return result
    return wrapper


def get_api_call(cusip_list: list, 
                 quantity_list: list, 
                 trade_type_list: list, 
                 username: str, 
                 password: str, 
                 time: str = None, 
                 server_idx: int = 0):
    assert len(cusip_list) == len(quantity_list) == len(trade_type_list), f'Number of CUSIPs: {len(cusip_list)} is not equal to the number of quantities: {len(quantity_list)} is not equal to the number of trade types: {len(trade_type_list)}'
    url = API_URL_LIST[server_idx] + '/api/batchpricing'
    data = {'username': username, 
            'password': password}
    data['cusipList'] = cusip_list
    data['quantityList'] = quantity_list
    data['tradeTypeList'] = trade_type_list
    data['currentTime'] = time
    return url, data
