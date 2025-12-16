'''
'''
import time
from datetime import timedelta
from functools import wraps

from auxiliary_variables import API_URL


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


def get_api_call(cusip_list: list, quantity_list: list, trade_type_list: list, user_price_list: list, trade_datetime_list: list, username: str, password: str):
    # Single assert statement checking all list lengths are equal
    all_lists = {
        'cusips': len(cusip_list),
        'quantities': len(quantity_list),
        'trade types': len(trade_type_list),
        'user prices': len(user_price_list),
        'trade datetimes': len(trade_datetime_list)
    }
    
    lengths = set(all_lists.values())
    if len(lengths) != 1:
        error_msg = 'Inconsistent list lengths:\t'
        for name, length in all_lists.items():
            error_msg += f'{name}: {length}\n'
        raise ValueError(error_msg)

    url = API_URL + '/api/compliance'
    data = {
        'username': username,
        'password': password,
        'cusipList': cusip_list,
        'quantityList': quantity_list,
        'tradeTypeList': trade_type_list,
        'userPriceList': user_price_list,
        'tradeDatetimeList': trade_datetime_list
    }
    
    return url, data
