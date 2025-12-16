import time
import logging as python_logging    # to not confuse with google.cloud.logging
from functools import wraps

import redis


MAX_RUNS = 10


def run_multiple_times_before_failing(error_types: tuple, max_runs: int = MAX_RUNS, exponential_backoff: bool = False):
    '''This function returns a decorator. It will run `function` over and over again until it does not 
    raise an Exception for a maximum of `max_runs` times. If `exponential_backoff` is set to `True`, then 
    the wait time is increased exponentially, otherwise it is a constant value.
    NOTE: max_runs = 1 is the same functionality as not having this decorator.'''
    def decorator(function):
        @wraps(function)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
        def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
            runs_so_far = 0
            while runs_so_far < max_runs:
                try:
                    return function(*args, **kwargs)
                except error_types as e:
                    runs_so_far += 1
                    if runs_so_far >= max_runs:
                        print(f'WARNING: Already caught {type(e)}: {e}, {max_runs} times in {function.__name__}, so will now raise the error')    # python_logging.warning(f'Already caught {type(e)}: {e}, {max_runs} times in {function.__name__}, so will now raise the error')
                        raise e
                    print(f'WARNING: Caught {type(e)}: {e}, and will retry {function.__name__} {max_runs - runs_so_far} more times')    # python_logging.warning(f'Caught {type(e)}: {e}, and will retry {function.__name__} {max_runs - runs_so_far} more times')
                    delay = min(2 ** (runs_so_far - 1), 10) if exponential_backoff else 1
                    time.sleep(delay)    # have a delay to prevent overloading the server
        return wrapper
    return decorator


def run_five_times_before_raising_redis_connector_error(function: callable) -> callable:
    return run_multiple_times_before_failing((redis.exceptions.ConnectionError,), 5)(function)