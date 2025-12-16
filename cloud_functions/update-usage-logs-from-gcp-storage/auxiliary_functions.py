'''
'''
import time
from functools import wraps
import logging as python_logging    # to not confuse with google.cloud.logging


def run_multiple_times_before_raising_error(errors, max_runs):    # using the same formatting from https://stackoverflow.com/questions/10176226/how-do-i-pass-extra-arguments-to-a-python-decorator
    '''This function customizes the returned decorator for a custom list of `errors` and a number of `max_runs`.'''
    def run_multiple_times_before_failing(function):
        '''This function is to be used as a decorator. It will run `function` over and over again until it does not 
        raise an Exception for a maximum of `max_runs` (specified below) times. `max_runs = 1` is the same functionality 
        as not having this decorator. It solves the following problems: (1) GCP limits how quickly files can be 
        downloaded from buckets and raises an `SSLError` or a `KeyError` when the buckets are accessed too quickly in 
        succession, (2) redis infrequently fails due to connectionError which succeeds upon running the function again.'''
        @wraps(function)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
        def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
            nonlocal max_runs    # `max_runs` belongs to the outer scope
            while max_runs > 0:
                try:
                    return function(*args, **kwargs)
                except tuple(errors) as e:
                    max_runs -= 1
                    if max_runs == 0: raise e
                    python_logging.warning(f'{function.__name__} raise error: {e}, but we will re-attempt it {max_runs} more times before failing')    # raise warning of error instead of error itself
                    time.sleep(1)    # have a one second delay to prevent overloading the number of calls
        return wrapper
    return run_multiple_times_before_failing
