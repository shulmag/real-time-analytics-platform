'''
'''
import time
from functools import wraps
from pytz import timezone
from datetime import datetime, timedelta

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday


EASTERN = timezone('US/Eastern')


def remove_hours_and_fractional_seconds_beyond_3_digits(time):
    '''Taken directly from ficc/app_engine/demo/server/modules/ficc/utils/auxiliary_functions.py.'''
    # time = str(time).split('.')[0]    # remove the fractional seconds
    time = str(time)[:-3]    # total of 6 digits after the decimal, so we keep everything but the last 3
    return time[time.find(':') + 1:]    # remove the hours


def function_timer(function_to_time):
    '''Taken directly from ficc/app_engine/demo/server/modules/ficc/utils/auxiliary_functions.py.
    This function is to be used as a decorator. It will print out the execution time of `function_to_time`.'''
    @wraps(function_to_time)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
    def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
        print(f'Begin execution of {function_to_time.__name__}')
        start_time = time.time()
        result = function_to_time(*args, **kwargs)
        end_time = time.time()
        print(f'Execution time of {function_to_time.__name__}: {remove_hours_and_fractional_seconds_beyond_3_digits(timedelta(seconds=end_time - start_time))}')
        return result
    return wrapper


def go_to_previous_weekday_if_weekend(datetime_object: datetime) -> datetime:
    '''If the given `datetime_object` is a Saturday or Sunday, this function will return the previous Friday.
    Otherwise, it will return the `datetime_object` itself.'''
    if datetime_object.weekday() == 5:    # Saturday
        return datetime_object - timedelta(days=1)
    elif datetime_object.weekday() == 6:    # Sunday
        return datetime_object - timedelta(days=2)
    else:
        return datetime_object


class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]


def today_is_a_holiday() -> bool:
    '''Determine whether today is a US national holiday.'''
    now = datetime.now(EASTERN)
    now = go_to_previous_weekday_if_weekend(now)    # if today is a Saturday or Sunday, we will use the previous Friday
    today = pd.Timestamp(now).tz_localize(None).normalize()    # `.tz_localize(None)` is to remove the time zone; `.normalize()` is used to remove the time component from the timestamp
    current_year = now.year
    holidays_in_last_year_and_next_year = set(USHolidayCalendarWithGoodFriday().holidays(start=f'{current_year - 1}-01-01',end=f'{current_year + 1}-12-31'))
    if today in holidays_in_last_year_and_next_year:
        print(f'Today, {today}, is a national holiday, and so we will not run this cloud function')
        return True
    return False
