"""
Description: This module gets the most recent Nelson-Siegel coefficient and the standard scalar coefficient from Redis Memorystore given a particular datetime.
             Between the hours of 9:30 and 15:59 ET on business days, the Nelson-Siegel coefficient will be the coefficient for that minute and the standard
             scalar will be that of the last business day.
"""

import pandas as pd
import pickle
from datetime import datetime, time

from pytz import timezone
import redis
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday

from dateutil import parser

from modules.ficc.utils.auxiliary_functions import (
    run_five_times_before_raising_redis_connector_error,
)


EASTERN = timezone(
    "US/Eastern"
)  # We only use datetime aware datetimes in ET; naive datetimes should be localized

REDIS_CLIENT = redis.Redis(host="10.227.69.60", port=6379, db=0)
# REDIS_CLIENT = redis.Redis(host="127.0.0.1", port=46379, db=0)


class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]


BUSINESS_DAY = CustomBusinessDay(
    calendar=USHolidayCalendarWithGoodFriday()
)  # used to skip over holidays when adding or subtracting business days


YEAR_MONTH_DAY_HOUR_MIN = "%Y-%m-%d:%H:%M"
YEAR_MONTH_DAY_HOUR_MIN_SEC = "%Y-%m-%d:%H:%M:%S"
START_DATE = "2021-08-03"  # '2021-07-27' was for the daily yield curve, but '2021-08-03' is for the realtime yield curve
START_TIME = "09:30"
START_DATE_TIME = EASTERN.localize(
    datetime.strptime(START_DATE + ":" + START_TIME, YEAR_MONTH_DAY_HOUR_MIN)
)  # date we began to collect YC data

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(15, 59)


def get_business_day(date_and_time):
    """Checks whether date_and_time is a datetime object, then checks whether the date_and_time is before we began
    collecting yield curve coefficients, then checks if the date_and_time is a weekend or a US Federal Holiday. If
    the last condition is true, the function loops back to the most recent business day."""
    if isinstance(date_and_time, datetime):
        if date_and_time.tzinfo is None:
            date_and_time = EASTERN.localize(date_and_time)
    else:
        date_and_time = parser.parse(date_and_time)
        date_and_time = EASTERN.localize(date_and_time)

    if date_and_time < START_DATE_TIME:
        return START_DATE_TIME
    elif date_and_time > datetime.now(EASTERN):
        date_and_time = datetime.now(EASTERN)

    if not BUSINESS_DAY.is_on_offset(
        date_and_time
    ):  # `is_on_offset(...)` checks if the date_and_time is a `BUSINESS_DAY`
        date_and_time = date_and_time - BUSINESS_DAY
        date_and_time = date_and_time.replace(hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute)
    return date_and_time


def get_last_business_time(date_and_time):
    """Checks whether the time of the datetime object is before or after business hours. If so, it sends us back
    to the last business datetime."""
    if date_and_time.time() < MARKET_OPEN:
        date_and_time = get_business_day(date_and_time - (BUSINESS_DAY * 1))
    if not is_time_between(
        MARKET_OPEN, MARKET_CLOSE, date_and_time.time()
    ):  # before `MARKET_OPEN` or after `MARKET_CLOSE`
        date_and_time = date_and_time.replace(hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute)
    return date_and_time


def is_time_between(begin_time, end_time, check_time):
    """If check time is not given, default to current UTC time."""
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else:  # crosses midnight
        return check_time >= begin_time or check_time <= end_time


@run_five_times_before_raising_redis_connector_error
def date_and_time_exists_in_redis(date_and_time):
    """Creating this function in order to be able to decorate it so that it retries when
    redis.exceptions.ConnectionError is raised."""
    return REDIS_CLIENT.exists(date_and_time.strftime(YEAR_MONTH_DAY_HOUR_MIN))


@run_five_times_before_raising_redis_connector_error
def get_date_and_time_in_redis(date_and_time):
    """Creating this function in order to be able to decorate it so that it retries when
    redis.exceptions.ConnectionError is raised."""
    return pickle.loads(REDIS_CLIENT.get(date_and_time.strftime(YEAR_MONTH_DAY_HOUR_MIN)))


def find_last_minute(date_and_time):
    """Checks whether the datetime exists as a key in redis. If not, we loop back to the previous datetime key."""

    def iterate_backward(date_and_time, num_mins_back, max_num_times=-1):
        """Iterate backward from `date_and_time` in chunks of `num_mins_back` for a maximum of
        `max_num_times`. If the date and time are not found, i.e., the `max_num_times` has been
        reached, then return the last searched date and time, with the second argument as `False`.
        Otherwise, return the second argument as `True`, with the date and time whenever it is found.
        """
        while not date_and_time_exists_in_redis(
            date_and_time
        ):  # find the `date_and_time` that is at most `num_mins_back` minutes in the past
            if max_num_times == 0:
                return date_and_time, False
            if is_time_between(
                MARKET_OPEN, MARKET_CLOSE, date_and_time.time()
            ):  # only in between 9:30 am to 4pm EST
                date_and_time -= pd.Timedelta(minutes=num_mins_back)
                # print(f'redis yc not found for {date_and_time.strftime(YEAR_MONTH_DAY_HOUR_MIN)}')
            else:
                date_and_time = get_business_day(date_and_time - (BUSINESS_DAY * 1))
                date_and_time = date_and_time.replace(
                    hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute
                )
                # print(f'*** Skip Day: redis yc not found for {date_and_time.strftime(YEAR_MONTH_DAY_HOUR_MIN)}')
            max_num_times -= 1
        return date_and_time, True

    def iterate_forward(date_and_time, num_mins_forward):
        """Iterate forward from `date_and_time` until the date and time is no longer found in the
        redis. Once it is not found, return the previous date and time inspected to be the most
        recent date and time that was found in the redis."""
        while date_and_time_exists_in_redis(
            date_and_time
        ):  # after finding `date_and_time` to most `num_mins_back` minutes in the past, increment up by 1 minute until we cannot find it anymore
            date_and_time += pd.Timedelta(minutes=num_mins_forward)
        return date_and_time - pd.Timedelta(minutes=num_mins_forward)

    def iterate_backward_backward_forward(date_and_time, num_mins):
        """Iterate back minute by minute for the first `num_mins` minutes. If the date and time does not
        exist in the redis client, then start iterating back by `num_mins` chunks. Once the date and time
        is found, iterate forward minute by minute until date and time is no longer found. This will be
        the most recent date and time for which we have coefficients."""
        date_and_time, found = iterate_backward(date_and_time, 1, num_mins)
        if found:
            return date_and_time
        date_and_time, _ = iterate_backward(date_and_time, num_mins)
        return iterate_forward(date_and_time, 1)

    if date_and_time.replace(tzinfo=EASTERN) < START_DATE_TIME:
        return START_DATE_TIME
    else:
        return iterate_backward_backward_forward(
            date_and_time, 5
        )  # iterate_backward(date_and_time, 1)[0]


def get_yc_data(date_and_time):
    """Fetches the most recent data from redis given a particular datetime."""
    date_and_time = get_business_day(date_and_time)
    date_and_time = get_last_business_time(date_and_time)
    date_and_time = find_last_minute(date_and_time)
    data = get_date_and_time_in_redis(date_and_time)
    return data
