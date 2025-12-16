'''
Updated on 04/19/2023 by Developer. 
This module gets the most recent Nelson-Siegel coefficient and the standard scalar coefficient from Redis Memorystore given a particular datetime. 
Between the hours of 9:30 and 15:59 ET on business days, the Nelson-Siegel coefficient will be the coefficient for that minute and the standard scalar will be that of the last business day.
'''
import pandas as pd
import pickle5 as pickle
import redis

from pytz import timezone
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from dateutil import parser
from datetime import datetime, time


eastern = timezone('US/Eastern')    # We only use datetime aware datetimes in ET; naive datetimes should be localized

redis_client = redis.Redis(host='10.227.69.60', port=6379, db=0)

# Creates a dataframe containing US Federal holidays.
dr = pd.date_range(start='2010-01-01', end='2100-01-01')
df = pd.DataFrame()
df['Date'] = dr
cal = calendar()
holidays = cal.holidays(start=dr.min(), end=dr.max())

YEAR_MONTH_DAY_HOUR_MIN = '%Y-%m-%d:%H:%M'
YEAR_MONTH_DAY_HOUR_MIN_SEC = '%Y-%m-%d:%H:%M:%S'
START_DATE = '2021-08-03'    # '2021-07-27' was for the daily yield curve, but '2021-08-03' is for the realtime yield curve
START_TIME = '09:30'


def get_business_day(date):
    '''Checks whether date is a datetime object, then checks whether the date is before we began collecting yield curve coefficients, 
    then checks if the date is a weekend or a US Federal Holiday. If the last condition is true, the function loops back to the most 
    recent business day.'''
    if isinstance(date, datetime):
        if date.tzinfo is None:
            date = eastern.localize(date)
    else:
        date = parser.parse(date)
        date = eastern.localize(date)

    start_date = datetime.strptime(START_DATE + ':' + START_TIME + ':00', YEAR_MONTH_DAY_HOUR_MIN_SEC)    # Date we began to collect YC data. 
    start_date = eastern.localize(start_date)
    if date < start_date:
        return start_date
    elif date > datetime.now(eastern):
        date = datetime.now(eastern)

    while date.strftime("%Y%m%d") in holidays or date.weekday() in {5, 6}:    # 5 represents saturday and 6 represents sunday
        date = date - pd.DateOffset(1)
        date = date.replace(hour=15, minute=59)
    return date


def get_last_business_time(date):
    '''Checks whether the time of the datetime object is before or after business hours. If so, it sends us back to the last 
    business datetime.'''
    market_open = time(9, 30)
    market_close = time(16, 0)
    if date.time() < market_open:
        date = get_business_day(date - pd.DateOffset(1))
        date = date.replace(hour=15, minute=59)
        return date
    elif date.time() > market_close:
        date = date.replace(hour=15, minute=59)
        return date
    else: 
        return date


def is_time_between(begin_time, end_time, check_time):
    # If check time is not given, default to current UTC time
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else:    # crosses midnight
        return check_time >= begin_time or check_time <= end_time

def find_last_minute(date):
    '''Checks whether the datetime exists as a key in redis.  If not, we loop back to the previous datetime key.'''
    if  date.replace(tzinfo=eastern) < datetime.strptime(START_DATE + ':' + START_TIME, YEAR_MONTH_DAY_HOUR_MIN).replace(tzinfo=eastern):
        date = datetime.strptime(START_DATE + ':' + START_TIME, YEAR_MONTH_DAY_HOUR_MIN)
        return date
    else: 
        while not redis_client.exists(date.strftime(YEAR_MONTH_DAY_HOUR_MIN)):
            # print(date)
            # only in between 9:30 am to 4pm EST
            if is_time_between(time(9, 30), time(16, 00), date.time()):
                date = date-pd.Timedelta(minutes=1)
                # print(f"redis yc not found for {date.strftime(YEAR_MONTH_DAY_HOUR_MIN)}")
            else:
                date=get_business_day(date-pd.Timedelta(days=1))
                date = date.replace(hour=15, minute=59)
                #print(f"*** Skip Day: redis yc not found for {date.strftime(YEAR_MONTH_DAY_HOUR_MIN)}")
        else:
            return date

def get_yc_data(date):
    '''Fetches the most recent data from redis given a particular datetime.'''
    date = get_business_day(date)
    date = get_last_business_time(date)
    date = find_last_minute(date)
    data = pickle.loads(redis_client.get(date.strftime(YEAR_MONTH_DAY_HOUR_MIN)))
    return data

# def find_last_minute(date):
#     counter = 0 
#     counter_threshold = 20
#     '''Checks whether the datetime exists as a key in redis.  If not, we loop back to the previous datetime key.'''
#     if  date.replace(tzinfo=eastern) < datetime.strptime(START_DATE + ':' + START_TIME, YEAR_MONTH_DAY_HOUR_MIN).replace(tzinfo=eastern):
#         date = datetime.strptime(START_DATE + ':' + START_TIME, YEAR_MONTH_DAY_HOUR_MIN)
#         return date
#     else:
#         while not redis_client.exists(date.strftime(YEAR_MONTH_DAY_HOUR_MIN)):
#             # print(date)
#             if counter > counter_threshold: 
#                 # print(f'Threshold reached for absent real time yield curve parameters on {date.strftime("%Y-%m-%d")}, jumping back 1 bday')
#                 date=get_business_day(date-pd.Timedelta(days=1))
#                 date = date.replace(hour=15, minute=59)
#                 counter = 0
#                 continue
                
#             # only in between 9:30 am to 4pm EST
#             if is_time_between(time(9, 30), time(16, 00), date.time()):
#                 date = date-pd.Timedelta(minutes=1)
#                 counter += 1
#                 # print(f"redis yc not found for {date.strftime(YEAR_MONTH_DAY_HOUR_MIN)}")
#             else:
#                 date=get_business_day(date-pd.Timedelta(days=1))
#                 date = date.replace(hour=15, minute=59)
#                 #print(f"*** Skip Day: redis yc not found for {date.strftime(YEAR_MONTH_DAY_HOUR_MIN)}")
#         else:
#             return date
