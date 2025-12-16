'''Point in time pricing for every timestamp from file
Last updated by Developer on 2024-03-13.

**NOTE**: This script needs to be run on a VM so that the yield curve redis can be accessed which is necessary for `process_data(...)`. The error that will be raised otherwise is a `TimeoutError`.

This script allows one to see what prices we would have returned on a specified date for a list of CUSIPs. The user specifies the date and time in `DATETIME_OF_INTEREST` and the file with the list of CUSIPs in `FILE_TO_BE_PRICED`. The sequence of events is as follows: 
1. create a trade history data file where the most recent trade is not after `DATETIME_OF_INTEREST`, 
2. create a reference data file where the data is the reference features for each CUSIP at the `DATETIME_OF_INTEREST`, and 
3. use the archived deployed models for the same day if the time is before 5pm PT or the business day after `DATETIME_OF_INTEREST`, since after business hours, we consider the model that was trained up until two business days before the day it is deployed and validated on the business day before it is deployed. 

The core idea is to use as much code that is deployed i.e., that in `app_engine/demo/server/modules/finance.py`, as possible to maintain consistencies to what is deployed.'''
import pandas as pd

from point_in_time_pricing_timestamp import price_cusips_from_file_point_in_time


START_DATE = '2024-07-05'
END_DATE = '2024-07-05'
TIMESTAMPS = [f'{hour}:00:00' for hour in range(16, 16 + 1, 2)]    # 9am to 3pm; the last argument in the range specifies the step size (e.g., set to 1 to get every hour or 2 to get every other hour)


def create_business_date_range(start_date: str, end_date: str):
    date_range = pd.bdate_range(start=start_date, end=end_date)    # use `.bdate_range(...)` instead of `.date_range(...)` to automatically filter out non-business days
    return [date.strftime('%Y-%m-%d') for date in date_range]


def create_timestamps_from_dates(date_range):
    '''Add each timestamp in `TIMESTAMPS` to each date in `date_range`. Assumes that `TIMESTAMPS` is a list of strings.'''
    return [date + 'T' + timestamp for date in date_range for timestamp in TIMESTAMPS]


if __name__ == '__main__':
    date_range = create_business_date_range(START_DATE, END_DATE)
    timestamp_range = create_timestamps_from_dates(date_range)

    print('timestamp_range:', timestamp_range)

    for timestamp in timestamp_range:
        price_cusips_from_file_point_in_time(datetime_of_interest=timestamp)
