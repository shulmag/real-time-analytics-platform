'''Point in time pricing for each timestamp from file
Last updated by Developer on 2024-05-09.

**NOTE**: This script needs to be run on a VM so that the yield curve redis can be accessed which is necessary for `process_data(...)`. The error that will be raised otherwise is a `TimeoutError`.

This script allows one to see what prices we would have returned on a specified date for a list of CUSIPs. The user specifies the date and time in `DATETIME_OF_INTEREST` and the file with the list of CUSIPs along with the specified timestamp and date in `FILE_TO_BE_PRICED`. The sequence of events is as follows: 
1. create a trade history data file where the most recent trade is not after `DATETIME_OF_INTEREST`, 
2. create a reference data file where the data is the reference features for each CUSIP at the `DATETIME_OF_INTEREST`, and 
3. use the archived deployed models for the same day if the time is before 5pm PT or the business day after `DATETIME_OF_INTEREST`, since after business hours, we consider the model that was trained up until two business days before the day it is deployed and validated on the business day before it is deployed. 

The core idea is to use as much code that is deployed i.e., that in `app_engine/demo/server/modules/finance.py`, as possible to maintain consistencies to what is deployed.'''
from tqdm import tqdm
import multiprocess as mp    # using `multiprocess` instead of `multiprocessing` because function to be called in `map` is in the same file as the function which is calling it: https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror

import pandas as pd

from point_in_time_pricing_timestamp import MODEL_FOLDERS, \
                                            MODEL_BUCKET, \
                                            function_timer, \
                                            load_model, \
                                            load_cusips_from_file_into_dataframe, \
                                            price_cusips_from_list_point_in_time


CSV_WITH_CUSIP_TIMESTAMP_DATE_FILENAME = 'investment_grade_tax_exempt_par_traded_less_than_1M_02012024_02152024.csv'    # modify to be the file containing the list of CUSIPs, timestamps, and dates to be priced
CSV_WITH_CUSIP_DATE_FILENAME = 'allspring_cusip_date_pairs_on_2024-11-13.csv'    # modify to be the file containing the list of CUSIPs and dates to be priced
CSV_WITHOUT_CUSIP_DATE_FILENAME = 'sumridge_1000.csv'    # modify to be the file containing the list of CUSIPs to be priced

MULTIPROCESSING = True
SAVE_DATA_WHEN_CREATING_REFERENCE_DATA_AND_TRADE_HISTORY = not MULTIPROCESSING

YEAR_MONTH_DAY = '%Y-%m-%d'


def load_csv_with_cusip_timestamp_date_into_dataframe(csv_with_cusip_timestamp_date_filename: str, verbose: bool = True) -> pd.DataFrame:
    '''TODO: allow for there not to be a third column and get the date from the timestamp column.'''
    to_be_priced_df = load_cusips_from_file_into_dataframe(csv_with_cusip_timestamp_date_filename, verbose)
    assert len(to_be_priced_df.columns) == 3, 'CSV must have 3 columns where the first column is the CUSIP, the second column is the timestamp, and the third column is the date (corresponding to this timestamp)'
    to_be_priced_df = to_be_priced_df.rename(columns={to_be_priced_df.columns[0]: 'cusip', 
                                                      to_be_priced_df.columns[1]: 'datetime_to_price_trade', 
                                                      to_be_priced_df.columns[2]: 'date_to_price_trade'})
    return to_be_priced_df


def load_csv_with_cusip_date_into_dataframe(csv_with_cusip_date_filename: str, verbose: bool = True) -> pd.DataFrame:
    to_be_priced_df = load_cusips_from_file_into_dataframe(csv_with_cusip_date_filename, verbose)
    assert len(to_be_priced_df.columns) == 2, 'CSV must have 2 columns where the first column is the CUSIP, the second column is the date'
    to_be_priced_df = to_be_priced_df.rename(columns={to_be_priced_df.columns[0]: 'cusip', 
                                                      to_be_priced_df.columns[1]: 'date_to_price_trade'})
    return to_be_priced_df


def load_csv_without_cusip_date_into_dataframe(csv_without_cusip_date_filename: str, verbose: bool = True) -> pd.DataFrame:
    to_be_priced_df = load_cusips_from_file_into_dataframe(csv_without_cusip_date_filename, verbose)
    assert len(to_be_priced_df.columns) == 1, 'CSV must have 1 column where the column is the CUSIP'
    to_be_priced_df = to_be_priced_df.rename(columns={to_be_priced_df.columns[0]: 'cusip'})
    return to_be_priced_df


def price_cusips_at_datetimes_for_specific_date(date_or_datetime: str, df_for_specific_date: pd.DataFrame, use_model_cache: bool = True) -> pd.DataFrame:
    '''Group by `datetime_to_price_trade` and price every CUSIP in this group. Most groups will have just one CUSIP. 
    The `date` argument is used for getting the model and creating the name for the CSV file.'''
    date = date_or_datetime if len(date_or_datetime) == 10 else date_or_datetime[:10]    # assume that it is a proper date string if there are 10 characters in it; 10 comes from the total number of characters in YYYY-MM-DD
    print('Date:', date)
    model_cache = populate_model_cache([date]) if use_model_cache else None
    
    def price_cusips_datetime_cusip_list(datetime_of_interest: str, cusip_list: list) -> pd.DataFrame:
        priced_df = price_cusips_from_list_point_in_time(cusip_list, 
                                                         datetime_of_interest=datetime_of_interest, 
                                                         use_multiprocessing=False, 
                                                         return_df=True, 
                                                         verbose=False, 
                                                         model_cache=model_cache, 
                                                         save_data_when_creating_reference_data_and_trade_history=SAVE_DATA_WHEN_CREATING_REFERENCE_DATA_AND_TRADE_HISTORY)    # since these groups are very small, we do not need multiprocessing for each datetime; we should instead use multiprocessing on the outer for loop
        priced_df['trade_datetime'] = datetime_of_interest
        return priced_df

    group_by_datetime = df_for_specific_date.groupby('datetime_to_price_trade')
    print(f'Number of datetimes for {date}:', group_by_datetime.ngroups)
    datetime_cusip_list = [(datetime_of_interest, df['cusip'].tolist()) for datetime_of_interest, df in group_by_datetime]
    
    ## using multiprocessing on the inner loop here causes a hangup
    # if MULTIPROCESSING and group_by_datetime.ngroups > 1:    # only do multiprocessing if the number of groups is more than 1
    #     with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
    #         all_priced_dfs = pool_object.starmap(price_cusips_datetime_cusip_list, datetime_cusip_list)
    # else:
    all_priced_dfs = [price_cusips_datetime_cusip_list(datetime_of_interest, cusip_list) for datetime_of_interest, cusip_list in tqdm(datetime_cusip_list, disable=group_by_datetime.ngroups == 1)]
    
    priced_df = pd.concat(all_priced_dfs, ignore_index=True)
    priced_df = priced_df.sort_values(by=['trade_datetime', 'cusip'])
    columns_list = priced_df.columns.tolist()
    priced_df = priced_df[columns_list[:3] + ['trade_datetime'] + columns_list[3:-1]]    # move the `trade_datetime` column after the first 3 which are cusip, quantity, and trade_type
    csv_filename = f'priced_{date}.csv'
    priced_df.to_csv(csv_filename, index=False)
    return csv_filename


@function_timer
def populate_model_cache(dates: list) -> dict:
    def get_yield_spread_and_dollar_price_models(date):
        return [load_model(pd.to_datetime(date), folder, bucket=MODEL_BUCKET) for folder in MODEL_FOLDERS]

    # if MULTIPROCESSING and len(dates) > 1:    # only do multiprocessing if the number of groups is more than 1
    #     with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
    #         models = pool_object.map(get_yield_spread_and_dollar_price_models, dates)
    # else:
    models = [get_yield_spread_and_dollar_price_models(date) for date in tqdm(dates, disable=len(dates) == 1)]

    yield_spread_models = {(date, MODEL_FOLDERS[0], MODEL_BUCKET): model[0] for date, model in zip(dates, models)}    # `model[0]` corresponds to the yield spread model (order determined by `MODEL_FOLDERS`)
    dollar_price_models = {(date, MODEL_FOLDERS[1], MODEL_BUCKET): model[1] for date, model in zip(dates, models)}    # `model[1]` corresponds to the dollar price model (order determined by `MODEL_FOLDERS`)
    return {**yield_spread_models, **dollar_price_models}
    

@function_timer
def price_cusips_at_datetimes(use_model_cache=True):
    to_be_priced_df = load_csv_with_cusip_timestamp_date_into_dataframe(CSV_WITH_CUSIP_TIMESTAMP_DATE_FILENAME)
    group_by_date = to_be_priced_df.groupby('date_to_price_trade')

    def create_model_cache_and_price_cusips_at_datetimes_for_specific_date(date, df):
        return price_cusips_at_datetimes_for_specific_date(date, df, use_model_cache)

    ## using multiprocessing on the outer loop does not allow us to track the progress of the procedure which may take hundreds of hours
    # if MULTIPROCESSING and group_by_date.ngroups > 1:
    #     with mp.Pool() as pool_object:
    #         pool_object.starmap(create_model_cache_and_price_cusips_at_datetimes_for_specific_date, group_by_date)    # need to use starmap since `price_cusips_df(...)` has multiple arguments: https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
    # else:
    for date, df in tqdm(group_by_date, disable=len(group_by_date) == 1):    # disable tqdm progress bar if the number of groups is 1
        create_model_cache_and_price_cusips_at_datetimes_for_specific_date(date, df)


def add_timestamp_of_end_of_day_to_date(df: pd.DataFrame, date_column: str):
    '''Add a timestamp of 6pm to the `date_column` of `df`.'''
    df[date_column] = pd.to_datetime(df[date_column])    # convert the `date_column` column to datetime format
    df[date_column] = df[date_column] + pd.Timedelta(hours=18)    # add 6pm timestamp to each date
    df[date_column] = df[date_column].dt.strftime(YEAR_MONTH_DAY + ' ' + '%H:%M:%S')    # convert the `date_column` column to string format
    return df


@function_timer
def price_cusips_at_end_of_day_with_df(to_be_priced_df: pd.DataFrame):
    to_be_priced_df = add_timestamp_of_end_of_day_to_date(to_be_priced_df, 'date_to_price_trade')
    to_be_priced_df = to_be_priced_df.rename(columns={'date_to_price_trade': 'datetime_to_price_trade'})
    group_by_datetime = to_be_priced_df.groupby('datetime_to_price_trade')

    price_cusips_at_datetimes_for_specific_date_caller = lambda datetime, df: price_cusips_at_datetimes_for_specific_date(datetime, df, False)
    # using multiprocessing on the outer loop does not allow us to track the progress of the procedure which may take hundreds of hours; only use it if you are sure that the data is small
    if MULTIPROCESSING and group_by_datetime.ngroups > 1:
        with mp.Pool() as pool_object:
            pool_object.starmap(price_cusips_at_datetimes_for_specific_date_caller, group_by_datetime)    # need to use starmap since `price_cusips_df(...)` has multiple arguments: https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
    else:
        for datetime, df in tqdm(group_by_datetime, disable=len(group_by_datetime) == 1):    # disable tqdm progress bar if the number of groups is 1
            price_cusips_at_datetimes_for_specific_date_caller(datetime, df)


def price_cusips_at_dates_end_of_day():
    '''Use this function if there are multiple dates in the CSV itself at column: `date_to_price_trade`.'''
    to_be_priced_df = load_csv_with_cusip_date_into_dataframe(CSV_WITH_CUSIP_DATE_FILENAME)
    price_cusips_at_end_of_day_with_df(to_be_priced_df)


def price_cusips_at_date_end_of_day(date_as_string: str):
    '''Use this function to price the entire CSV at a single date specified by `date_as_string`.'''
    to_be_priced_df = load_csv_without_cusip_date_into_dataframe(CSV_WITHOUT_CUSIP_DATE_FILENAME)
    to_be_priced_df['date_to_price_trade'] = date_as_string
    price_cusips_at_end_of_day_with_df(to_be_priced_df)


if __name__ == '__main__':
    price_cusips_at_date_end_of_day('2024-04-15')
