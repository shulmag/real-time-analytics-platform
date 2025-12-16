'''
Description: Point-in-time pricing

             **NOTE**: When updating functions inside a module in this script, need to update the functions using dot notation. For example, inside the `batch_pricing` module is where the most up-to-date version resides, so once it is called from the `price_cusips_list` function, it must be updated with `batch_pricing.predict_spread = ...` (otherwise the import from the other modules has already happened so the name is corresponding to the original function from the imported module, i.e., when `batch_pricing` loads, it imports `predict_spread` from the `pricing_functions` module, and so this is the version that is used unless the name in the `batch_pricing` module is updated)
             **NOTE**: This script needs to be run on a VM so that the yield curve redis can be accessed which is necessary for `process_data(...)`. The error that will be raised otherwise is a `TimeoutError`.
             **NOTE**: To not use the yield spread with similar trades model for point-in-time pricing, the following variable must be changed: `app_engine/demo/server/modules/auxiliary_variables.py::USE_SIMILAR_TRADES_MODEL_FOR_YIELD_SPREAD_PREDICTIONS_FOR_POINT_IN_TIME_PRICING`. For 50k unique CUSIPs and 4 different quantities (200k line items), with 8 CPUs, pricing without similar trades takes approximately 1.75 hours and requires 100 GB of memory, whereas pricing with similar trades takes approximately 3 hours and requires 120 GB of memory.
             **NOTE**: To see the output of this script in an `output.txt` file use the command: $ python -u point_in_time_pricing_timestamp.py >> output.txt. 
             **NOTE**: To run the procedure in the background, use the command: $ nohup python -u point_in_time_pricing_timestamp.py >> output.txt 2>&1 &. This will return a process number such as [1] 66581, which can be used to kill the process.
             Breakdown:
             1. `nohup`: This allows the script to continue running even after you log out or close the terminal.
             2. python -u <file_name>.py: This part is executing your Python script in unbuffered mode (meaning that output is written immediately). If you are using Python 3, you might want to specify python3 instead of just python, depending on your environment.
             3. >> output.txt 2>&1:
                 * >> output.txt appends the standard output (stdout) of the script to output.txt instead of overwriting it.
                 * 2>&1 redirects standard error (stderr) to the same file as standard output, so both stdout and stderr go into output.txt.
             4. &: This runs the command in the background.

             To redirect the error to a different file, you can use 2> error.txt. Note that just ignoring it (not including 2>...) will just output to std out in this case.

             To kill the command, run
             $ kill 66581
             or
             $ kill -9 66581
             The -9 forces the operation.

             This script allows one to see what prices we would have returned on a specified date for a list of CUSIPs. The user specifies the date and time in `DATETIME_OF_INTEREST` and the file with the list of CUSIPs in `FILE_TO_BE_PRICED`. The sequence of events is as follows: 
             1. create a trade history data file where the most recent trade is not after `DATETIME_OF_INTEREST`, 
             2. create a reference data file where the data is the reference features for each CUSIP at the `DATETIME_OF_INTEREST`, and 
             3. use the archived deployed models for the same day if the time is before 5pm PT or the business day after `DATETIME_OF_INTEREST`, since after business hours, we consider the model that was trained up until two business days before the day it is deployed and validated on the business day before it is deployed. 

             The core idea is to use as much code that is deployed i.e., that in `app_engine/demo/server/modules/finance.py`, as possible to maintain consistencies to what is deployed.

             To select an assortment of CUSIPs within a date range, use the following query:
             SELECT
               DISTINCT(cusip)
             FROM
               `auxiliary_views_v2.trade_history_same_issue_5_yr_mat_bucket_1_materialized`
             WHERE
               sp_long_integer < 8
               AND moodys_long_integer < 8
               AND calc_date > "2025-01-01"
               AND dated_date < "2024-02-15"
               AND outstanding_indicator
               AND federal_tax_status = 2
'''
import os
import sys
from functools import wraps
import multiprocess as mp
from datetime import datetime
import pickle as pickle

import numpy as np
import pandas as pd


server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'app_engine', 'demo', 'server'))    # get the directory containing the 'app_engine/demo/server' package
sys.path.append(server_dir)    # add the directory to sys.path
print('NOTE: this file must be run from the `notebooks/point_in_time_pricing/` directory')


from modules.get_creds import get_creds
get_creds()


from modules.auxiliary_variables import NUMERICAL_ERROR
from modules.auxiliary_functions import create_df_chunks
from modules.data_preparation_for_pricing import process_cusip
from modules.batch_pricing import prepare_batch_pricing_results_for_logging, \
                                  prepare_batch_pricing_results_to_output_to_user, \
                                  price_cusips_list, \
                                  process_quantity
from modules.point_in_time_pricing import get_table_string


SET_QUANTITY_TO_AMOUNT_OUTSTANDING_IF_LESS_THAN_GIVEN_QUANTITY = False

QUANTITIES = [1000]    # list(range(10, 100, 10)) + list(range(100, 1000 + 1, 100))    # [25, 50, 250]    # [50, 250, 1000]    # [5, 25, 50, 100, 250, 500, 1000]
TRADE_TYPES = ['P']

DATETIME_OF_INTEREST = '2024-05-09T16:00:00'    # modify to be the datetime at which the pricing occurs

MULTIPROCESSING = True    # setting to `True` means that the `process_data(...)` procedure will be called multiple times in parallel
VERBOSE = True

FILE_TO_BE_PRICED = 'sumridge_all.csv'    # modify to be the file containing the list of CUSIPs to be priced

COLUMNS_TO_CONVERT_FROM_DECIMAL_TO_FLOAT = [    # columns that are in decimal format when getting them from BigQuery and need to be converted to float otherwise downstream operations will fail
    'coupon', 
    'par_price', 
    'next_call_price', 
    'par_call_price', 
    'refund_price', 
    'min_amount_outstanding', 
    'max_amount_outstanding', 
    'maturity_amount', 
    'current_coupon_rate', 
    'issue_amount', 
    'orig_principal_amount', 
    'original_yield', 
    'outstanding_amount', 
]


def function_timer(function_to_time):
    '''This function is to be used as a decorator. It will print out the execution time of `function_to_time`.'''
    @wraps(function_to_time)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
    def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
        print(f'Begin execution of {function_to_time.__name__}')
        start_time = datetime.now()
        result = function_to_time(*args, **kwargs)
        end_time = datetime.now()
        print(f'Execution time of {function_to_time.__name__}: {end_time - start_time}')
        return result
    return wrapper


@function_timer
def load_cusips_from_file_into_dataframe(file_to_be_priced: str, verbose: bool) -> pd.DataFrame:
    '''First check if the filepath `file_to_be_priced` is a CSV or XLSX and use the appropriate pandas function 
    to read it in to a dataframe.'''
    file_to_be_priced_extension = file_to_be_priced.lower()
    if file_to_be_priced_extension.endswith('.csv'):
        to_be_priced_df = pd.read_csv(file_to_be_priced, header=None)
    elif file_to_be_priced_extension.endswith(('.xls', '.xlsx')):
        to_be_priced_df = pd.read_excel(file_to_be_priced, header=None)
    else:
        raise ValueError('Unsupported file format')
    if verbose:
        print('First 10 CUSIPs to be priced')
        print(to_be_priced_df.head(10).to_markdown())
        if len(to_be_priced_df) > 10:
            print('Last 10 CUSIPs to be priced')
            print(to_be_priced_df.tail(10).to_markdown())
    return to_be_priced_df


def adding_the_cusip_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'cusip' in df.columns:
        assert 'cusip' == df.columns[0]    # first column should be cusip
    else:
        df = df.rename(columns={df.columns[0]: 'cusip'})    # rename the first column to 'cusip'
    df['cusip'] = df['cusip'].apply(process_cusip)
    return df


@function_timer
def adding_quantity_and_trade_type_column(df: pd.DataFrame, verbose: bool = VERBOSE) -> pd.DataFrame:
    num_columns = len(df.columns)

    if ('quantity' in df.columns) and ('trade_type' in df.columns):
        print('Both `quantity` and `trade_type` already exist in the passed in dataframe, and so these columns are not being added')
    elif num_columns > 1:
        df = df.rename(columns={df.columns[1]: 'quantity'})    # rename the second column to 'quantity'
    elif num_columns > 2:
        df = df.rename(columns={df.columns[2]: 'trade_type'})    # rename the third column to 'trade_type'
    else:
        assert num_columns == 1, 'Cannot have just the quantity and no trade_type; must be either just CUSIP or CUSIP, quantity, trade_type'    # TODO: implement this functionality
        num_repeats = len(QUANTITIES) * len(TRADE_TYPES)
        df = pd.DataFrame(np.repeat(df.values, num_repeats, axis=0), columns=df.columns)
        df = df.reset_index(drop=True)

        df['quantity'] = QUANTITIES[0]    # initialize the `quantity` column
        for idx, quantity in enumerate(QUANTITIES):
            indices = []
            for shift in range(len(TRADE_TYPES)):
                indices.extend(list(range(idx * len(TRADE_TYPES) + shift, len(df), num_repeats)))
            df.loc[indices, 'quantity'] = quantity

        df['trade_type'] = TRADE_TYPES[0]    # initialize the `trade_type` column
        for idx, trade_type in enumerate(TRADE_TYPES):
            indices = list(range(idx, len(df), len(TRADE_TYPES)))
            df.loc[indices, 'trade_type'] = trade_type

    if verbose:
        columns_to_display = ['cusip', 'quantity', 'trade_type'] + [column for column in ['trade_datetime', 'rtrs_control_number'] if column in df.columns]
        print('First 10 CUSIPs to be priced (with trade types and quantities)')
        print(df[columns_to_display].head(10).to_markdown())
        if len(df) > 10:
            print('Last 10 CUSIPs to be priced (with trade types and quantities)')
            print(df[columns_to_display].tail(10).to_markdown())
    return df


def get_cusips_quantites_trade_types(df: pd.DataFrame):
    cusip_list = df['cusip'].tolist()
    quantity_list = df['quantity'].tolist()
    trade_type_list = df['trade_type'].tolist()
    print('cusip_list[:10]\n', cusip_list[:10])
    print('quantity_list[:10]\n', quantity_list[:10])
    print('trade_type_list[:10]\n', trade_type_list[:10])
    return cusip_list, quantity_list, trade_type_list


def convert_decimal_to_float(df: pd.DataFrame, columns: list = COLUMNS_TO_CONVERT_FROM_DECIMAL_TO_FLOAT) -> pd.DataFrame:
    '''Convert the columns in `columns` to float. This is necessary because the model expects the columns to be of type float.'''
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(float)
    return df


@function_timer
def inspect_results_post_pricing(priced_df: pd.DataFrame) -> None:
    did_not_price = priced_df[priced_df['ytw'] == NUMERICAL_ERROR]
    num_trades_that_did_not_price = len(did_not_price)
    print(f'Number of trades that did not price: {num_trades_that_did_not_price}')
    if num_trades_that_did_not_price > 0:
        print('First 10 trades that did not price')
        print(did_not_price.head(10).to_markdown())


@function_timer
def _price_cusips_point_in_time(to_be_priced_df: pd.DataFrame, 
                                datetime_of_interest: str, 
                                use_multiprocessing: bool, 
                                return_df: bool, 
                                verbose: bool, 
                                use_trade_datetime_column_for_pricing: bool = False, 
                                additional_columns_in_output: list = [], 
                                keep_only_essential_columns_in_output: bool = False, 
                                contains_reference_data_and_trade_history: bool = False):
    '''If `contains_reference_data_and_trade_history` is `True`, then `to_be_priced_df` is assumed 
    to have all of the trade history and reference data needed to call the model, so we need not 
    create the data in this function.'''
    datetime_format = '%Y-%m-%dT%H:%M:%S' if 'T' in datetime_of_interest else '%Y-%m-%d %H:%M:%S'
    datetime_of_interest = datetime.strptime(datetime_of_interest, datetime_format)
    
    to_be_priced_df = adding_the_cusip_column(to_be_priced_df)
    to_be_priced_df = adding_quantity_and_trade_type_column(to_be_priced_df, verbose)

    def price_cusips_list_func(to_be_priced_df_chunk: pd.DataFrame):
        trade_datetime_list = to_be_priced_df_chunk['trade_datetime'].tolist() if use_trade_datetime_column_for_pricing else [datetime_of_interest] * len(to_be_priced_df_chunk)
        cusip_list, quantity_list, trade_type_list = get_cusips_quantites_trade_types(to_be_priced_df_chunk)
        quantity_list = [process_quantity(quantity, None) for quantity in quantity_list]    # setting the default_quantity to `None` will raise an error further downstream if the default quantity is used which is the desired behavior
        return price_cusips_list(cusip_list, 
                                 quantity_list, 
                                 trade_type_list, 
                                 datetime_of_interest, 
                                 trade_datetime_list=trade_datetime_list,    # optional argument `trade_datetime_list` triggers the data collection from the historical data
                                 set_quantity_to_amount_outstanding_if_less_than_given_quantity=SET_QUANTITY_TO_AMOUNT_OUTSTANDING_IF_LESS_THAN_GIVEN_QUANTITY, 
                                 cusip_with_trade_history_and_reference_data_df=to_be_priced_df_chunk if contains_reference_data_and_trade_history else None)

    num_cpus = os.cpu_count()
    if use_multiprocessing and len(to_be_priced_df) > num_cpus:
        print(f'Using multiprocessing inside _price_cusips_point_in_time(...) with num_cpus={num_cpus}')
        with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
            priced_df = pool_object.map(price_cusips_list_func, create_df_chunks(to_be_priced_df, num_cpus))
        priced_df = pd.concat(priced_df).reset_index(drop=True)    # need .reset_index(...) to reset index from 0...n
    else:
        priced_df = price_cusips_list_func(to_be_priced_df)

    priced_df = prepare_batch_pricing_results_to_output_to_user(prepare_batch_pricing_results_for_logging(priced_df), additional_columns_in_output)
    if keep_only_essential_columns_in_output: priced_df = priced_df.drop(columns=['coupon', 'security_description', 'maturity_date'])
    if verbose:
        print('First 10 trades priced')
        print(priced_df.head(10).to_markdown())
        if len(priced_df) > 10:
            print('Last 10 trades priced')
            print(priced_df.tail(10).to_markdown())
        inspect_results_post_pricing(priced_df)

    if return_df:
        return priced_df
    else:
        # save to CSV
        csv_filename = f'priced_{get_table_string(datetime_of_interest)}.csv'
        priced_df.to_csv(csv_filename, index=False)
        return csv_filename


@function_timer
def price_cusips_from_file_point_in_time(file_to_be_priced: str = FILE_TO_BE_PRICED, datetime_of_interest: str = DATETIME_OF_INTEREST, use_multiprocessing: bool = MULTIPROCESSING, return_df: bool = False, verbose: bool = True):
    print(f'Running `price_cusips_from_file_point_in_time(...)` with datetime_of_interest={datetime_of_interest} and file_to_be_priced={file_to_be_priced} and use_multiprocessing={use_multiprocessing}')
    to_be_priced_df = load_cusips_from_file_into_dataframe(file_to_be_priced, verbose)
    return _price_cusips_point_in_time(to_be_priced_df, datetime_of_interest, use_multiprocessing, return_df, verbose)


@function_timer
def price_cusips_from_list_point_in_time(list_of_cusips: list, list_of_quantities: list = None, list_of_trade_types: list = None, datetime_of_interest: str = DATETIME_OF_INTEREST, use_multiprocessing: bool = MULTIPROCESSING, return_df: bool = False, verbose: bool = True, save_data_when_creating_reference_data_and_trade_history: bool = False):
    '''If `list_of_quantities` is `None`, then perform a cross product between each CUSIP in `list_of_cusips` and each quantity in `list_of_quantities`. 
    Similarly, if `list_of_trade_types` is `None`, then perform a cross product between each (CUSIP, quantity) pair and each trade type in `list_of_trade_types`.'''
    print(f'Running `price_cusips_from_list_point_in_time(...)` with datetime_of_interest={datetime_of_interest} and use_multiprocessing={use_multiprocessing}')
    if list_of_quantities is None and list_of_trade_types is None:
        to_be_priced_df = pd.DataFrame(list_of_cusips, columns=['cusip'])
    else:
        if list_of_quantities is None:
            assert len(list_of_cusips) == len(list_of_trade_types), f'`list_of_cusips` has length {len(list_of_cusips)}, which must match that of `list_of_quantities` which as length {len(list_of_trade_types)}'
            list_of_cusips = np.repeat(list_of_cusips, len(QUANTITIES))    # np.repeat([1, 2, 3], 3) -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
            list_of_trade_types = np.repeat(list_of_trade_types, len(QUANTITIES))    # np.repeat([1, 2, 3], 3) -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
            list_of_quantities = np.tile(QUANTITIES, len(list_of_cusips) // len(QUANTITIES))    # np.tile([1, 2, 3], 3) -> [1, 2, 3, 1, 2, 3, 1, 2, 3]
        elif list_of_trade_types is None:
            assert len(list_of_cusips) == len(list_of_quantities), f'`list_of_cusips` has length {len(list_of_cusips)}, which must match that of `list_of_quantities` which as length {len(list_of_quantities)}'
            list_of_cusips = np.repeat(list_of_cusips, len(TRADE_TYPES))    # np.repeat([1, 2, 3], 3) -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
            list_of_quantities = np.repeat(list_of_quantities, len(TRADE_TYPES))    # np.repeat([1, 2, 3], 3) -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
            list_of_trade_types = np.tile(TRADE_TYPES, len(list_of_cusips) // len(TRADE_TYPES))    # np.tile([1, 2, 3], 3) -> [1, 2, 3, 1, 2, 3, 1, 2, 3]
        assert len(list_of_cusips) == len(list_of_quantities) == len(list_of_trade_types), f'`list_of_cusips` has length {len(list_of_cusips)}, which must match that of `list_of_quantities` which as length {len(list_of_quantities)}, which must match that of `list_of_trade_types` which as length {len(list_of_trade_types)}'
        to_be_priced_df = pd.DataFrame({'cusip': list_of_cusips, 'quantity': list_of_quantities, 'trade_type': list_of_trade_types})
    return _price_cusips_point_in_time(to_be_priced_df, datetime_of_interest, use_multiprocessing, return_df, verbose, save_data_when_creating_reference_data_and_trade_history=save_data_when_creating_reference_data_and_trade_history)


@function_timer
def price_trades_at_different_quantities_trade_types(trades_df: pd.DataFrame, date_of_interest: str, use_multiprocessing: bool = MULTIPROCESSING, return_df: bool = False, verbose: bool = True, additional_columns_in_output: list = [], keep_only_essential_columns_in_output: bool = False):
    '''`date_of_interest` is used to get the model on the correct date. The quantities and trade types are populated from `QUANTITIES` and 
    `TRADE_TYPES` (cross product between the rows in `trades_df` and each of the lists).'''
    return _price_cusips_point_in_time(trades_df, date_of_interest + 'T00:00:00', use_multiprocessing, return_df, verbose, contains_reference_data_and_trade_history=True, use_trade_datetime_column_for_pricing=True, additional_columns_in_output=additional_columns_in_output, keep_only_essential_columns_in_output=keep_only_essential_columns_in_output)


if __name__ == '__main__':
    price_cusips_from_file_point_in_time()
