'''
Description: This script runs the compliance module, and allows it to be run in both cases of the data already being created and the data needing to be created. Specifically, 
             the use case for the data already being created, is if one wants to run the compliance module for trades that have already occurred, and can get the reference data 
             and trade history and similar trade history from the materialized trade history table. This data is then passed in as a pickle file (or can be created with a query) 
             before running `price_cusips_list(...)` which actually calls the compliance module with this data.
'''
import os
import sys
import multiprocess as mp
import pandas as pd


server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'app_engine', 'demo', 'server'))    # get the directory containing the 'app_engine/demo/server' package
sys.path.append(server_dir)    # add the directory to sys.path
print('NOTE: this file must be run from the `notebooks/compliance/` directory')


from modules.auxiliary_variables import ADDITIONAL_FEATURES_FOR_COMPLIANCE_CSV
from modules.auxiliary_functions import create_df_chunks
from modules.batch_pricing import price_cusips_list, process_quantity, prepare_batch_pricing_results_to_output_to_user, prepare_batch_pricing_results_for_logging


def price_compliance_df(df: pd.DataFrame, 
                        df_contains_trade_history_and_reference_data: bool = True, 
                        keep_additional_columns_from_original_df: bool = False, 
                        quantity_already_processed: bool = True, 
                        additional_columns_in_output_csv: list = []):
    '''Use `price_cusips_list(...)` to run the compliance module on `df`.'''
    def price_cusips_list_func(to_be_priced_df_chunk: pd.DataFrame):
        cusip_list = to_be_priced_df_chunk['cusip'].tolist()
        quantity_list = to_be_priced_df_chunk['quantity'].astype(float).tolist()
        trade_type_list = to_be_priced_df_chunk['trade_type'].tolist()
        user_price_list = to_be_priced_df_chunk['user_price'].astype(float).tolist()
        trade_datetime_list = to_be_priced_df_chunk['trade_datetime'].tolist()
        
        print(f'quantity_list: {quantity_list}')    # have the user make sure that `quantity_list` does not need to be multiplied by 1000
        if not quantity_already_processed: quantity_list = [process_quantity(quantity, None) for quantity in quantity_list]    # do not need this line because df already has quantity multiplied by 1000; setting the default_quantity to `None` will raise an error further downstream if the default quantity is used which is the desired behavior
        
        return price_cusips_list(cusip_list,
                                 quantity_list, 
                                 trade_type_list,
                                 use_trade_datetime_column_for_pricing=True,
                                 use_for_compliance=True,
                                 user_price_list=user_price_list,
                                 trade_datetime_list=trade_datetime_list,
                                 cusip_with_trade_history_and_reference_data_df=to_be_priced_df_chunk if df_contains_trade_history_and_reference_data else None, 
                                 additional_columns_for_compliance=to_be_priced_df_chunk[additional_columns_for_compliance] if keep_additional_columns_from_original_df else None)

    
    if keep_additional_columns_from_original_df:
        columns_already_used = {'cusip', 'quantity', 'trade_type', 'user_price', 'trade_datetime'}
        additional_columns_for_compliance = [column for column in df.columns if column not in columns_already_used]
        columns_to_add_to_output_csv = [column for column in additional_columns_for_compliance if column not in additional_columns_in_output_csv]
        additional_columns_in_output_csv += columns_to_add_to_output_csv

    num_cpus = os.cpu_count()
    if len(df) > num_cpus:
        print(f'Using multiprocessing inside _price_cusips_point_in_time(...) with num_cpus={num_cpus}')
        with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
            priced_df = pool_object.map(price_cusips_list_func, create_df_chunks(df, num_cpus))
        priced_df = pd.concat(priced_df).reset_index(drop=True)    # need `.reset_index(...)` to reset index from 0...n
    else:
        priced_df = price_cusips_list_func(df)

    priced_df = prepare_batch_pricing_results_to_output_to_user(prepare_batch_pricing_results_for_logging(priced_df), ADDITIONAL_FEATURES_FOR_COMPLIANCE_CSV + additional_columns_in_output_csv)
    priced_df = priced_df.drop(columns=['coupon', 'security_description', 'maturity_date'])    # drop non-essential columns from output
    return priced_df
