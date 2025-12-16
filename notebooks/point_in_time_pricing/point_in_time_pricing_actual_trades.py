'''
Description: Point in time pricing actual trades

             **NOTE**: This script needs to be run on a VM so that the yield curve redis can be accessed which is necessary for `process_data(...)`. The error that will be raised otherwise is a `TimeoutError`.
             **NOTE**: Use `point_in_time_pricing_requirements.txt` to install the requirements for this script. 
             **NOTE**: To see the output of this script in an `output.txt` file use the command: $ python -u point_in_time_pricing_actual_trades.py >> output.txt. 

             This script allows one to see what prices we would have returned on a specified date for a trades that actually occurred. The user specifies a `DATE_OF_INTEREST` and chooses either
             (1) `price_trades_for_date_of_interest(...)`: prices each trade at the time of the trade on the `DATE_OF_INTEREST` for each quantity specified in `point_in_time_pricing_timestamp.py::QUANTITIES` and each direction specified in `point_in_time_pricing_timestamp.py::TRADE_TYPES`
             (2) `price_trades_for_date_of_interest_with_original_quantities_and_trade_types(...)`: prices each trade at the time of the trade on the `DATE_OF_INTEREST` for the original quantity and direction of the trade

             The workhorse functions are in `point_in_time_pricing_timestamp.py`.'''
import os
import sys
import pickle
import pandas as pd

from point_in_time_pricing_timestamp import function_timer, convert_decimal_to_float, price_trades_at_different_quantities_trade_types
from point_in_time_pricing_every_timestamp_from_file import create_business_date_range


server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'app_engine', 'demo', 'server'))    # get the directory containing the 'app_engine/demo/server' package
sys.path.append(server_dir)    # add the directory to sys.path

from modules.auxiliary_variables import bq_client, USE_SIMILAR_TRADES_MODEL_FOR_YIELD_SPREAD_PREDICTIONS_FOR_POINT_IN_TIME_PRICING
from modules.ficc.utils.auxiliary_functions import sqltodf


TESTING = False
TESTING_QUERY_LIMIT = 100

DATE_OF_INTEREST = '2025-05-09'

# TODO: prune this list to contain only the necessary features
FEATURES_NEEDED_FOR_PREDICTION = ['cusip', 
                                   'coupon', 
                                   'series_id', 
                                   'security_description', 
                                #    'ref_valid_from_date', 
                                #    'ref_valid_to_date', 
                                   'incorporated_state_code', 
                                   'organization_primary_name', 
                                #    'instrument_primary_name', 
                                   'issue_key', 
                                   'issue_text', 
                                   'conduit_obligor_name', 
                                   'is_called', 
                                   'is_callable', 
                                #    'is_escrowed_or_pre_refunded', 
                                #    'first_call_date', 
                                #    'call_date_notice', 
                                   'callable_at_cav', 
                                   'par_price', 
                                   'call_defeased', 
                                   'call_timing', 
                                   'call_timing_in_part', 
                                   'extraordinary_make_whole_call', 
                                #    'extraordinary_redemption', 
                                   'make_whole_call', 
                                   'next_call_date', 
                                   'next_call_price', 
                                #    'call_redemption_id', 
                                #    'first_optional_redemption_code', 
                                #    'second_optional_redemption_code', 
                                #    'third_optional_redemption_code', 
                                #    'first_mandatory_redemption_code', 
                                #    'second_mandatory_redemption_code', 
                                #    'third_mandatory_redemption_code', 
                                   'par_call_date', 
                                   'par_call_price', 
                                #    'maximum_call_notice_period', 
                                   'called_redemption_type', 
                                #    'muni_issue_type', 
                                   'refund_date', 
                                   'refund_price', 
                                #    'redemption_cav_flag', 
                                #    'max_notification_days', 
                                #    'min_notification_days', 
                                #    'next_put_date', 
                                #    'put_end_date', 
                                #    'put_feature_price', 
                                #    'put_frequency', 
                                #    'put_start_date', 
                                #    'put_type', 
                                   'maturity_date', 
                                   'sp_long', 
                                #    'sp_stand_alone', 
                                #    'sp_icr_school', 
                                #    'sp_prelim_long', 
                                #    'sp_outlook_long', 
                                #    'sp_watch_long', 
                                #    'sp_Short_Rating', 
                                #    'sp_Credit_Watch_Short_Rating', 
                                #    'sp_Recovery_Long_Rating', 
                                #    'moodys_long', 
                                #    'moodys_short', 
                                #    'moodys_Issue_Long_Rating', 
                                #    'moodys_Issue_Short_Rating', 
                                #    'moodys_Credit_Watch_Long_Rating', 
                                #    'moodys_Credit_Watch_Short_Rating', 
                                #    'moodys_Enhanced_Long_Rating', 
                                #    'moodys_Enhanced_Short_Rating', 
                                #    'moodys_Credit_Watch_Long_Outlook_Rating', 
                                #    'has_sink_schedule', 
                                   'next_sink_date', 
                                   'sink_indicator', 
                                #    'sink_amount_type_text', 
                                #    'sink_amount_type_type', 
                                   'sink_frequency', 
                                #    'sink_defeased', 
                                #    'additional_next_sink_date', 
                                   'sink_amount_type', 
                                #    'additional_sink_frequency', 
                                   'min_amount_outstanding', 
                                   'max_amount_outstanding', 
                                   'default_exists', 
                                   'has_unexpired_lines_of_credit', 
                                #    'years_to_loc_expiration', 
                                   'escrow_exists', 
                                #    'escrow_obligation_percent', 
                                #    'escrow_obligation_agent', 
                                #    'escrow_obligation_type', 
                                #    'child_linkage_exists', 
                                #    'put_exists', 
                                #    'floating_rate_exists', 
                                   'bond_insurance_exists', 
                                   'is_general_obligation', 
                                   'has_zero_coupons', 
                                   'delivery_date', 
                                   'issue_price', 
                                #    'primary_market_settlement_date', 
                                   'issue_date', 
                                   'outstanding_indicator', 
                                   'federal_tax_status', 
                                   'maturity_amount', 
                                #    'available_denom', 
                                #    'denom_increment_amount', 
                                #    'min_denom_amount', 
                                   'accrual_date', 
                                   'bond_insurance', 
                                   'coupon_type', 
                                   'current_coupon_rate', 
                                #    'daycount_basis_type', 
                                   'debt_type', 
                                   'default_indicator', 
                                   'first_coupon_date', 
                                   'interest_payment_frequency', 
                                   'issue_amount', 
                                   'last_period_accrues_from_date', 
                                   'next_coupon_payment_date', 
                                #    'odd_first_coupon_date', 
                                   'orig_principal_amount', 
                                   'original_yield', 
                                   'outstanding_amount', 
                                   'previous_coupon_payment_date', 
                                   'sale_type', 
                                   'settlement_type', 
                                #    'additional_project_txt', 
                                   'asset_claim_code', 
                                   'additional_state_code', 
                                #    'backed_underlying_security_id', 
                                #    'bank_qualified', 
                                   'capital_type', 
                                #    'conditional_call_date', 
                                #    'conditional_call_price', 
                                #    'designated_termination_date', 
                                #    'DTCC_status', 
                                #    'first_execution_date', 
                                #    'formal_award_date', 
                                   'maturity_description_code', 
                                   'muni_security_type', 
                                #    'mtg_insurance', 
                                #    'orig_cusip_status', 
                                #    'orig_instrument_enhancement_type', 
                                #    'other_enhancement_type', 
                                #    'other_enhancement_company', 
                                #    'pac_bond_indicator', 
                                #    'project_name', 
                                   'purpose_class', 
                                   'purpose_sub_class', 
                                #    'refunding_issue_key', 
                                #    'refunding_dated_date', 
                                #    'sale_date', 
                                #    'sec_regulation', 
                                #    'secured', 
                                   'series_name', 
                                #    'sink_fund_redemption_method', 
                                   'state_tax_status', 
                                #    'tax_credit_frequency', 
                                #    'tax_credit_percent', 
                                   'use_of_proceeds', 
                                #    'use_of_proceeds_supplementary', 
                                #    'rating_downgrade', 
                                #    'rating_upgrade', 
                                #    'rating_downgrade_to_junk', 
                                #    'min_sp_rating_this_year', 
                                #    'max_sp_rating_this_year', 
                                #    'min_moodys_rating_this_year', 
                                #    'max_moodys_rating_this_year', 
                                   'trade_date', 
                                   'trade_datetime', 
                                   'publish_datetime', 
                                   'rtrs_control_number', 
                                   'recent', 
                                   'par_traded', 
                                   'trade_type']
if USE_SIMILAR_TRADES_MODEL_FOR_YIELD_SPREAD_PREDICTIONS_FOR_POINT_IN_TIME_PRICING: FEATURES_NEEDED_FOR_PREDICTION.append('recent_5_year_mat')    # `recent_5_year_mat` is the name given to the similar trade history in the BigQuery table auxiliary_views_v2.trade_history_same_issue_5_yr_mat_bucket_1_materialized


@function_timer
def set_trade_history_to_none_for_trades_with_no_history(trades_df: pd.DataFrame) -> pd.DataFrame:
    '''Sets the trade history column, `recent`, to `None` instead of what comes from Bigquery, 
    which is a list of dictionaries, and to represent an empty history, this list of dictionaries 
    is a list with a single dictionary where all of the values are `None` except for `current_rtrs_control_number`.
    NOTE: decided to not use multiprocessing since this procedure takes about 5 seconds for 40k lines in `trades_df`.'''
    trade_history_column_names = ['recent']
    if USE_SIMILAR_TRADES_MODEL_FOR_YIELD_SPREAD_PREDICTIONS_FOR_POINT_IN_TIME_PRICING:
        trades_df = trades_df.rename(columns={'recent_5_year_mat': 'recent_similar'})    # `recent_similar` is the name given to the similar trade history in the server code
        trade_history_column_names.append('recent_similar')
    for trade_history_column_name in trade_history_column_names:
        indices_with_empty_trade_history = [trade_idx for trade_idx, trade in trades_df.iterrows() if len(trade[trade_history_column_name]) == 0 or pd.isna(trade[trade_history_column_name][0]['rtrs_control_number'])]
        trades_df.loc[indices_with_empty_trade_history, trade_history_column_name] = None
    return trades_df


def get_trades_on_date_of_interest_query(date_or_dates_of_interest, par_traded_threshold: int = None, include_yield_and_price: bool = False) -> str:
    '''Retruns the query to get the trades on the date or dates of interest. `date_or_dates_of_interest` could be a 
    single date in which case it would be a string, or a list of dates in which case it would be a list of strings.'''
    par_traded_threshold_query_suffix = '' if par_traded_threshold is None else f' AND par_traded <= {par_traded_threshold}'
    testing_suffix = f' LIMIT {TESTING_QUERY_LIMIT}' if TESTING else ''

    assert type(date_or_dates_of_interest) == list or type(date_or_dates_of_interest) == str, f'Incorrect type for `date_or_dates_of_interest`: {type(date_or_dates_of_interest)}, must be list or string'
    if type(date_or_dates_of_interest) == list:    # list of dates
        date_condition = f'IN {tuple(date_or_dates_of_interest)}'
    else:    # single date
        date_condition = f'= "{date_or_dates_of_interest}"'
    date_condition = f'trade_date {date_condition} AND publish_date {date_condition}'

    yield_and_price_addendum = ['yield', 'dollar_price'] if include_yield_and_price else []
    materialized_trades_table_name = 'trade_history_same_issue_5_yr_mat_bucket_1_materialized'
    trades_on_date_of_interest_query = f'''SELECT {', '.join(FEATURES_NEEDED_FOR_PREDICTION + yield_and_price_addendum)} FROM `auxiliary_views_v2.{materialized_trades_table_name}` WHERE {date_condition}{par_traded_threshold_query_suffix}{testing_suffix}'''
    print('query from `get_trades_on_date_of_interest_query(...)`:\n', trades_on_date_of_interest_query)
    return trades_on_date_of_interest_query


@function_timer
def get_trades_on_date_of_interest_from_query(query: str, date_or_dates_of_interest, save_data: bool = True) -> pd.DataFrame:
    '''Get trades from `query`. `date_or_dates_of_interest` is used to create the pickle file name to store the results 
    afterwards or to retrieve the values before calling the query if they already exist.'''
    assert type(date_or_dates_of_interest) == list or type(date_or_dates_of_interest) == str, f'Incorrect type for `date_or_dates_of_interest`: {type(date_or_dates_of_interest)}, must be list or string'
    if type(date_or_dates_of_interest) == list:    # list of dates
        date_or_dates_of_interest = sorted(date_or_dates_of_interest)
        pickle_filename_suffix = date_or_dates_of_interest[0] + '--' + date_or_dates_of_interest[-1]    # keep just the first and last dates (in case there are many dates, the filename may become too long to read if all the dates are kept)
    else:    # single date
        pickle_filename_suffix = date_or_dates_of_interest

    saved_query = None
    saved_query_and_result_df_filename = f'query_and_trades_df_{pickle_filename_suffix}.pkl'
    if os.path.exists(saved_query_and_result_df_filename):
        print(f'Found {saved_query_and_result_df_filename} so will try to load the pickle file')
        with open(saved_query_and_result_df_filename, 'rb') as f:
            saved_query, result_df = pickle.load(f)
    if saved_query == query:
        print(f'Saved query matched the desired query so will load the dataframe')
        trades_on_date_of_interest = result_df
    else:
        if saved_query is not None: print(f'Found a saved query, but it did not match the current query. Saved query:\n{saved_query}')
        trades_on_date_of_interest = sqltodf(query, bq_client)
        if save_data:
            print(f'Saving the query and dataframe in the pickle file: {saved_query_and_result_df_filename}')
            with open(saved_query_and_result_df_filename, 'wb') as f:
                pickle.dump((query, trades_on_date_of_interest), f)
    return trades_on_date_of_interest


def process_trades_from_materialized_trade_history(trades_df: pd.DataFrame, keep_original_quantity_and_trade_type: bool = False) -> pd.DataFrame:
    '''Perform general processing on the dataframe of trades `trades_df` after getting the 
    results from the materialized trade history BigQuery table.'''
    if not keep_original_quantity_and_trade_type: trades_df = trades_df.drop(columns=['par_traded', 'trade_type'])    # drop these columns later instead of changing the query so that the pickled query and dataframe are more universal
    if 'par_traded' in trades_df.columns:
        trades_df = trades_df.rename(columns={'par_traded': 'quantity'})
        trades_df['quantity'] = trades_df['quantity'].astype(int) // 1000
    trades_df = trades_df.sort_values(by=['publish_datetime'], ignore_index=True)    # setting `ignore_index` to `True` results in the index being labeled 0, 1, …, n - 1
    trades_df = trades_df.drop_duplicates(subset=['rtrs_control_number'], keep='last')    # keep the RTRS control number corresponding to the most recent publish datetime (since we sorted by publish datetime)
    if not keep_original_quantity_and_trade_type: trades_df = trades_df.drop_duplicates(subset=['cusip', 'trade_datetime'])    # keep only one copy of the row for all rows where the CUSIP and trade_datetime are the same (the same CUSIP traded many times at a particular datetime, but we do not need to keep all of them)
    trades_df = set_trade_history_to_none_for_trades_with_no_history(trades_df)
    return trades_df


@function_timer
def get_processed_trades_on_date_of_interest(date_of_interest: str, par_traded_threshold: int = None, keep_original_quantity_and_trade_type: bool = False) -> pd.DataFrame:
    '''Query the appropriate BigQuery table to get all of the trades 
    that occurred on the date `date_of_interest`, and keep the most recent RTRS control 
    number corresponding to the `publish_datetime`. For a given day, we want to keep only 
    the last published trade_message for a given rtrs_control_number/trade as of end of day 
    on the day of interest. We do this by first choosing the trade_messages published on 
    the trade_date, then selecting the most recent for a given trade/rtrs_control_number.'''
    trades_on_date_of_interest_query = get_trades_on_date_of_interest_query(date_of_interest, par_traded_threshold)
    trades_on_date_of_interest = get_trades_on_date_of_interest_from_query(trades_on_date_of_interest_query, date_of_interest)
    assert len(trades_on_date_of_interest) > 0, f'No trades found for the below query:\n{trades_on_date_of_interest_query}'
    trades_on_date_of_interest = process_trades_from_materialized_trade_history(trades_on_date_of_interest, keep_original_quantity_and_trade_type)
    trades_on_date_of_interest = convert_decimal_to_float(trades_on_date_of_interest)    # convert the columns that are decimal.Decimal type which comes from BigQuery to float
    return trades_on_date_of_interest.sort_values(by='trade_datetime').reset_index(drop=True)    # sort by `trade_datetime` for easier debugging


def price_trades_for_date_of_interest(date_of_interest: str = DATE_OF_INTEREST):
    '''Used for Aditya Mothadaka (Deutsche Bank / Elequin Capital) data project. Description: https://docs.google.com/document/d/12WRWis7xlyXG7R7-uO5Z-btreHc7emvWyuPBfrAqraQ/.'''
    trades_on_date_of_interest = get_processed_trades_on_date_of_interest(date_of_interest, par_traded_threshold=1_000_000)    # using an underscore in a integer is for readability purposes
    price_trades_at_different_quantities_trade_types(trades_on_date_of_interest, date_of_interest, additional_columns_in_output=['trade_datetime', 'rtrs_control_number'], keep_only_essential_columns_in_output=True)


def price_trades_for_date_of_interest_with_original_quantities_and_trade_types(date_of_interest: str = DATE_OF_INTEREST):
    '''Used for the Nelson Fernanades (BMO) data project. Description: https://docs.google.com/document/d/1-FXi3AwjvWg0PzhY3ANilizjsptU35ZUA-qhccVLoIs/.
    Used for Solve data project to price all trades from 2024-07 to 2024-10.'''
    trades_on_date_of_interest = get_processed_trades_on_date_of_interest(date_of_interest, keep_original_quantity_and_trade_type=True)
    price_trades_at_different_quantities_trade_types(trades_on_date_of_interest, date_of_interest, additional_columns_in_output=['trade_datetime', 'rtrs_control_number'], keep_only_essential_columns_in_output=True)


if __name__ == '__main__':
    if TESTING:
        # price_trades_for_date_of_interest()
        price_trades_for_date_of_interest_with_original_quantities_and_trade_types()
    else:
        price_trades_for_date_of_interest_with_original_quantities_and_trade_types()
        # for date in create_business_date_range('2024-07-01', '2024-10-28'):
        #     price_trades_for_date_of_interest(date)
