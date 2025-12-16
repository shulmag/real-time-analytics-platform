import os
import sys
import pickle
import pandas as pd
import numpy as np  
import redis

from point_in_time_pricing_timestamp import function_timer, price_trades_at_different_quantities_trade_types
from point_in_time_pricing_every_timestamp_from_file import create_business_date_range


server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'app_engine', 'demo', 'server'))    # get the directory containing the 'app_engine/demo/server' package
sys.path.append(server_dir)    # add the directory to sys.path

from modules.auxiliary_variables import USE_SIMILAR_TRADES_MODEL_FOR_YIELD_SPREAD_PREDICTIONS_FOR_POINT_IN_TIME_PRICING

# Set up Redis client for reference data (matching the server configuration)
# REFERENCE_DATA_REDIS_CLIENT = redis.Redis(host='10.108.4.36', port=6379, db=0)
REFERENCE_DATA_REDIS_CLIENT = redis.Redis(host='127.0.0.1', port=6379, db=0)


TESTING = True
TESTING_LIMIT = 50  # Increased to 100 for more comprehensive testing

# Suppress debug prints from server modules for cleaner output
import builtins
original_print = builtins.print
def filtered_print(*args, **kwargs):
    # Filter out DEBUG lines
    if args and isinstance(args[0], str) and args[0].startswith("DEBUG:"):
        return
    return original_print(*args, **kwargs)
builtins.print = filtered_print

DATE_OF_INTEREST = '2025-05-13'
#PICKLE_FILE_PATH = '/Users/gil/git/ficc/notebooks/point_in_time_pricing/processed_data_yield_spread_with_similar_trades_v2_eod_yield_curve_last_duration.pkl'
PICKLE_FILE_PATH = '/Users/gil/git/ficc/notebooks/point_in_time_pricing/processed_data_0513.pkl'
# PICKLE_FILE_PATH = '/Users/gil/git/ficc/notebooks/gil_modeling/analytics_module/processed_data_0522.pkl'


def determine_model_for_processed_data(row):
    """
    Adapt the model selection logic from trade_list_to_array() for processed pickle data.
    
    Returns tuple: (model_to_use, reason_for_using_dollar_price_model)
    """
    
    # Initialize - default to yield spread model
    reason_for_using_dollar_price_model = ''
    model_to_use = 'yield_spread'
    
    # Check 1: Adjustable rate coupon (same logic as trade_list_to_array)
    coupon_type = row.get('coupon_type', None)
    if pd.notna(coupon_type) and coupon_type == 3:
        model_to_use = 'dollar_price'
        reason_for_using_dollar_price_model = 'adjustable_rate_coupon'
        return model_to_use, reason_for_using_dollar_price_model
    
    # Check 2: Missing or negative yields in trade history
    # For processed data, we check the trade_history numpy array
    trade_history = row.get('trade_history', None)
    
    # FIX: Check if trade_history exists and is not None/NaN properly
    if trade_history is not None and not (isinstance(trade_history, float) and pd.isna(trade_history)):
        if isinstance(trade_history, np.ndarray) and trade_history.size > 0:
            try:
                # Based on the sample data structure, yield spread appears to be in column 0
                # The values look like they're in basis points (e.g., 203.026, 142.322, etc.)
                yield_spreads = trade_history[:, 0]  # First column contains yield spreads
                
                # Check for negative or extremely high yields (which indicate missing/bad data)
                # Negative yields or yields > 5000 basis points (50%) are problematic
                has_negative_yields = np.any(yield_spreads < 0)
                has_extreme_yields = np.any(yield_spreads > 5000)  # 50% in basis points
                
                if has_negative_yields or has_extreme_yields:
                    model_to_use = 'dollar_price'
                    reason_for_using_dollar_price_model = 'missing_or_negative_yields'
                    return model_to_use, reason_for_using_dollar_price_model
                    
            except Exception as e:
                print(f"Warning: Could not process trade_history for model selection: {e}")
                # If we can't process trade history, assume there's an issue and use dollar price model
                model_to_use = 'dollar_price'
                reason_for_using_dollar_price_model = 'missing_or_negative_yields'
                return model_to_use, reason_for_using_dollar_price_model
    
    # Check 3: Other conditions from the original get_processed_data_for_single_cusip logic
    
    # Default exists or default indicator
    default_exists = row.get('default_exists', False)
    default_indicator = row.get('default_indicator', False)
    
    if pd.notna(default_exists) and default_exists:
        model_to_use = 'dollar_price'
        reason_for_using_dollar_price_model = 'defaulted'
        return model_to_use, reason_for_using_dollar_price_model
    elif pd.notna(default_indicator) and default_indicator:
        model_to_use = 'dollar_price'
        reason_for_using_dollar_price_model = 'defaulted'
        return model_to_use, reason_for_using_dollar_price_model
    
    # Maturity check (≤ 60 days)
    if pd.notna(row.get('maturity_date')) and pd.notna(row.get('trade_date')):
        try:
            maturity_date = pd.to_datetime(row['maturity_date'])
            trade_date = pd.to_datetime(row['trade_date'])
            days_to_maturity = (maturity_date - trade_date).days
            if days_to_maturity <= 60:
                model_to_use = 'dollar_price'
                reason_for_using_dollar_price_model = 'maturing_soon'
                return model_to_use, reason_for_using_dollar_price_model
        except:
            pass  # If date parsing fails, continue with other checks
    
    # High yield in history check
    if trade_history is not None and not (isinstance(trade_history, float) and pd.isna(trade_history)):
        if isinstance(trade_history, np.ndarray) and trade_history.size > 0:
            try:
                # Check if any yields are >= 15% (1500 basis points)
                yield_spreads = trade_history[:, 0]
                if np.any(yield_spreads >= 1500):  # 15% in basis points
                    model_to_use = 'dollar_price'
                    reason_for_using_dollar_price_model = 'high_yield_in_history'
                    return model_to_use, reason_for_using_dollar_price_model
            except:
                pass  # If we can't check, keep default
    
    # Default: use yield spread model
    return model_to_use, reason_for_using_dollar_price_model


def add_model_selection_columns_adapted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add model_used and reason_for_using_dollar_price_model columns using 
    the adapted logic from trade_list_to_array().
    """
    print("Adding model selection columns using adapted trade_list_to_array logic...")
    
    # Try df.apply() approach first (much faster)
    
    # Apply the model selection logic to each row
    model_results = df.apply(determine_model_for_processed_data, axis=1)
    
    # Unpack the results
    df['model_used'] = [result[0] for result in model_results]
    df['reason_for_using_dollar_price_model'] = [result[1] for result in model_results]
            
    # Print summary
    model_counts = df['model_used'].value_counts()
    reason_counts = df['reason_for_using_dollar_price_model'].value_counts()
    
    print(f"Model selection results:")
    print(f"  yield_spread: {model_counts.get('yield_spread', 0)}")
    print(f"  dollar_price: {model_counts.get('dollar_price', 0)}")
    
    if len(reason_counts) > 1:  # More than just empty strings
        print(f"Reasons for dollar_price model:")
        for reason, count in reason_counts.items():
            if reason:  # Only show non-empty reasons
                print(f"  {reason}: {count}")
    
    return df


@function_timer
def enrich_with_reference_data_from_redis(df: pd.DataFrame, date_of_interest: str) -> pd.DataFrame:
    """
    Fetch specific reference data columns from Redis for all CUSIPs in the DataFrame.
    Starting with call_defeased and timing-related columns that compute_price needs.
    """
    from modules.auxiliary_variables import REFERENCE_DATA_FEATURES
    from modules.point_in_time_pricing import get_point_in_time_reference_data_from_deque
    from datetime import datetime
    
    print(f"Fetching reference data from Redis for {len(df['cusip'].unique())} unique CUSIPs...")
    
    # Get unique CUSIPs
    unique_cusips = df['cusip'].unique().tolist()
    
    # Convert date_of_interest to datetime
    datetime_of_interest = datetime.strptime(date_of_interest, '%Y-%m-%d')
    
    # Fetch reference data from Redis using our local client
    reference_data_pickles = REFERENCE_DATA_REDIS_CLIENT.mget(unique_cusips)
    
    # Unpickle and get point-in-time data
    reference_data_list = []
    for ref_pickle in reference_data_pickles:
        if ref_pickle is not None:
            ref_deque = pickle.loads(ref_pickle)
            ref_data = get_point_in_time_reference_data_from_deque(ref_deque, datetime_of_interest)
            reference_data_list.append(ref_data)
        else:
            reference_data_list.append(None)
    
    # Create mapping of CUSIP to reference data
    cusip_to_ref_data = {}
    for cusip, ref_data in zip(unique_cusips, reference_data_list):
        if ref_data is not None:
            # Convert numpy array to Series with proper column names
            cusip_to_ref_data[cusip] = pd.Series(ref_data, index=REFERENCE_DATA_FEATURES)
    
    # Define columns to fetch (starting with call_defeased, adding timing columns as suggested)
    columns_to_fetch = [
        'call_defeased',
        'call_timing', 
        'call_timing_in_part',
        'sink_timing', 
        'sink_timing_in_part', 
        'put_timing'
    ]
    
    # Add only the specific columns we need
    missing_columns = []
    for col in columns_to_fetch:
        if col not in df.columns and col in REFERENCE_DATA_FEATURES:
            df[col] = df['cusip'].map(lambda x: cusip_to_ref_data.get(x, {}).get(col, None))
            missing_columns.append(col)
    
    print(f"Added {len(missing_columns)} reference data columns from Redis: {missing_columns}")
    
    # Check how many CUSIPs we successfully enriched
    successful_cusips = len([cusip for cusip in unique_cusips if cusip in cusip_to_ref_data])
    print(f"Successfully fetched reference data for {successful_cusips}/{len(unique_cusips)} CUSIPs")
    
    return df


@function_timer
def load_processed_trades_from_pickle(pickle_file_path: str, date_of_interest: str = None) -> pd.DataFrame:
    '''Load pre-processed trades data from a pickle file. 
    If date_of_interest is provided, filter the data to that specific date.'''
    print(f'Loading processed trades data from pickle file: {pickle_file_path}')
    
    if not os.path.exists(pickle_file_path):
        raise FileNotFoundError(f'Pickle file not found: {pickle_file_path}')
    
    with open(pickle_file_path, 'rb') as f:
        trades_df = pickle.load(f)
    
    # Filter by date if specified
    if date_of_interest:
        if 'trade_date' in trades_df.columns:
            original_count = len(trades_df)
            trades_df = trades_df[trades_df['trade_date'] == date_of_interest]
            print(f'Filtered to {len(trades_df)} trades for date {date_of_interest} (from {original_count} total trades)')
        else:
            print(f'Warning: trade_date column not found, cannot filter by date {date_of_interest}')
    
    if len(trades_df) == 0:
        raise ValueError(f'No trades found for the specified criteria')
    
    return trades_df


@function_timer
def validate_and_prepare_trades_data(trades_df: pd.DataFrame, keep_original_quantity_and_trade_type: bool = False) -> pd.DataFrame:
    '''Validate that the pre-processed trades data has the required columns and format it properly.'''
    
    # Required columns that should be present in processed data
    required_columns = ['cusip', 'trade_datetime', 'rtrs_control_number']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in trades_df.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns in trades data: {missing_columns}')
    
    # Remove unnamed index columns that might have been created during CSV export/import
    unnamed_cols = [col for col in trades_df.columns if col.startswith('Unnamed:') or col == '']
    if unnamed_cols:
        print(f"Dropping unnamed columns: {unnamed_cols}")
        trades_df = trades_df.drop(columns=unnamed_cols)
    
    # Ensure 'cusip' is the first column (required by adding_the_cusip_column function)
    if 'cusip' in trades_df.columns:
        cols = trades_df.columns.tolist()
        cols.remove('cusip')
        cols.insert(0, 'cusip')  # Move cusip to first position
        trades_df = trades_df[cols]
        print("Reordered columns to make 'cusip' the first column")
    
    # Handle quantity and trade_type columns
    if keep_original_quantity_and_trade_type:
        if 'par_traded' in trades_df.columns and 'quantity' not in trades_df.columns:
            trades_df = trades_df.rename(columns={'par_traded': 'quantity'})
            trades_df['quantity'] = trades_df['quantity'].astype(int) // 1000
        elif 'quantity' not in trades_df.columns:
            raise ValueError('Neither quantity nor par_traded column found in trades data')
        
        if 'trade_type' not in trades_df.columns:
            raise ValueError('trade_type column not found in trades data when keep_original_quantity_and_trade_type=True')
    else:
        # Drop quantity and trade_type columns if they exist (they will be added later based on QUANTITIES and TRADE_TYPES)
        columns_to_drop = [col for col in ['par_traded', 'trade_type', 'quantity'] if col in trades_df.columns]
        if columns_to_drop:
            trades_df = trades_df.drop(columns=columns_to_drop)
    
    # Sort by publish_datetime if available, otherwise by trade_datetime
    sort_column = 'publish_datetime' if 'publish_datetime' in trades_df.columns else 'trade_datetime'
    trades_df = trades_df.sort_values(by=[sort_column], ignore_index=True)
    
    # Remove duplicates by RTRS control number, keeping the most recent
    trades_df = trades_df.drop_duplicates(subset=['rtrs_control_number'], keep='last')
    
    # Remove duplicates by CUSIP and trade_datetime if not keeping original quantities/trade types
    if not keep_original_quantity_and_trade_type:
        trades_df = trades_df.drop_duplicates(subset=['cusip', 'trade_datetime'])
    
    # Handle trade history columns - set to None if they contain empty/invalid data
    trade_history_columns = ['recent']
    if USE_SIMILAR_TRADES_MODEL_FOR_YIELD_SPREAD_PREDICTIONS_FOR_POINT_IN_TIME_PRICING:
        if 'recent_5_year_mat' in trades_df.columns:
            trades_df = trades_df.rename(columns={'recent_5_year_mat': 'recent_similar'})
            trade_history_columns.append('recent_similar')
    
    for col in trade_history_columns:
        if col in trades_df.columns:
            # Check if trade history data is valid - this is a simplified check
            # In the original code, it checks if the first dictionary in the list has None values
            # Here we'll assume if the column exists but is None/empty, we set it to None
            trades_df[col] = trades_df[col].apply(lambda x: None if pd.isna(x) or x == '' else x)
    
    trades_df = add_model_selection_columns_adapted(trades_df)

    # Convert decimal.Decimal columns to float to avoid type errors in pricing calculations
    import decimal
    print("Converting Decimal columns to float...")
    
    # First, convert known numeric columns
    numeric_columns = [
        'yield', 'coupon', 'par_call_price', 'next_call_price', 'sink_fund_percent', 
        'outstanding_amount', 'put_price', 'sink_fund_price', 'first_call_price',
        'maturity_price', 'redemption_value', 'interest_rate', 'original_offering_price'
    ]
    for col in numeric_columns:
        if col in trades_df.columns:
            trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')
    
    # Also automatically detect and convert any remaining Decimal columns
    decimal_columns = []
    for col in trades_df.columns:
        if trades_df[col].dtype == 'object':  # Check object columns for Decimals
            sample_val = trades_df[col].dropna().iloc[0] if not trades_df[col].dropna().empty else None
            if isinstance(sample_val, decimal.Decimal):
                decimal_columns.append(col)
                trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')
    
    if decimal_columns:
        print(f"Converted additional Decimal columns: {decimal_columns}")
    
    # Add security_description column if missing (required for output CSV)
    if 'security_description' not in trades_df.columns:
        trades_df['security_description'] = 'Not Available'
        print("Added missing 'security_description' column")
    
    # Create trade_history_dollar_price column if missing (required for dollar price model)
    if 'trade_history_dollar_price' not in trades_df.columns and 'trade_history' in trades_df.columns:
        print("Creating 'trade_history_dollar_price' column from 'trade_history'...")
        # For dollar price model, we remove the treasury_spread feature (last column) from each trade
        def create_dollar_price_trade_history(trade_history):
            if trade_history is None or (isinstance(trade_history, float) and pd.isna(trade_history)):
                return None
            if isinstance(trade_history, np.ndarray) and trade_history.size > 0:
                # Remove the last column (treasury_spread) for dollar price model
                return trade_history[:, :-1] if trade_history.shape[1] > 1 else trade_history
            return trade_history
        
        trades_df['trade_history_dollar_price'] = trades_df['trade_history'].apply(create_dollar_price_trade_history)
        print("Created 'trade_history_dollar_price' column")
    
    # Debug: Check trade history structure first
    if 'trade_history' in trades_df.columns:
        sample_trade_history = None
        for idx, row in trades_df.iterrows():
            if row['trade_history'] is not None and isinstance(row['trade_history'], np.ndarray) and row['trade_history'].size > 0:
                sample_trade_history = row['trade_history']
                print(f"Sample trade history shape: {sample_trade_history.shape}")
                print(f"Sample trade history first row: {sample_trade_history[0]}")
                print(f"Number of features per trade: {sample_trade_history.shape[1] if len(sample_trade_history.shape) > 1 else 'N/A'}")
                break
    
    # Check if we already have these features in the DataFrame
    dp_features_check = ['max_dp_dp', 'min_dp_ago', 'max_qty_dp']
    ys_features_check = ['max_ys_ys', 'min_ys_ago', 'max_qty_ys']
    
    has_dp_features = all(col in trades_df.columns for col in dp_features_check)
    has_ys_features = all(col in trades_df.columns for col in ys_features_check)
    
    if has_dp_features and has_ys_features:
        print("Trade history derived features already present in DataFrame")
    else:
        print("WARNING: Trade history derived features are missing and the structure doesn't match expected format")
        print("This pickle file may not have the complete preprocessed data needed for pricing")
        
        # Add placeholder features to allow the pipeline to continue
        # In production, you would need to regenerate the pickle with all features
        from modules.ficc.utils.trade_history_features import get_trade_history_derived_features_for_ys_model, get_trade_history_derived_features_for_dp_model
        
        all_features = get_trade_history_derived_features_for_ys_model() + get_trade_history_derived_features_for_dp_model()
        for feature in all_features:
            if feature not in trades_df.columns:
                trades_df[feature] = 0.0
        
        print("Added placeholder trade history features - RESULTS MAY NOT BE ACCURATE")
    
    print(f'Validated and prepared {len(trades_df)} trades for pricing')
    print(f'Final column order: {list(trades_df.columns[:5])}...')  # Show first 5 columns
    return trades_df.sort_values(by='trade_datetime').reset_index(drop=True)


def price_trades_for_date_of_interest_from_pickle(date_of_interest: str = DATE_OF_INTEREST, 
                                                  pickle_file_path: str = PICKLE_FILE_PATH):
    '''Price trades using pre-processed data from a pickle file.
    Used for Aditya Mothadaka (Deutsche Bank / Elequin Capital) data project. 
    Description: https://docs.google.com/document/d/12WRWis7xlyXG7R7-uO5Z-btreHc7emvWyuPBfrAqraQ/.'''
    
    # Load pre-processed trades data
    if pickle_file_path.endswith('.pkl'):
        trades_on_date_of_interest = load_processed_trades_from_pickle(pickle_file_path, date_of_interest)
    elif pickle_file_path.endswith('.csv'):
        # Simple CSV loader as fallback
        trades_on_date_of_interest = pd.read_csv(pickle_file_path)
        # Filter by date if needed
        if 'trade_date' in trades_on_date_of_interest.columns:
            original_count = len(trades_on_date_of_interest)
            trades_on_date_of_interest = trades_on_date_of_interest[trades_on_date_of_interest['trade_date'] == date_of_interest]
            print(f'Filtered to {len(trades_on_date_of_interest)} trades for date {date_of_interest} (from {original_count} total trades)')
    else:
        raise ValueError('File must be either .pkl or .csv format')
    
    # Apply par_traded threshold if needed (equivalent to original 1M threshold)
    if 'par_traded' in trades_on_date_of_interest.columns:
        original_count = len(trades_on_date_of_interest)
        trades_on_date_of_interest = trades_on_date_of_interest[trades_on_date_of_interest['par_traded'] <= 1_000_000]
        print(f'Applied par_traded threshold: {len(trades_on_date_of_interest)} trades remaining (from {original_count})')
    
    # Limit for testing BEFORE validation and enrichment
    if TESTING and len(trades_on_date_of_interest) > TESTING_LIMIT:
        trades_on_date_of_interest = trades_on_date_of_interest.head(TESTING_LIMIT)
    
    # Validate and prepare the data
    trades_on_date_of_interest = validate_and_prepare_trades_data(trades_on_date_of_interest, 
                                                                  keep_original_quantity_and_trade_type=False)
    
    # Enrich with reference data from Redis
    trades_on_date_of_interest = enrich_with_reference_data_from_redis(trades_on_date_of_interest, date_of_interest)
    
    # Price the trades
    price_trades_at_different_quantities_trade_types(trades_on_date_of_interest, 
                                                   date_of_interest, 
                                                   use_multiprocessing=False,  # Disable multiprocessing for better local performance
                                                   additional_columns_in_output=['trade_datetime', 'rtrs_control_number'], 
                                                   keep_only_essential_columns_in_output=True,
                                                   data_is_fully_processed=True)  # Use the flag for processed data


def price_trades_for_date_of_interest_with_original_quantities_and_trade_types_from_pickle(date_of_interest: str = DATE_OF_INTEREST, 
                                                                                           pickle_file_path: str = PICKLE_FILE_PATH):
    '''Price trades using pre-processed data from a pickle file, preserving original quantities and trade types.
    Used for the Nelson Fernanades (BMO) data project. 
    Description: https://docs.google.com/document/d/1-FXi3AwjvWg0PzhY3ANilizjsptU35ZUA-qhccVLoIs/.
    Used for Solve data project to price all trades from 2024-07 to 2024-10.'''
    
    trades_on_date_of_interest = load_processed_trades_from_pickle(pickle_file_path, date_of_interest)
    # Filter by date if needed
    if 'trade_date' in trades_on_date_of_interest.columns:
        original_count = len(trades_on_date_of_interest)
        trades_on_date_of_interest = trades_on_date_of_interest[trades_on_date_of_interest['trade_date'] == date_of_interest]
        print(f'Filtered to {len(trades_on_date_of_interest)} trades for date {date_of_interest} (from {original_count} total trades)')

    
    # Limit for testing BEFORE validation and enrichment
    if TESTING and len(trades_on_date_of_interest) > TESTING_LIMIT:
        trades_on_date_of_interest = trades_on_date_of_interest.head(TESTING_LIMIT)
    
    # Validate and prepare the data (keeping original quantities and trade types)
    trades_on_date_of_interest = validate_and_prepare_trades_data(trades_on_date_of_interest, 
                                                                  keep_original_quantity_and_trade_type=True)
    
    # Enrich with reference data from Redis
    trades_on_date_of_interest = enrich_with_reference_data_from_redis(trades_on_date_of_interest, date_of_interest)
    
    # Price the trades
    price_trades_at_different_quantities_trade_types(trades_on_date_of_interest, 
                                                   date_of_interest, 
                                                   use_multiprocessing=False,  # Disable multiprocessing for better local performance
                                                   additional_columns_in_output=['trade_datetime', 'rtrs_control_number'], 
                                                   keep_only_essential_columns_in_output=True,
                                                   data_is_fully_processed=True)  # NEW FLAG!


if __name__ == '__main__':
    # Fix for macOS multiprocessing fork issue - must be at the very top of main
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    if TESTING:
        price_trades_for_date_of_interest_with_original_quantities_and_trade_types_from_pickle()
    else:
        # Production usage with pickle file
        price_trades_for_date_of_interest_with_original_quantities_and_trade_types_from_pickle()