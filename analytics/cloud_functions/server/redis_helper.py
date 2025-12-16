"""
Redis helper for ficc yield curve data.
READ-ONLY - do not write to Redis.
"""

import redis
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Redis connection settings (through bastion host)
REDIS_HOST = '10.227.69.60' #'127.0.0.1'  # Connect through local port forwarding | '10.227.69.60' for Production
REDIS_PORT = 6379
REDIS_DB = 0

# Required fields for yield curve calculation
REQUIRED_FIELDS = ['const', 'exponential', 'laguerre']


def get_redis_client():
    """
    Create and return a Redis client connection.
    
    Note: This requires the bastion host to be running with port forwarding:
    1. Start bastion: gcloud compute instances start redis-bastion --project=eng-reactor-287421 --zone=us-central1-c
    2. SSH tunnel: gcloud compute ssh redis-bastion --project=eng-reactor-287421 --zone=us-central1-c --tunnel-through-iap -- -L 6379:10.227.69.60:6379
    
    Returns:
        Redis client object or None if connection fails
    """
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            socket_timeout=5
        )
        # Test connection
        client.ping()
        return client
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        return None

def get_latest_key(client):
    """
    Get the most recent key in Redis.
    
    Args:
        client: Redis client object
        
    Returns:
        String containing the most recent key or None if no keys found
    """
    all_keys = client.keys('2*')  # Keys starting with a year
    if not all_keys:
        return None
    
    # Sort by timestamp (most recent first)
    sorted_keys = sorted([k.decode('utf-8') for k in all_keys], reverse=True)
    return sorted_keys[0]

def get_keys_for_day(client, date):
    """
    Get all keys for a specific date in Redis.
    
    Args:
        client: Redis client object
        date: Date string in YYYY-MM-DD format
        
    Returns:
        List of keys for the given date, sorted by time (empty list if no keys found)
    """
    date_pattern = f"{date}:*"
    day_keys = client.keys(date_pattern)
    
    if not day_keys:
        return []
    
    # Sort keys by timestamp
    sorted_keys = sorted([k.decode('utf-8') for k in day_keys])
    return sorted_keys

def get_keys_for_days(client, dates):
    """
    Get all keys for multiple dates in Redis.
    
    Args:
        client: Redis client object
        dates: List of date strings in YYYY-MM-DD format
        
    Returns:
        Dictionary with dates as keys and lists of Redis keys as values
        (only includes dates that have keys, empty dict if no keys found)
    """
    result = {}
    
    for date in dates:
        keys = get_keys_for_day(client, date)
        if keys:
            result[date] = keys
    
    return result

def get_yield_data_from_key(client, key):
    """
    Get yield curve data from a specific Redis key.
    Returns the unpickled data and the Redis key used.
    
    Args:
        client: Redis client object
        key: Redis key to fetch
        
    Returns:
        Tuple of (unpickled_data, key) where unpickled_data is the parsed data structure
        or None if data could not be retrieved or parsed
    
    The expected Redis data structure is:
    {
        'nelson_values': DataFrame with columns ['const', 'exponential', 'laguerre'],
        'scalar_values': Series with values [exponential_mean, exponential_std, laguerre_mean, laguerre_std],
        'shape_parameter': float
    }
    """
    data = client.get(key)
    if not data:
        return None, key
    
    # Unpickle the data
    try:
        unpickled_data = pickle.loads(data)
        
        # Check if it's a dict with required values
        if isinstance(unpickled_data, dict) and 'nelson_values' in unpickled_data:
            if 'scalar_values' not in unpickled_data:
                print(f"Warning: 'scalar_values' missing for key {key}")
            if 'shape_parameter' not in unpickled_data:
                print(f"Warning: 'shape_parameter' missing for key {key}")
                
            # Print what we have for debugging
            print(f"Data for key {key}:")
            print(f"  nelson_values shape: {unpickled_data['nelson_values'].shape}")
            print(f"  nelson_values columns: {unpickled_data['nelson_values'].columns.tolist()}")
            
            if 'scalar_values' in unpickled_data:
                print(f"  scalar_values: {unpickled_data['scalar_values'].tolist() if hasattr(unpickled_data['scalar_values'], 'tolist') else unpickled_data['scalar_values']}")
            
            if 'shape_parameter' in unpickled_data:
                print(f"  shape_parameter: {unpickled_data['shape_parameter']}")
                
            return unpickled_data, key
    except Exception as e:
        print(f"Error unpickling data for key {key}: {e}")
    
    return None, key

def get_yield_data_from_keys(client, keys):
    """
    Get yield curve data from multiple Redis keys using mget.
    
    Args:
        client: Redis client object
        keys: List of Redis keys to fetch
        
    Returns:
        Dictionary mapping keys to their unpickled data
        (only includes successfully unpickled data, empty dict if no valid data)
    """
    if not keys:
        return {}
    
    # Get all data in one batch operation
    batch_data = client.mget(keys)
    
    result = {}
    for i, key in enumerate(keys):
        data = batch_data[i]
        
        if data:
            try:
                unpickled_data = pickle.loads(data)
                
                # Check if it's a dict with nelson_values
                if isinstance(unpickled_data, dict) and 'nelson_values' in unpickled_data:
                    result[key] = unpickled_data
            except Exception as e:
                print(f"Error unpickling data for key {key}: {e}")
    
    return result

def get_yield_curve(client=None, maturities=None):
    """
    Get yield curve data for a single timestamp.
    
    Args:
        client: Redis client object (will be created if None)
        maturities: List of maturities to include (must be provided)
        
    Returns:
        Dict with timestamp and yield values for each maturity or None if data unavailable
        Example: {'timestamp': '2025-05-14:10:30', 'values': {'5': 3.45, '10': 3.78}}
    """
    if maturities is None:
        return None  # Must provide maturities, no defaults
    
    # Make sure all maturities are integers
    maturities = [int(m) for m in maturities]
    
    # Get Redis client if not provided
    if client is None:
        client = get_redis_client()
        if client is None:
            print("Failed to connect to Redis")
            return None
    
    # Get the most recent key
    key = get_latest_key(client)
    if not key:
        print("No keys found in Redis")
        return None
    
    # Get the data for this key
    data, key = get_yield_data_from_key(client, key)
    if not data:
        print(f"No valid data found for key: {key}")
        return None
    
    # Get the nelson_values from the data
    nelson_values = data['nelson_values']
    
    # Get parameters from the first row
    row = nelson_values.iloc[0]
    
    # Check if we have all required parameters
    if 'const' not in row or 'exponential' not in row or 'laguerre' not in row:
        print(f"Missing required parameters in row data")
        return None
    
    # Calculate yield for each maturity
    result = {
        'timestamp': key,
        'values': {}
    }
    
    # Prepare the model parameters
    params = {
        'const': row['const'],
        'exponential': row['exponential'],
        'laguerre': row['laguerre']
    }
        
    # We need shape and scalar parameters
    if 'shape_parameter' not in data:
        print(f"Missing shape parameter in data")
        return None
    
    if 'scalar_values' not in data or not hasattr(data['scalar_values'], 'tolist'):
        print(f"Missing or invalid scalar values in data")
        return None
        
    scalar_list = data['scalar_values'].tolist()
    if len(scalar_list) < 4:
        print(f"Insufficient scalar values: {len(scalar_list)}")
        return None
        
    # Extract parameters
    shape_parameter = float(data['shape_parameter'])
    exponential_mean = scalar_list[0]
    exponential_std = scalar_list[1]
    laguerre_mean = scalar_list[2]
    laguerre_std = scalar_list[3]
    
    # Calculate yield for each maturity
    for maturity in maturities:
        # Use Nelson-Siegel model formula
        t = float(maturity)
        
        # Calculate the decay and laguerre functions
        decay = shape_parameter * (1 - np.exp(-t/shape_parameter)) / t
        lag = (shape_parameter * (1 - np.exp(-t/shape_parameter)) / t) - np.exp(-t/shape_parameter)
        
        # Scale the factors
        X1 = (decay - exponential_mean) / exponential_std
        X2 = (lag - laguerre_mean) / laguerre_std
        
        # Calculate yield
        yield_value = params['const'] + params['exponential'] * X1 + params['laguerre'] * X2
        
        # Store the result
        result['values'][str(maturity)] = float(yield_value)
    
    return result

def calculate_yield_values(nelson_params, maturities):
    """
    Calculate yield values for given maturities using Nelson-Siegel parameters.
    
    Args:
        nelson_params: Dictionary with Nelson-Siegel parameters
            (must contain 'nelson_values' DataFrame with const, exponential, laguerre,
             'scalar_values' with mean/std values, and 'shape_parameter')
        maturities: List of maturities to calculate yields for
        
    Returns:
        Dictionary mapping maturities (as strings) to yield values (as floats)
        Example: {'5': 3.45, '10': 3.78, '15': 4.12}
        Returns None if any required parameters are missing
    """
    # Check for required components
    if 'nelson_values' not in nelson_params:
        print("Missing nelson_values in parameters")
        return None
        
    if 'scalar_values' not in nelson_params or not hasattr(nelson_params['scalar_values'], 'tolist'):
        print("Missing or invalid scalar_values in parameters")
        return None
        
    if 'shape_parameter' not in nelson_params:
        print("Missing shape_parameter in parameters")
        return None
    
    # Get parameters from the data
    row = nelson_params['nelson_values'].iloc[0]
    scalar_list = nelson_params['scalar_values'].tolist()
    shape_parameter = float(nelson_params['shape_parameter'])
    
    # Check if we have all required row values
    if 'const' not in row or 'exponential' not in row or 'laguerre' not in row:
        print("Missing required parameters in nelson_values")
        return None
        
    # Check scalar list length
    if len(scalar_list) < 4:
        print(f"Insufficient scalar values: {len(scalar_list)}")
        return None
    
    # Extract parameters
    const = row['const']
    exponential = row['exponential']
    laguerre = row['laguerre']
    exponential_mean = scalar_list[0]
    exponential_std = scalar_list[1]
    laguerre_mean = scalar_list[2]
    laguerre_std = scalar_list[3]
        
    # Calculate yield for each maturity
    result = {}
    for maturity in maturities:
        # Use Nelson-Siegel model formula
        t = float(maturity)
        
        # Calculate the decay and laguerre functions
        decay = shape_parameter * (1 - np.exp(-t/shape_parameter)) / t
        lag = (shape_parameter * (1 - np.exp(-t/shape_parameter)) / t) - np.exp(-t/shape_parameter)
        
        # Scale the factors
        X1 = (decay - exponential_mean) / exponential_std
        X2 = (lag - laguerre_mean) / laguerre_std
        
        # Calculate yield
        yield_value = const + exponential * X1 + laguerre * X2
        
        # Store the result
        result[str(maturity)] = float(yield_value)
    
    return result

def get_yield_curves(maturities=None, prev_business_day=None, today_date=None):
    """
    Get yield curve data from Redis for multiple timestamps.
    This returns the raw Nelson-Siegel parameters so that main.py can use 
    FiccYieldCurve.py to calculate the yields.
    
    Args:
        maturities: List of maturities to include (default [5, 10, 15, 20])
        prev_business_day: Previous business day datetime
        today_date: Today's datetime
        
    Returns:
        List of dicts containing timestamps and their Nelson-Siegel parameters
        Each dict has format: {'timestamp': 'YYYY-MM-DD:HH:MM', 'params': {...}}
        where params contains the model parameters needed for yield calculation
        Returns None if no data available or connection fails
    """
    print(f">>> Fetching yield curves for {maturities} maturities, {prev_business_day} and {today_date}")

    client = get_redis_client()
    if client is None:
        print("Failed to connect to Redis")
        return None
    
    today = today_date.strftime('%Y-%m-%d')
    yesterday = prev_business_day.strftime('%Y-%m-%d')
    
    print(f"Fetching data for today ({today}) and yesterday ({yesterday})")
    
    # Get all keys for today and yesterday
    days_keys = get_keys_for_days(client, [today, yesterday])
    
    if not days_keys:
        print("No keys found for today or yesterday")
        return None
    
    # Flatten the keys list and fetch all data in one batch
    all_keys = []
    for day_keys in days_keys.values():
        all_keys.extend(day_keys)
    
    # Get yield data for all keys
    all_data = get_yield_data_from_keys(client, all_keys)
    
    if not all_data:
        print("No yield data found")
        return None
    
    # Print summary of data found
    print(f"Found data for {len(all_data)} timestamps:")
    for day, day_keys in days_keys.items():
        data_keys = [k for k in day_keys if k in all_data]
        print(f" ***  {day}: {len(data_keys)} data points ***")
    
    # Prepare the data for return to main.py
    # We need to extract all parameters needed by FiccYieldCurve.predict_ytw()
    redis_data = []
    for key, data in all_data.items():
        # Get nelson_values (contains const, exponential, laguerre)
        nelson_values = data['nelson_values']
        row = nelson_values.iloc[0]
        
        # Get scalar_values (contains exponential_mean, exponential_std, laguerre_mean, laguerre_std)
        if 'scalar_values' not in data:
            print(f"Warning: scalar_values missing for key {key}")
            return None
            
        scalar_values = data['scalar_values']
        if not hasattr(scalar_values, 'tolist'):
            print(f"Warning: scalar_values has unexpected type: {type(scalar_values)}")
            return None
            
        # Only use data if it has the expected format and length
        scalar_list = scalar_values.tolist()
        if len(scalar_list) < 4:
            print(f"Warning: scalar_values has unexpected length: {len(scalar_list)}")
            return None
            
        exponential_mean = scalar_list[0]
        exponential_std = scalar_list[1]
        laguerre_mean = scalar_list[2]
        laguerre_std = scalar_list[3]
        
        # Get shape_parameter
        if 'shape_parameter' not in data:
            print(f"Warning: shape_parameter missing for key {key}")
            return None
            
        shape_parameter = float(data['shape_parameter'])
        
        # Store all parameters needed for predict_ytw
        data_point = {
            'timestamp': key,
            'params': {
                # Nelson-Siegel coefficients
                'const': float(row['const']),
                'exponential': float(row['exponential']),
                'laguerre': float(row['laguerre']),
                
                # Scalar parameters
                'exponential_mean': float(exponential_mean),
                'exponential_std': float(exponential_std),
                'laguerre_mean': float(laguerre_mean),
                'laguerre_std': float(laguerre_std),
                
                # Shape parameter
                'shape_parameter': float(shape_parameter)
            }
        }
        
        redis_data.append(data_point)
    
    # Sort results by timestamp
    redis_data.sort(key=lambda x: x['timestamp'])
    
    print(f"Total timestamps with Nelson-Siegel parameters: {len(redis_data)}")
    
    # Print some sample data points
    if redis_data and len(redis_data) > 0:
        print("\nSample data point:")
        i = 0
        print(f"  {redis_data[i]['timestamp']}:")
        for param, value in redis_data[i]['params'].items():
            print(f"    {param}: {value}")
    
    return redis_data
    
# CoD functionality removed - now calculated on the frontend using yield-curves data