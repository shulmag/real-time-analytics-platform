"""
Municipal Bond Embedding Helper Functions
Core utilities for Siamese network training on municipal bond data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
import warnings
from typing import List, Tuple, Dict, Optional, Any

# ===========================
# CONSTANTS
# ===========================

NUM_OF_DAYS_IN_YEAR = 360  # MSRB convention

# Coupon frequency mapping
COUPON_FREQUENCY_DICT = {
    0: 'Unknown', 1: 'Semiannually', 2: 'Monthly', 3: 'Annually', 4: 'Weekly',
    5: 'Quarterly', 6: 'Every 2 years', 7: 'Every 3 years', 8: 'Every 4 years',
    9: 'Every 5 years', 10: 'Every 7 years', 11: 'Every 8 years', 12: 'Biweekly',
    13: 'Changeable', 14: 'Daily', 15: 'Term mode', 16: 'Interest at maturity',
    17: 'Bimonthly', 18: 'Every 13 weeks', 19: 'Irregular', 20: 'Every 28 days',
    21: 'Every 35 days', 22: 'Every 26 weeks', 23: 'Not Applicable', 24: 'Tied to prime',
    25: 'One time', 26: 'Every 10 years', 27: 'Frequency to be determined',
    28: 'Mandatory put', 29: 'Every 52 weeks', 30: 'When interest adjusts-commercial paper',
    31: 'Zero coupon', 32: 'Certain years only', 33: 'Under certain circumstances',
    34: 'Every 15 years', 35: 'Custom', 36: 'Single Interest Payment'
}

COUPON_FREQUENCY_TYPE = {
    'Unknown': 1e6, 'Semiannually': 2, 'Monthly': 12, 'Annually': 1,
    'Weekly': 52, 'Quarterly': 4, 'Every 2 years': 0.5, 'Every 3 years': 1/3,
    'Every 4 years': 0.25, 'Every 5 years': 0.2, 'Every 7 years': 1/7,
    'Every 8 years': 1/8, 'Every 10 years': 1/10, 'Biweekly': 26,
    'Changeable': 44, 'Daily': 360, 'Term mode': 0, 'Interest at maturity': 0, 
    'Not Applicable': 1e6, 'Zero coupon': 0, 'Bimonthly': 6, 'Every 13 weeks': 4, 
    'Irregular': 1e6, 'Every 28 days': 13, 'Every 35 days': 10.3, 'Every 26 weeks': 2,
    'Every 52 weeks': 1, 'When interest adjusts-commercial paper': 1e6,
    'Certain years only': 1e6, 'Under certain circumstances': 1e6,
    'Every 15 years': 1/15, 'Custom': 1e6, 'Single Interest Payment': 0,
    'Mandatory put': 1e6, 'Tied to prime': 1e6, 'Frequency to be determined': 1e6
}

RATING_SCORE_MAP = {
    "AAA": 22, "AA+": 21, "AA": 20, "AA-": 19,
    "A+": 18, "A": 17, "A-": 16,
    "BBB+": 15, "BBB": 14, "BBB-": 13,
    "BB+": 12, "BB": 11, "BB-": 10,
    "B+": 9, "B": 8, "B-": 7,
    "CCC+": 6, "CCC": 5, "CCC-": 4,
    "CC": 3, "C": 2, "D": 1, "MR": 0, "NR": 0
}

# ===========================
# TRADE HISTORY PROCESSING
# ===========================

def get_trade_history(hist_str):
    """Parse trade history from string format or return numpy array"""
    if isinstance(hist_str, np.ndarray):
        return hist_str
    if isinstance(hist_str, str):
        try:
            return np.array(eval(hist_str))
        except:
            return np.array([])
    return np.array([]) if hist_str is None else np.array(hist_str)

def compute_behavioral_similarity_fast(hist1, hist2):
    """
    Compute cosine similarity between two trade history arrays
    This is the core similarity metric for pair labeling
    
    Args:
        hist1: Trade history array for CUSIP 1 (n x 6 array)
        hist2: Trade history array for CUSIP 2 (n x 6 array)
    
    Returns:
        Cosine similarity score between -1 and 1
    """
    # Ensure we have enough history for meaningful comparison
    if len(hist1) < 5 or len(hist2) < 5:
        return 0.0
    
    # Flatten the trade history arrays
    hist1_flat = hist1.flatten()
    hist2_flat = hist2.flatten()
    
    # L2 normalize
    hist1_norm = hist1_flat / (np.linalg.norm(hist1_flat) + 1e-8)
    hist2_norm = hist2_flat / (np.linalg.norm(hist2_flat) + 1e-8)
    
    # Return dot product (cosine similarity)
    return np.dot(hist1_norm, hist2_norm)

# ===========================
# DATA TYPE CONVERSIONS
# ===========================

def to_numeric(series):
    """Convert a series to numeric, handling Decimal types"""
    if series.dtype == object:
        # Check if series contains Decimal objects
        sample = series.dropna().head()
        if len(sample) > 0 and any(isinstance(x, Decimal) for x in sample):
            return series.apply(lambda x: float(x) if isinstance(x, Decimal) else x).astype(float)
    return pd.to_numeric(series, errors='coerce')

# ===========================
# DATE CALCULATIONS
# ===========================

def diff_in_days_two_dates_360_30(end_date, start_date):
    """Calculate difference in days using 360/30 convention (MSRB Rule G-33)"""
    if pd.isna(end_date) or pd.isna(start_date):
        return np.nan
    
    Y2, Y1 = end_date.year, start_date.year
    M2, M1 = end_date.month, start_date.month
    D2, D1 = end_date.day, start_date.day
    
    # 30/360 day count convention adjustments
    D1 = min(D1, 30)
    if D1 == 30:
        D2 = min(D2, 30)
    
    return (Y2 - Y1) * 360 + (M2 - M1) * 30 + (D2 - D1)

def diff_in_days_two_dates_exact(end_date, start_date):
    """Calculate exact difference in days"""
    if pd.isna(end_date) or pd.isna(start_date):
        return np.nan
    return (end_date - start_date).days

def diff_in_days(trade, convention='360/30', calc_type=None):
    """Calculate days difference for a trade record"""
    if calc_type == 'accrual':
        if pd.isnull(trade.get('accrual_date')):
            start_date = trade.get('dated_date')
        else:
            start_date = trade.get('accrual_date')
    else:
        start_date = trade.get('dated_date')
    
    end_date = trade.get('settlement_date')
    
    if pd.isna(start_date) or pd.isna(end_date):
        return 0
    
    if convention == '360/30':
        return diff_in_days_two_dates_360_30(end_date, start_date)
    else:
        return diff_in_days_two_dates_exact(end_date, start_date)

def days_in_interest_payment(trade):
    """Calculate days in interest payment period"""
    if 'interest_payment_frequency' not in trade:
        return 180  # Default semi-annual
    
    freq_val = trade['interest_payment_frequency']
    
    # Handle if it's already converted to string
    if isinstance(freq_val, str):
        frequency = COUPON_FREQUENCY_TYPE.get(freq_val, 1e6)
    else:
        # Convert numeric code to string first
        freq_str = COUPON_FREQUENCY_DICT.get(freq_val, 'Unknown')
        frequency = COUPON_FREQUENCY_TYPE.get(freq_str, 1e6)
    
    if frequency == 0 or frequency >= 1e6:
        return 1e6
    
    return 360 / frequency

def calculate_a_over_e(row):
    """Calculate A/E ratio (Accrued/Expected interest)"""
    if not pd.isnull(row.get('previous_coupon_payment_date')) and not pd.isnull(row.get('settlement_date')):
        try:
            A = (row['settlement_date'] - row['previous_coupon_payment_date']).days
            days_ip = row.get('days_in_interest_payment', 180)
            if days_ip > 0:
                return A / days_ip
        except:
            pass
    
    accrued = row.get('accrued_days', 0)
    return accrued / NUM_OF_DAYS_IN_YEAR if NUM_OF_DAYS_IN_YEAR > 0 else 0

# ===========================
# DEFAULT VALUES
# ===========================

# Default values for missing features
FEATURES_AND_DEFAULT_VALUES = {
    'purpose_class': 0, 
    'call_timing': 0, 
    'call_timing_in_part': 0,
    'sink_frequency': 0, 
    'sink_amount_type': 10, 
    'issue_text': 'No issue text',
    'state_tax_status': 0, 
    'series_name': 'No series name', 
    'transaction_type': 'I',
    'next_call_price': 100, 
    'par_call_price': 100, 
    'min_amount_outstanding': 0,
    'max_amount_outstanding': 0, 
    'maturity_amount': 0,
    'issue_price': lambda df: to_numeric(df.issue_price).mean(skipna=True) if 'issue_price' in df.columns else 100,
    'orig_principal_amount': lambda df: np.log10(to_numeric(10 ** to_numeric(df.orig_principal_amount)).mean(skipna=True)) if 'orig_principal_amount' in df.columns else 0,
    'par_price': 100, 
    'called_redemption_type': 0, 
    'extraordinary_make_whole_call': False,
    'make_whole_call': False, 
    'default_indicator': False, 
    'days_to_settle': 0,
    'days_to_maturity': 0, 
    'days_to_refund': 0, 
    'call_to_maturity': 0,
    'days_in_interest_payment': 180
}

FEATURES_AND_DEFAULT_COLUMNS = {
    'days_to_par': 'days_to_maturity',
    'days_to_call': 'days_to_maturity'
}

def fill_missing_values(df):
    """Fill missing values with defaults"""
    df = df.copy()
    
    # Fill with default values
    for feature, default_value in FEATURES_AND_DEFAULT_VALUES.items():
        if feature in df.columns:
            if callable(default_value):
                try:
                    default_value = default_value(df)
                except Exception as e:
                    print(f"Warning: Could not compute default for {feature}: {e}")
                    default_value = 0 if feature in ['orig_principal_amount'] else 100
            df[feature] = df[feature].fillna(default_value)
    
    # Fill with other columns
    for feature, feature_to_replace_with in FEATURES_AND_DEFAULT_COLUMNS.items():
        if feature in df.columns and feature_to_replace_with in df.columns:
            df[feature] = df[feature].fillna(df[feature_to_replace_with])
    
    return df