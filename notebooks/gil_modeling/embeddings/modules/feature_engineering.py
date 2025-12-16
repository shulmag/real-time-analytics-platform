"""
Feature Engineering for Municipal Bond Embeddings
Handles all feature transformations and engineering for Siamese network training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
import warnings
from typing import Tuple, Dict, List, Optional
from .bond_embedding_helpers import (
    to_numeric, diff_in_days, days_in_interest_payment,
    calculate_a_over_e, fill_missing_values,
    COUPON_FREQUENCY_DICT, COUPON_FREQUENCY_TYPE,
    RATING_SCORE_MAP
)

def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature processing function - transforms raw data into ML-ready features
    
    Args:
        df: Raw dataframe with bond trades
    
    Returns:
        Processed dataframe with engineered features
    """
    df = df.copy()
    
    # Convert Decimal columns to float
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head()
            if len(sample) > 0 and any(isinstance(x, type(x)) for x in sample if str(type(x)) == "<class 'decimal.Decimal'>"):
                df[col] = df[col].apply(lambda x: float(x) if str(type(x)) == "<class 'decimal.Decimal'>" else x)
    
    # Process interest payment frequency
    if 'interest_payment_frequency' in df.columns:
        df['interest_payment_frequency'] = df['interest_payment_frequency'].fillna(0)
        # Handle both numeric and string inputs
        if df['interest_payment_frequency'].dtype in [np.int64, np.float64]:
            df['interest_payment_frequency'] = df['interest_payment_frequency'].apply(
                lambda x: COUPON_FREQUENCY_DICT.get(int(x), 'Unknown') if pd.notna(x) else 'Unknown'
            )
    
    # Process quantity (log transform)
    if 'par_traded' in df.columns:
        df['par_traded'] = to_numeric(df['par_traded'])
        df['quantity'] = np.log10(df['par_traded'].clip(lower=1))
    
    # Process amounts with log transformation
    amount_cols = ['issue_amount', 'maturity_amount', 'orig_principal_amount', 'max_amount_outstanding']
    for col in amount_cols:
        if col in df.columns:
            df[col] = to_numeric(df[col])
            df[col] = np.log10(1 + df[col].fillna(0).clip(lower=0))
    
    # Process coupon
    if 'coupon' in df.columns:
        df['coupon'] = to_numeric(df['coupon']).fillna(0)
    
    # Create binary features
    binary_features = {
        'callable': ('is_callable', bool),
        'called': ('is_called', bool),
        'zerocoupon': (None, lambda df: df.get('coupon', pd.Series([1])) == 0),
        'whenissued': (None, lambda df: df.get('delivery_date', pd.Series([pd.NaT])) >= df.get('trade_date', pd.Series([pd.NaT]))),
        'sinking': (None, lambda df: ~df.get('next_sink_date', pd.Series([None])).isnull()),
        'deferred': (None, lambda df: (df.get('interest_payment_frequency', pd.Series([''])) == 'Unknown') | 
                                     (df.get('zerocoupon', pd.Series([False])) == 1))
    }
    
    for new_col, (source_col, transform) in binary_features.items():
        if source_col and source_col in df.columns:
            df[new_col] = df[source_col].astype(bool).astype(int)
        elif transform is not None:
            try:
                df[new_col] = transform(df).astype(int)
            except:
                df[new_col] = 0
    
    # Process dates - convert to days from trade date
    if 'settlement_date' in df.columns and 'trade_date' in df.columns:
        df['days_to_settle'] = (df['settlement_date'] - df['trade_date']).dt.days.fillna(0)
        # Remove trades settling >= 30 days from trade date (likely errors)
        df = df[df['days_to_settle'] < 30]
    
    # Calculate days to various dates with log transformation
    date_calculations = [
        ('days_to_maturity', 'maturity_date', 'trade_date'),
        ('days_to_call', 'next_call_date', 'trade_date'),
        ('days_to_refund', 'refund_date', 'trade_date'),
        ('days_to_par', 'par_call_date', 'trade_date'),
        ('call_to_maturity', 'maturity_date', 'next_call_date')
    ]
    
    for new_col, end_col, start_col in date_calculations:
        if end_col in df.columns and start_col in df.columns:
            days_diff = (df[end_col] - df[start_col]).dt.days.fillna(0).clip(lower=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                df[new_col] = np.log10(1 + days_diff)
                df[new_col] = df[new_col].replace([-np.inf, np.inf], 0)
    
    # Calculate accrual features
    if 'settlement_date' in df.columns:
        df['accrued_days'] = df.apply(lambda row: diff_in_days(row, calc_type='accrual'), axis=1)
        df['accrued_days'] = df['accrued_days'].fillna(0)
    
    if 'interest_payment_frequency' in df.columns:
        df['days_in_interest_payment'] = df.apply(days_in_interest_payment, axis=1)
    else:
        df['days_in_interest_payment'] = 180  # Default semi-annual
    
    if 'accrued_days' in df.columns and 'days_in_interest_payment' in df.columns:
        df['scaled_accrued_days'] = df['accrued_days'] / (360 / df['days_in_interest_payment'].clip(lower=1))
        df['scaled_accrued_days'] = df['scaled_accrued_days'].fillna(0)
        df['A/E'] = df.apply(calculate_a_over_e, axis=1)
        df['A/E'] = df['A/E'].fillna(0)
    
    # Process rating
    if 'sp_long' in df.columns:
        df['sp_long'] = df['sp_long'].fillna('MR')
        df['rating'] = df['sp_long']
    elif 'rating' not in df.columns:
        df['rating'] = 'MR'
    
    # Fill missing values
    df = fill_missing_values(df)
    
    return df

def engineer_features_complete(df_raw: pd.DataFrame, 
                              fit: bool = True, 
                              artifacts: Optional[Dict] = None) -> Tuple[np.ndarray, Dict, List[str]]:
    """
    Complete feature engineering pipeline with scaling and encoding
    
    Args:
        df_raw: Raw dataframe with bond data
        fit: Whether to fit new encoders/scalers (True for training, False for inference)
        artifacts: Dictionary containing fitted encoders and scalers
    
    Returns:
        Tuple of (feature matrix, artifacts dictionary, feature names list)
    """
    # First apply the basic processing
    df = process_features(df_raw)
    
    if artifacts is None:
        artifacts = {}
    
    # Define feature groups
    binary_features = [
        'callable', 'called', 'zerocoupon', 'whenissued', 'sinking', 'deferred',
        'is_non_transaction_based_compensation', 'is_general_obligation',
        'callable_at_cav', 'extraordinary_make_whole_call', 'make_whole_call',
        'has_unexpired_lines_of_credit', 'escrow_exists', 'default_indicator'
    ]
    
    numeric_direct = [
        'coupon', 'issue_price', 'par_price', 'original_yield',
        'next_call_price', 'par_call_price', 'days_to_settle',
        'days_to_maturity', 'days_to_call', 'days_to_refund',
        'days_to_par', 'call_to_maturity', 'accrued_days',
        'days_in_interest_payment', 'scaled_accrued_days', 'A/E'
    ]
    
    numeric_log = [
        'quantity', 'issue_amount', 'maturity_amount',
        'orig_principal_amount', 'max_amount_outstanding'
    ]
    
    categorical_features = [
        'incorporated_state_code', 'trade_type', 'purpose_class',
        'rating', 'purpose_sub_class', 'called_redemption_type',
        'call_timing', 'call_timing_in_part', 'sink_frequency',
        'sink_amount_type', 'state_tax_status', 'transaction_type',
        'coupon_type', 'federal_tax_status', 'use_of_proceeds',
        'muni_security_type', 'muni_issue_type', 'capital_type',
        'other_enhancement_type', 'series_name'
    ]
    
    feature_list = []
    feature_names = []
    
    # Process binary features
    for feat in binary_features:
        if feat in df.columns:
            val = pd.to_numeric(df[feat], errors='coerce').fillna(0).astype(float).values.reshape(-1, 1)
            feature_list.append(val)
            feature_names.append(feat)
    
    # Process numeric direct features
    for feat in numeric_direct:
        if feat in df.columns:
            val = to_numeric(df[feat]).fillna(0).astype(float).values.reshape(-1, 1)
            feature_list.append(val)
            feature_names.append(feat)
    
    # Process numeric log features
    for feat in numeric_log:
        if feat in df.columns:
            val = to_numeric(df[feat]).fillna(0).astype(float).values.reshape(-1, 1)
            feature_list.append(val)
            feature_names.append(feat)
    
    # Add rating score
    if 'rating' in df.columns:
        rating_score = df['rating'].map(RATING_SCORE_MAP).fillna(0).values.reshape(-1, 1)
        feature_list.append(rating_score)
        feature_names.append('rating_score')
    
    # Combine numeric features
    if feature_list:
        X_numeric = np.hstack(feature_list)
    else:
        X_numeric = np.zeros((len(df), 0))
    
    # Scale numeric features
    if fit:
        scaler = RobustScaler()
        X_numeric_scaled = scaler.fit_transform(X_numeric)
        artifacts['scaler'] = scaler
    else:
        scaler = artifacts.get('scaler')
        if scaler:
            X_numeric_scaled = scaler.transform(X_numeric)
        else:
            X_numeric_scaled = X_numeric
    
    # One-hot encode categorical features
    cat_encoded_list = []
    cat_feature_names = []
    
    if fit:
        artifacts['encoders'] = {}
    
    for cat in categorical_features:
        if cat in df.columns:
            # Handle different data types robustly
            dtype_name = str(df[cat].dtype)
            
            # For nullable integer types (Int64, Int32, etc.)
            if 'Int' in dtype_name:
                # Convert to regular int first, filling NaN with -9999
                cat_values = df[cat].fillna(-9999).astype('int64').astype(str)
                cat_values = cat_values.replace('-9999', 'MISSING')
            # For float types
            elif 'float' in dtype_name.lower():
                cat_values = df[cat].fillna(-9999.0).astype('int64').astype(str)
                cat_values = cat_values.replace('-9999', 'MISSING')
            # For object/string types
            else:
                cat_values = df[cat].fillna('MISSING').astype(str)
            
            if fit:
                encoder = LabelEncoder()
                unique_vals = cat_values.unique().tolist()
                if 'UNKNOWN' not in unique_vals:
                    unique_vals.append('UNKNOWN')
                encoder.fit(unique_vals)
                artifacts['encoders'][cat] = encoder
            else:
                encoder = artifacts.get('encoders', {}).get(cat)
                if encoder is None:
                    continue
            
            # Handle unseen categories
            cat_values = cat_values.apply(lambda x: x if x in encoder.classes_ else 'UNKNOWN')
            encoded = encoder.transform(cat_values)
            
            # One-hot encode
            n_classes = len(encoder.classes_)
            one_hot = np.zeros((len(df), n_classes))
            one_hot[np.arange(len(df)), encoded] = 1
            
            cat_encoded_list.append(one_hot)
            
            # Add feature names
            for i, class_name in enumerate(encoder.classes_):
                cat_feature_names.append(f"{cat}_{class_name}")
    
    # Combine all features
    if cat_encoded_list:
        X_cat = np.hstack(cat_encoded_list)
        X = np.hstack([X_numeric_scaled, X_cat])
        all_feature_names = feature_names + cat_feature_names
    else:
        X = X_numeric_scaled
        all_feature_names = feature_names
    
    print(f"Engineered {len(all_feature_names)} features from {len(df)} trades")
    
    return X.astype(np.float32), artifacts, all_feature_names