#!/usr/bin/env python3
"""
CUSIP Siamese Network Training Script
Usage: python train_cusip_embeddings.py --pairs pairs.pkl --trades trades.csv [options]
"""

import os
import sys
import argparse
import pickle
import warnings
import logging
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Configure logging
def setup_logging(log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    return logging.getLogger(__name__)

# ===========================
# CONSTANTS AND UTILITIES
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
    'Changeable': 44, 'Daily': 360, 'Interest at maturity': 0, 'Not Applicable': 1e6
}

# Helper to convert Decimal to float
def to_numeric(series):
    """Convert a series to numeric, handling Decimal types"""
    if series.dtype == object:
        if any(isinstance(x, Decimal) for x in series.dropna().head()):
            return series.apply(lambda x: float(x) if isinstance(x, Decimal) else x).astype(float)
    return pd.to_numeric(series, errors='coerce')

# Default values for missing features
FEATURES_AND_DEFAULT_VALUES = {
    'purpose_class': 0, 'call_timing': 0, 'call_timing_in_part': 0,
    'sink_frequency': 0, 'sink_amount_type': 10, 'issue_text': 'No issue text',
    'state_tax_status': 0, 'series_name': 'No series name', 'transaction_type': 'I',
    'next_call_price': 100, 'par_call_price': 100, 'min_amount_outstanding': 0,
    'max_amount_outstanding': 0, 'maturity_amount': 0,
    'issue_price': lambda df: to_numeric(df.issue_price).mean(skipna=True) if 'issue_price' in df.columns else 100,
    'orig_principal_amount': lambda df: np.log10(to_numeric(10 ** to_numeric(df.orig_principal_amount)).mean(skipna=True)) if 'orig_principal_amount' in df.columns else 0,
    'par_price': 100, 'called_redemption_type': 0, 'extraordinary_make_whole_call': False,
    'make_whole_call': False, 'default_indicator': False, 'days_to_settle': 0,
    'days_to_maturity': 0, 'days_to_refund': 0, 'call_to_maturity': 0,
    'days_in_interest_payment': 180
}

FEATURES_AND_DEFAULT_COLUMNS = {
    'days_to_par': 'days_to_maturity',
    'days_to_call': 'days_to_maturity'
}

# ===========================
# HELPER FUNCTIONS
# ===========================

def diff_in_days_two_dates_360_30(end_date, start_date):
    """Calculate difference in days using 360/30 convention (MSRB Rule G-33)"""
    if pd.isna(end_date) or pd.isna(start_date):
        return np.nan
    
    Y2, Y1 = end_date.year, start_date.year
    M2, M1 = end_date.month, start_date.month
    D2, D1 = end_date.day, start_date.day
    
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
    """Calculate days difference for a trade"""
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
        return 180
    
    freq_val = trade['interest_payment_frequency']
    
    if isinstance(freq_val, str):
        frequency = COUPON_FREQUENCY_TYPE.get(freq_val, 1e6)
    else:
        freq_str = COUPON_FREQUENCY_DICT.get(freq_val, 'Unknown')
        frequency = COUPON_FREQUENCY_TYPE.get(freq_str, 1e6)
    
    if frequency == 0 or frequency >= 1e6:
        return 1e6
    
    return 360 / frequency

def calculate_a_over_e(row):
    """Calculate A/E ratio"""
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

def fill_missing_values(df):
    """Fill missing values with defaults"""
    df = df.copy()
    
    for feature, default_value in FEATURES_AND_DEFAULT_VALUES.items():
        if feature in df.columns:
            if callable(default_value):
                try:
                    default_value = default_value(df)
                except Exception as e:
                    logging.warning(f"Could not compute default for {feature}: {e}")
                    default_value = 0 if feature in ['orig_principal_amount'] else 100
            df[feature] = df[feature].fillna(default_value)
    
    for feature, feature_to_replace_with in FEATURES_AND_DEFAULT_COLUMNS.items():
        if feature in df.columns and feature_to_replace_with in df.columns:
            df[feature] = df[feature].fillna(df[feature_to_replace_with])
    
    return df

# ===========================
# FEATURE ENGINEERING
# ===========================

def process_features(df):
    """Main feature processing function"""
    df = df.copy()
    
    # Convert Decimal columns to float
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head()
            if len(sample) > 0 and any(isinstance(x, Decimal) for x in sample):
                df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
    
    # Process interest payment frequency
    if 'interest_payment_frequency' in df.columns:
        df['interest_payment_frequency'] = df['interest_payment_frequency'].fillna(0)
        df['interest_payment_frequency'] = df['interest_payment_frequency'].apply(
            lambda x: COUPON_FREQUENCY_DICT.get(int(x), 'Unknown') if isinstance(x, (int, float)) else x
        )
    
    # Process quantity
    if 'par_traded' in df.columns:
        df['par_traded'] = to_numeric(df['par_traded'])
        df['quantity'] = np.log10(df['par_traded'].clip(lower=1))
    
    # Process amounts with log transformation
    for col in ['issue_amount', 'maturity_amount', 'orig_principal_amount', 'max_amount_outstanding']:
        if col in df.columns:
            df[col] = to_numeric(df[col])
            df[col] = np.log10(1 + df[col].fillna(0).clip(lower=0))
    
    # Process coupon
    if 'coupon' in df.columns:
        df['coupon'] = to_numeric(df['coupon']).fillna(0)
    
    # Create binary features
    if 'is_callable' in df.columns:
        df['callable'] = df['is_callable'].astype(bool).astype(int)
    if 'is_called' in df.columns:
        df['called'] = df['is_called'].astype(bool).astype(int)
    if 'coupon' in df.columns:
        df['zerocoupon'] = (df['coupon'] == 0).astype(int)
    if 'delivery_date' in df.columns and 'trade_date' in df.columns:
        df['whenissued'] = (df['delivery_date'] >= df['trade_date']).astype(int)
    if 'next_sink_date' in df.columns:
        df['sinking'] = (~df['next_sink_date'].isnull()).astype(int)
    if 'interest_payment_frequency' in df.columns and 'zerocoupon' in df.columns:
        df['deferred'] = ((df['interest_payment_frequency'] == 'Unknown') | (df['zerocoupon'] == 1)).astype(int)
    
    # Process dates
    if 'settlement_date' in df.columns and 'trade_date' in df.columns:
        df['days_to_settle'] = (df['settlement_date'] - df['trade_date']).dt.days.fillna(0)
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
        df['days_in_interest_payment'] = 180
    
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
    
    df = fill_missing_values(df)
    
    return df

def engineer_features_complete(df_raw, fit=True, artifacts=None):
    """Complete feature engineering"""
    df = process_features(df_raw)
    
    if artifacts is None:
        artifacts = {}
    
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
    
    # Binary features
    for feat in binary_features:
        if feat in df.columns:
            val = pd.to_numeric(df[feat], errors='coerce').fillna(0).astype(float).values.reshape(-1, 1)
            feature_list.append(val)
            feature_names.append(feat)
    
    # Numeric direct features
    for feat in numeric_direct:
        if feat in df.columns:
            val = to_numeric(df[feat]).fillna(0).astype(float).values.reshape(-1, 1)
            feature_list.append(val)
            feature_names.append(feat)
    
    # Numeric log features
    for feat in numeric_log:
        if feat in df.columns:
            val = to_numeric(df[feat]).fillna(0).astype(float).values.reshape(-1, 1)
            feature_list.append(val)
            feature_names.append(feat)
    
    # Add rating score
    rating_map = {
        "AAA": 22, "AA+": 21, "AA": 20, "AA-": 19,
        "A+": 18, "A": 17, "A-": 16,
        "BBB+": 15, "BBB": 14, "BBB-": 13,
        "BB+": 12, "BB": 11, "BB-": 10,
        "B+": 9, "B": 8, "B-": 7,
        "CCC+": 6, "CCC": 5, "CCC-": 4,
        "CC": 3, "C": 2, "D": 1, "MR": 0, "NR": 0
    }
    
    if 'rating' in df.columns:
        rating_score = df['rating'].map(rating_map).fillna(0).values.reshape(-1, 1)
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
            dtype_name = str(df[cat].dtype)
            
            if 'Int' in dtype_name or 'int' in dtype_name.lower():
                cat_series = df[cat].copy()
                cat_series = cat_series.fillna(-9999)
                cat_series = cat_series.astype('int64')
                cat_values = cat_series.astype(str)
                cat_values = cat_values.replace('-9999', 'MISSING')
            elif 'float' in dtype_name.lower():
                cat_series = df[cat].copy()
                cat_series = cat_series.fillna(-9999.0)
                cat_series = cat_series.astype('int64')
                cat_values = cat_series.astype(str)
                cat_values = cat_values.replace('-9999', 'MISSING')
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
            
            cat_values = cat_values.apply(lambda x: x if x in encoder.classes_ else 'UNKNOWN')
            encoded = encoder.transform(cat_values)
            
            n_classes = len(encoder.classes_)
            one_hot = np.zeros((len(df), n_classes))
            one_hot[np.arange(len(df)), encoded] = 1
            
            cat_encoded_list.append(one_hot)
            
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
    
    logging.info(f"Engineered {len(all_feature_names)} features")
    return X.astype(np.float32), artifacts, all_feature_names

# ===========================
# SIAMESE NETWORK
# ===========================

def create_base_network(input_dim, embedding_dim=128, dropout_rate=0.1):
    """Create the base network for one side of the Siamese network"""
    inputs = Input(shape=(input_dim,))
    
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    embeddings = layers.Dense(embedding_dim, activation='linear', name='embeddings')(x)
    embeddings = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embeddings)
    
    model = Model(inputs=inputs, outputs=embeddings, name='base_network')
    return model

def create_siamese_network(input_dim, embedding_dim=128, dropout_rate=0.1):
    """Create the full Siamese network"""
    base_network = create_base_network(input_dim, embedding_dim, dropout_rate)
    
    input_a = Input(shape=(input_dim,), name='input_a')
    input_b = Input(shape=(input_dim,), name='input_b')
    
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)
    
    cosine_similarity = layers.Dot(axes=1, normalize=False)([embedding_a, embedding_b])
    
    siamese_model = Model(inputs=[input_a, input_b], outputs=cosine_similarity)
    
    return siamese_model, base_network

def contrastive_loss(y_true, y_pred, margin=0.5):
    """Contrastive loss for Siamese network"""
    y_pred_dist = 1 - y_pred
    pos_loss = y_true * tf.square(y_pred_dist)
    neg_loss = (1 - y_true) * tf.square(tf.maximum(0.0, margin - y_pred_dist))
    return tf.reduce_mean(pos_loss + neg_loss)

def prepare_pairs_for_training(features_df, pairs_df, artifacts=None, num_months=None):
    """Prepare pairs of CUSIPs for training
    
    Args:
        features_df: DataFrame with trade features
        pairs_df: DataFrame with CUSIP pairs
        artifacts: Feature engineering artifacts
        num_months: Number of months to use (from most recent backwards)
    """
    logging.info("Engineering features...")
    
    # Ensure date columns are datetime
    date_columns = [
        'refund_date', 'accrual_date', 'dated_date', 'next_sink_date',
        'delivery_date', 'trade_date', 'trade_datetime', 'par_call_date',
        'maturity_date', 'settlement_date', 'next_call_date',
        'previous_coupon_payment_date', 'next_coupon_payment_date',
        'first_coupon_date', 'last_period_accrues_from_date'
    ]
    
    for col in date_columns:
        if col in features_df.columns:
            features_df[col] = pd.to_datetime(features_df[col], errors='coerce')
    
    # Filter to specified number of months if requested
    if num_months is not None and 'trade_date' in features_df.columns:
        logging.info(f"Filtering to most recent {num_months} months of data...")
        
        # Find the date range
        max_date = features_df['trade_date'].max()
        min_date = max_date - pd.DateOffset(months=num_months)
        
        # Log the date range
        logging.info(f"Date range: {min_date.date()} to {max_date.date()}")
        
        # Filter the dataframe
        original_len = len(features_df)
        features_df = features_df[features_df['trade_date'] >= min_date].copy()
        filtered_len = len(features_df)
        
        logging.info(f"Filtered from {original_len:,} to {filtered_len:,} trades")
    
    X_all, artifacts, feature_names = engineer_features_complete(
        features_df, 
        fit=(artifacts is None), 
        artifacts=artifacts
    )
    
    logging.info(f"Feature dimension: {X_all.shape[1]}")
    logging.info(f"Total CUSIPs processed: {X_all.shape[0]}")
    
    cusip_to_vec = dict(zip(features_df["cusip"].values, X_all))
    
    features_a = []
    features_b = []
    labels = []
    missing_cusips = set()
    
    for _, row in pairs_df.iterrows():
        cusip1, cusip2 = row["cusip1"], row["cusip2"]
        
        if cusip1 in cusip_to_vec and cusip2 in cusip_to_vec:
            features_a.append(cusip_to_vec[cusip1])
            features_b.append(cusip_to_vec[cusip2])
            labels.append(row["label"])
        else:
            if cusip1 not in cusip_to_vec:
                missing_cusips.add(cusip1)
            if cusip2 not in cusip_to_vec:
                missing_cusips.add(cusip2)
    
    if missing_cusips:
        logging.warning(f"{len(missing_cusips)} CUSIPs from pairs not found in features")
    
    X_pairs = (np.array(features_a, dtype=np.float32), 
               np.array(features_b, dtype=np.float32))
    y = np.array(labels, dtype=np.float32)
    
    logging.info(f"Valid pairs prepared: {len(y)}")
    
    return X_pairs, y, artifacts, feature_names

def train_siamese_network(X_train, y_train, X_val, y_val, 
                         input_dim, embedding_dim=128, dropout_rate=0.1,
                         epochs=100, batch_size=256, learning_rate=0.001,
                         output_dir='.'):
    """Train the Siamese network"""
    siamese_model, base_network = create_siamese_network(input_dim, embedding_dim, dropout_rate)
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    siamese_model.compile(
        optimizer=optimizer,
        loss=contrastive_loss,
        metrics=['mae']
    )
    
    logging.info("\nBase network summary:")
    base_network.summary(print_fn=logging.info)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            os.path.join(output_dir, 'best_siamese_model.h5'), 
            save_best_only=True, 
            monitor='val_loss', 
            verbose=1
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    logging.info("\nStarting training...")
    history = siamese_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return siamese_model, base_network, history

def run_training_pipeline(features_df, pairs_df, args):
    """Run the complete training pipeline"""
    logging.info("="*60)
    logging.info("Starting CUSIP Siamese Network Training")
    logging.info("="*60)
    
    logging.info("Preparing features and pairs...")
    X_pairs, y, artifacts, feature_names = prepare_pairs_for_training(
        features_df, pairs_df, num_months=args.num_months
    )
    
    logging.info("\nSplitting data...")
    X_train_a, X_val_a, X_train_b, X_val_b, y_train, y_val = train_test_split(
        X_pairs[0], X_pairs[1], y, 
        test_size=args.test_size, 
        random_state=args.random_state, 
        stratify=y
    )
    X_train = (X_train_a, X_train_b)
    X_val = (X_val_a, X_val_b)
    
    input_dim = X_train[0].shape[1]
    logging.info(f"\nData summary:")
    logging.info(f"  Input dimension: {input_dim}")
    logging.info(f"  Number of features: {len(feature_names)}")
    logging.info(f"  Training samples: {len(y_train):,}")
    logging.info(f"  Validation samples: {len(y_val):,}")
    logging.info(f"  Positive ratio in train: {y_train.mean():.2%}")
    logging.info(f"  Positive ratio in val: {y_val.mean():.2%}")
    
    siamese_model, base_network, history = train_siamese_network(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        embedding_dim=args.embedding_dim,
        dropout_rate=args.dropout_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    logging.info("\nSaving model and artifacts...")
    base_network.save(os.path.join(args.output_dir, 'cusip_embedding_model.h5'))
    
    with open(os.path.join(args.output_dir, 'feature_artifacts.pkl'), 'wb') as f:
        pickle.dump(artifacts, f)
    
    with open(os.path.join(args.output_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(args.output_dir, 'training_history.csv'), index=False)
    
    logging.info("Training complete!")
    logging.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    return base_network, artifacts, history

def main():
    parser = argparse.ArgumentParser(description='Train CUSIP Siamese Network Embeddings')
    
    # Required arguments
    parser.add_argument('--pairs', type=str, required=True, help='Path to pairs pickle file')
    parser.add_argument('--trades', type=str, required=True, help='Path to trades pickle file')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for models and artifacts')
    parser.add_argument('--num-months', type=int, default=None, help='Number of months to use (from most recent backwards)')
    parser.add_argument('--embedding-dim', type=int, default=128, help='Dimension of embeddings')
    parser.add_argument('--dropout-rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test-size', type=float, default=0.2, help='Validation set size')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--log-file', type=str, help='Path to log file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data
        logging.info(f"Loading pairs from: {args.pairs}")
        pairs_df = pd.read_pickle(args.pairs)
        logging.info(f"Loaded {len(pairs_df):,} pairs")
        
        logging.info(f"Loading trades from: {args.trades}")
        # Check file extension to determine how to load
        if args.trades.endswith('.pkl') or args.trades.endswith('.pickle'):
            features_df = pd.read_pickle(args.trades)
        elif args.trades.endswith('.csv'):
            features_df = pd.read_csv(args.trades)
        else:
            # Default to pickle for your specific file
            features_df = pd.read_pickle(args.trades)
        logging.info(f"Loaded {len(features_df):,} trades")
        
        # Log training configuration
        if args.num_months:
            logging.info(f"Will use most recent {args.num_months} months of data for training")
        
        # Run training
        base_network, artifacts, history = run_training_pipeline(
            features_df, 
            pairs_df, 
            args
        )
        
        logging.info("="*60)
        logging.info("Training completed successfully!")
        logging.info(f"Models saved to: {args.output_dir}")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()