import pickle
import redis
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from tensorflow import keras

# Import custom layer registration
import siamese_network
# Import the feature engineering functions
from feature_engineering import engineer_features_complete

class RedisCusipEmbedder:
    """Embedder that fetches reference data from Redis instead of BigQuery"""
    
    def __init__(self, model_path, artifacts_path, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize the embedder with model and Redis connection"""
        # Load model
        self.model = keras.models.load_model(model_path, compile=False)
        
        # Load feature artifacts (contains scaler and encoders)
        with open(artifacts_path, 'rb') as f:
            self.artifacts = pickle.load(f)
            #print(f"Loaded artifacts with {len(self.artifacts.get('encoders', {}))} encoders")
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        
        # Reference data features - MUST match the order from update_new_pipeline_redis.py
        self.REFERENCE_DATA_FEATURES = [
            'coupon', 'cusip', 'ref_valid_from_date', 'ref_valid_to_date',
            'incorporated_state_code', 'organization_primary_name', 'instrument_primary_name',
            'issue_key', 'issue_text', 'conduit_obligor_name', 'is_called', 'is_callable',
            'is_escrowed_or_pre_refunded', 'first_call_date', 'call_date_notice',
            'callable_at_cav', 'par_price', 'call_defeased', 'call_timing',
            'call_timing_in_part', 'extraordinary_make_whole_call', 'extraordinary_redemption',
            'make_whole_call', 'next_call_date', 'next_call_price', 'call_redemption_id',
            'first_optional_redemption_code', 'second_optional_redemption_code',
            'third_optional_redemption_code', 'first_mandatory_redemption_code',
            'second_mandatory_redemption_code', 'third_mandatory_redemption_code',
            'par_call_date', 'par_call_price', 'maximum_call_notice_period',
            'called_redemption_type', 'muni_issue_type', 'refund_date', 'refund_price',
            'redemption_cav_flag', 'max_notification_days', 'min_notification_days',
            'next_put_date', 'put_end_date', 'put_feature_price', 'put_frequency',
            'put_start_date', 'put_type', 'maturity_date', 'sp_long', 'sp_stand_alone',
            'sp_icr_school', 'sp_prelim_long', 'sp_outlook_long', 'sp_watch_long',
            'sp_Short_Rating', 'sp_Credit_Watch_Short_Rating', 'sp_Recovery_Long_Rating',
            'moodys_long', 'moodys_short', 'moodys_Issue_Long_Rating',
            'moodys_Issue_Short_Rating', 'moodys_Credit_Watch_Long_Rating',
            'moodys_Credit_Watch_Short_Rating', 'moodys_Enhanced_Long_Rating',
            'moodys_Enhanced_Short_Rating', 'moodys_Credit_Watch_Long_Outlook_Rating',
            'has_sink_schedule', 'next_sink_date', 'sink_indicator', 'sink_amount_type_text',
            'sink_amount_type_type', 'sink_frequency', 'sink_defeased',
            'additional_next_sink_date', 'sink_amount_type', 'additional_sink_frequency',
            'min_amount_outstanding', 'max_amount_outstanding', 'default_exists',
            'has_unexpired_lines_of_credit', 'years_to_loc_expiration', 'escrow_exists',
            'escrow_obligation_percent', 'escrow_obligation_agent', 'escrow_obligation_type',
            'child_linkage_exists', 'put_exists', 'floating_rate_exists',
            'bond_insurance_exists', 'is_general_obligation', 'has_zero_coupons',
            'delivery_date', 'issue_price', 'primary_market_settlement_date', 'issue_date',
            'outstanding_indicator', 'federal_tax_status', 'maturity_amount',
            'available_denom', 'denom_increment_amount', 'min_denom_amount', 'accrual_date',
            'bond_insurance', 'coupon_type', 'current_coupon_rate', 'daycount_basis_type',
            'debt_type', 'default_indicator', 'first_coupon_date',
            'interest_payment_frequency', 'issue_amount', 'last_period_accrues_from_date',
            'next_coupon_payment_date', 'odd_first_coupon_date', 'orig_principal_amount',
            'original_yield', 'outstanding_amount', 'previous_coupon_payment_date',
            'sale_type', 'settlement_type', 'additional_project_txt', 'asset_claim_code',
            'additional_state_code', 'backed_underlying_security_id', 'bank_qualified',
            'capital_type', 'conditional_call_date', 'conditional_call_price',
            'designated_termination_date', 'DTCC_status', 'first_execution_date',
            'formal_award_date', 'maturity_description_code', 'muni_security_type',
            'mtg_insurance', 'orig_cusip_status', 'orig_instrument_enhancement_type',
            'other_enhancement_type', 'other_enhancement_company', 'pac_bond_indicator',
            'project_name', 'purpose_class', 'purpose_sub_class', 'refunding_issue_key',
            'refunding_dated_date', 'sale_date', 'sec_regulation', 'secured', 'series_name',
            'sink_fund_redemption_method', 'state_tax_status', 'tax_credit_frequency',
            'tax_credit_percent', 'use_of_proceeds', 'use_of_proceeds_supplementary',
            'series_id', 'security_description'
        ]
        
        # Create feature to index mapping
        self.REFERENCE_DATA_FEATURE_TO_INDEX = {
            feature: idx for idx, feature in enumerate(self.REFERENCE_DATA_FEATURES)
        }
        
        # Get model input shape
        self.expected_features = self.model.input_shape[1]
        print(f"Model expects {self.expected_features} features")
    
    def get_reference_data_from_redis(self, cusip, datetime_of_interest=None):
        """Fetch reference data from Redis for a single CUSIP"""
        # Get data from Redis
        reference_data_pickle = self.redis_client.get(cusip)
        # print(f"Reference data pickle: { pickle.loads(reference_data_pickle)}")
        if reference_data_pickle is None:
            raise ValueError(f"No reference data found for CUSIP {cusip} in Redis")
        
        # Deserialize the deque
        reference_data_deque = pickle.loads(reference_data_pickle)
        
        # Get the appropriate snapshot
        reference_data = self._get_point_in_time_reference_data(
            reference_data_deque, datetime_of_interest
        )
        
        return reference_data
    
    def _get_point_in_time_reference_data(self, reference_data_deque, datetime_of_interest=None):
        """Select the reference data snapshot that is current as of datetime_of_interest"""
        most_recent_snapshot = reference_data_deque[0].copy()  # Make a copy to avoid modifying original
        if datetime_of_interest is None:
            return most_recent_snapshot
        
        ref_valid_from_date_idx = self.REFERENCE_DATA_FEATURE_TO_INDEX['ref_valid_from_date']
        valid_from_date = most_recent_snapshot[ref_valid_from_date_idx]
        
        # Handle timezone if present
        if hasattr(valid_from_date, 'tz_localize'):
            valid_from_date = valid_from_date.tz_localize(None)
        elif hasattr(valid_from_date, 'replace') and valid_from_date.tzinfo:
            valid_from_date = valid_from_date.replace(tzinfo=None)
        
        if datetime_of_interest >= valid_from_date:
            return most_recent_snapshot
        
        # Walk back through history
        reference_data_index = 1
        differences_dicts = []
        
        while reference_data_index < len(reference_data_deque):
            current_item = reference_data_deque[reference_data_index]
            
            # If it's a dict, it's a differences dict
            if isinstance(current_item, dict):
                differences_dicts.append(current_item)
                # Check if this snapshot's from_date is before our target
                if ref_valid_from_date_idx in current_item:
                    snapshot_from_date = current_item[ref_valid_from_date_idx]
                    if hasattr(snapshot_from_date, 'tz_localize'):
                        snapshot_from_date = snapshot_from_date.tz_localize(None)
                    elif hasattr(snapshot_from_date, 'replace') and snapshot_from_date.tzinfo:
                        snapshot_from_date = snapshot_from_date.replace(tzinfo=None)
                    
                    if datetime_of_interest >= snapshot_from_date:
                        break
            
            reference_data_index += 1
        
        # Apply differences to most recent snapshot
        for differences in reversed(differences_dicts):  # Apply in reverse order
            for feature_idx, value in differences.items():
                most_recent_snapshot[feature_idx] = value
        
        return most_recent_snapshot
    
    def prepare_dataframe(self, reference_data, trade_type='P', quantity=100):
        """
        Convert Redis reference data to DataFrame matching expected format for feature engineering
        """
        # Use the features that match the actual data length
        if len(reference_data) > len(self.REFERENCE_DATA_FEATURES):
            print(f"WARNING: Redis data has {len(reference_data)} values but only {len(self.REFERENCE_DATA_FEATURES)} feature names defined")
            actual_features = self.REFERENCE_DATA_FEATURES + [f'unknown_{i}' for i in range(len(self.REFERENCE_DATA_FEATURES), len(reference_data))]
        else:
            actual_features = self.REFERENCE_DATA_FEATURES[:len(reference_data)]
        
        # Convert reference data array to DataFrame
        ref_df = pd.DataFrame([reference_data], columns=actual_features)
        
        # Convert date columns to datetime
        date_columns = [
            'maturity_date', 'next_call_date', 'refund_date', 'par_call_date',
            'delivery_date', 'accrual_date', 'previous_coupon_payment_date',
            'next_coupon_payment_date', 'first_coupon_date', 'last_period_accrues_from_date',
            'next_sink_date', 'next_put_date', 'put_end_date', 'put_start_date',
            'issue_date', 'primary_market_settlement_date', 'conditional_call_date',
            'first_execution_date', 'formal_award_date', 'refunding_dated_date',
            'sale_date', 'ref_valid_from_date', 'ref_valid_to_date', 'first_call_date'
        ]
        
        for col in date_columns:
            if col in ref_df.columns:
                ref_df[col] = pd.to_datetime(ref_df[col], errors='coerce')
        
        # Create output dataframe matching expected structure
        # Initialize with a dictionary to ensure we have a single row
        df_dict = {}
        
        # Add trade-specific fields
        df_dict['cusip'] = ref_df['cusip'].iloc[0] if 'cusip' in ref_df.columns else '646039YM3'
        df_dict['par_traded'] = float(quantity) * 1000  # Convert to dollar amount
        df_dict['trade_date'] = pd.Timestamp.now().date()
        df_dict['trade_datetime'] = pd.Timestamp.now()
        df_dict['settlement_date'] = (pd.Timestamp.now() + pd.Timedelta(days=2)).date()
        df_dict['trade_type'] = trade_type
        df_dict['transaction_type'] = 'I'  # Default from SQL
        
        # Map reference fields - note that Redis already has 'coupon' directly
        df_dict['coupon'] = ref_df['coupon'].iloc[0] if 'coupon' in ref_df.columns else 0.0
        
        # Map sp_long to rating
        df_dict['rating'] = ref_df['sp_long'].iloc[0] if 'sp_long' in ref_df.columns else 'NR'
        
        # Copy other fields directly from reference data
        fields_to_copy = [
            'delivery_date', 'coupon_type', 'maturity_date', 'issue_price', 'issue_amount',
            'maturity_amount', 'orig_principal_amount', 'original_yield',
            'is_callable', 'is_called', 'callable_at_cav', 'next_call_date',
            'next_call_price', 'par_call_date', 'par_call_price',
            'called_redemption_type', 'call_timing', 'call_timing_in_part',
            'incorporated_state_code', 'state_tax_status', 'federal_tax_status',
            'purpose_class', 'purpose_sub_class', 'use_of_proceeds',
            'muni_security_type', 'muni_issue_type', 'capital_type',
            'series_name', 'min_amount_outstanding', 'max_amount_outstanding',
            'next_sink_date', 'has_sink_schedule', 'sink_frequency',
            'sink_amount_type', 'other_enhancement_type',
            'has_unexpired_lines_of_credit', 'years_to_loc_expiration',
            'is_escrowed_or_pre_refunded', 'default_exists', 'make_whole_call',
            'extraordinary_make_whole_call', 'is_general_obligation',
            'refund_date', 'interest_payment_frequency', 'accrual_date',
            'previous_coupon_payment_date', 'next_coupon_payment_date',
            'first_coupon_date', 'last_period_accrues_from_date', 'par_price',
            'default_indicator', 'escrow_exists'
        ]
        
        for field in fields_to_copy:
            if field in ref_df.columns:
                df_dict[field] = ref_df[field].iloc[0]
        
        # Add special handling for boolean fields that might be missing
        df_dict['is_non_transaction_based_compensation'] = False
        
        # Add dated_date as alias for accrual_date
        df_dict['dated_date'] = df_dict.get('accrual_date', pd.NaT)
        
        # Create DataFrame from dictionary - this ensures we have exactly 1 row
        df = pd.DataFrame([df_dict])
        
        # Ensure all date columns have proper type
        for col in df.columns:
            if col in date_columns or col.endswith('_date'):
                if col in df.columns and pd.api.types.is_object_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def embed(self, cusip, trade_type='P', quantity=100, datetime_of_interest=None):
        """Generate embedding for a CUSIP using reference data from Redis"""
        # Get reference data from Redis
        reference_data = self.get_reference_data_from_redis(cusip, datetime_of_interest)
        
        # Convert to DataFrame format expected by feature engineering
        df = self.prepare_dataframe(reference_data, trade_type, quantity)
        
        # Use the existing feature engineering pipeline
        features, _, feature_names = engineer_features_complete(
            df, 
            fit=False,  # We're using pre-fitted artifacts
            artifacts=self.artifacts
        )
        
        # Generate embedding
        embedding = self.model.predict(features, verbose=0)[0]
        
        # Prepare metadata
        meta = {
            'cusip': cusip,
            'trade_type': trade_type,
            'quantity': quantity,
            'datetime': datetime_of_interest.isoformat() if datetime_of_interest else 'current',
            'embedding_shape': embedding.shape,
            'source': 'redis',
            'features_used': features.shape[1],
            'feature_names_count': len(feature_names)
        }
        
        return embedding, meta