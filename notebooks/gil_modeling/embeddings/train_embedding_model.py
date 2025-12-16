#!/usr/bin/env python3
"""
Standalone Training Script for CUSIP Embedding Model
Loads data from GCS, generates pairs, and trains Siamese network
"""

import os
import sys
import argparse
import pickle
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from google.cloud import storage

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/gil/git/creds.json"

# Add modules directory to Python path BEFORE importing
# This allows the modules to find each other
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.join(script_dir, 'modules')
sys.path.insert(0, modules_dir)

# Import modules
from modules.pair_generation import run_pair_generation_pipeline
from modules.siamese_network import run_training_pipeline, get_embeddings


def load_data_from_gcs(bucket_name: str, blob_name: str) -> pd.DataFrame:
    """Load pickle file from Google Cloud Storage"""
    print(f"\n{'='*60}")
    print("LOADING DATA FROM GCS")
    print(f"{'='*60}")
    print(f"Bucket: {bucket_name}")
    print(f"Blob: {blob_name}")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Download to temporary file
    temp_file = "/tmp/training_data.pkl"
    blob.download_to_filename(temp_file)
    
    # Load pickle
    df = pd.read_pickle(temp_file)
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
    print(f"Unique CUSIPs: {df['cusip'].nunique():,}")
    
    # Clean up temp file
    os.remove(temp_file)
    
    return df


def filter_recent_data(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Filter data to last N days"""
    print(f"\n{'='*60}")
    print(f"FILTERING TO LAST {days} DAYS")
    print(f"{'='*60}")
    
    # Convert trade_date to datetime if needed
    if df['trade_date'].dtype == 'object':
        df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    # Calculate cutoff date
    max_date = df['trade_date'].max()
    cutoff_date = max_date - timedelta(days=days)
    
    print(f"Max date in data: {max_date}")
    print(f"Cutoff date: {cutoff_date}")
    
    # Filter
    df_filtered = df[df['trade_date'] >= cutoff_date].copy()
    
    print(f"Filtered data shape: {df_filtered.shape}")
    print(f"Unique CUSIPs after filter: {df_filtered['cusip'].nunique():,}")
    print(f"Date range after filter: {df_filtered['trade_date'].min()} to {df_filtered['trade_date'].max()}")
    
    return df_filtered


def main():
    parser = argparse.ArgumentParser(description='Train CUSIP Embedding Model')
    
    # Data parameters
    parser.add_argument('--training_window_days', type=int, default=31,
                        help='Number of days of historical data to use for training (default: 31)')
    parser.add_argument('--gcs_uri', type=str,
                        default='gs://automated_training/processed_data/processed_data_yield_spread_with_similar_trades_v2.pkl',
                        help='GCS URI for input data')
    
    # Pair generation parameters
    parser.add_argument('--time_window_seconds', type=int, default=30,
                        help='Time window for co-occurrence in pair generation (default: 30)')
    parser.add_argument('--min_similarity', type=float, default=0.8,
                        help='Minimum similarity threshold for positive pairs (default: 0.8)')
    parser.add_argument('--max_pairs_per_window', type=int, default=500,
                        help='Maximum pairs per time window (default: 500)')
    parser.add_argument('--n_processes', type=int, default=None,
                        help='Number of parallel processes for pair generation (default: CPU count)')
    
    # Training parameters
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str,
                        default='/home/gil/git/ficc/notebooks/gil_modeling/embeddings',
                        help='Output directory for model and artifacts')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse GCS URI
    gcs_uri = args.gcs_uri.replace('gs://', '')
    bucket_name = gcs_uri.split('/')[0]
    blob_name = '/'.join(gcs_uri.split('/')[1:])
    
    print(f"\n{'='*60}")
    print("CUSIP EMBEDDING MODEL TRAINING")
    print(f"{'='*60}")
    print(f"Start time: {datetime.now()}")
    print(f"\nParameters:")
    print(f"  Training window: {args.training_window_days} days")
    print(f"  Time window for pairs: {args.time_window_seconds} seconds")
    print(f"  Min similarity: {args.min_similarity}")
    print(f"  Embedding dimension: {args.embedding_dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output directory: {args.output_dir}")
    
    try:
        # Step 1: Load data from GCS
        df = load_data_from_gcs(bucket_name, blob_name)
        
        # Step 2: Filter to recent data
        df_filtered = filter_recent_data(df, args.training_window_days)
        
        # Save filtered training data
        max_date = df_filtered['trade_date'].max()
        training_data_filename = f"training_data_{args.training_window_days}days_ending_{max_date.strftime('%Y%m%d')}.pkl"
        training_data_path = os.path.join(args.output_dir, training_data_filename)
        df_filtered.to_pickle(training_data_path)
        print(f"\nTraining data saved to: {training_data_path}")
        print(f"Filename: {training_data_filename}")
        
        # Step 3: Generate pairs
        print(f"\n{'='*60}")
        print("GENERATING TRAINING PAIRS")
        print(f"{'='*60}")
        
        pairs_output_path = os.path.join(args.output_dir, 'training_pairs_temporal.pkl')
        pairs_df = run_pair_generation_pipeline(
            df_filtered,
            time_window_seconds=args.time_window_seconds,
            min_similarity=args.min_similarity,
            max_pairs_per_window=args.max_pairs_per_window,
            output_path=pairs_output_path,
            n_processes=args.n_processes
        )
        
        print(f"\nPairs saved to: {pairs_output_path}")
        
        # Step 4: Train model
        print(f"\n{'='*60}")
        print("TRAINING SIAMESE NETWORK")
        print(f"{'='*60}")
        
        base_network, artifacts, history = run_training_pipeline(
            features_df=df_filtered,
            pairs_df=pairs_df,
            test_size=args.test_size,
            embedding_dim=args.embedding_dim,
            epochs=args.epochs
        )
        
        # Model and artifacts are already saved by run_training_pipeline
        # Move them to output directory if different
        default_model_path = "cusip_embedding_model_temporal.keras"
        default_artifacts_path = "feature_artifacts_temporal.pkl"
        default_feature_names_path = "feature_names_temporal.pkl"
        
        final_model_path = os.path.join(args.output_dir, "cusip_embedding_model_temporal.keras")
        final_artifacts_path = os.path.join(args.output_dir, "feature_artifacts_temporal.pkl")
        final_feature_names_path = os.path.join(args.output_dir, "feature_names_temporal.pkl")
        
        # Move files if output directory is different from current directory
        if os.path.abspath(args.output_dir) != os.path.abspath('.'):
            import shutil
            if os.path.exists(default_model_path):
                shutil.move(default_model_path, final_model_path)
            if os.path.exists(default_artifacts_path):
                shutil.move(default_artifacts_path, final_artifacts_path)
            if os.path.exists(default_feature_names_path):
                shutil.move(default_feature_names_path, final_feature_names_path)
        else:
            final_model_path = default_model_path
            final_artifacts_path = default_artifacts_path
            final_feature_names_path = default_feature_names_path
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Training data saved to: {training_data_path}")
        print(f"Model saved to: {final_model_path}")
        print(f"Artifacts saved to: {final_artifacts_path}")
        print(f"Feature names saved to: {final_feature_names_path}")
        print(f"Pairs saved to: {pairs_output_path}")
        print(f"\nFinal validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"End time: {datetime.now()}")
        
        # Print usage example
        print(f"\n{'='*60}")
        print("USAGE EXAMPLE")
        print(f"{'='*60}")
        print("To generate embeddings for new data:")
        print(f"""
from tensorflow import keras
import pickle
from modules.siamese_network import get_embeddings

# Load model and artifacts
base_network = keras.models.load_model('{final_model_path}', compile=False)
with open('{final_artifacts_path}', 'rb') as f:
    artifacts = pickle.load(f)

# Generate embeddings
embeddings_df = get_embeddings(your_cusip_df, base_network, artifacts)
        """)
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR OCCURRED")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())