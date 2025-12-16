#!/usr/bin/env python3
"""
Bond2Vec Pair Generation Script


Optimized for high-performance computing (176 CPUs, 1400GB RAM)

Usage:
    python generate_bond_pairs.py --input data.pkl --output pairs.pkl
"""

import pandas as pd
import numpy as np
from itertools import combinations
import random
import pickle
from multiprocessing import Pool, cpu_count
import time
import argparse
import os
import sys
from pathlib import Path

# ============================================
# CONFIGURATION CONSTANTS
# ============================================
MIN_TRADE_HISTORY_LENGTH = 5  # Minimum number of historical trades required
POSITIVE_SIM_THRESHOLD = 0.5   # Similarity threshold for positive pairs
NEGATIVE_SIM_THRESHOLD = 0.2   # Similarity threshold for negative pairs
EPSILON = 1e-8                 # Small value to prevent division by zero

# High-performance settings for your machine
DEFAULT_TIME_WINDOW = 30       # seconds
DEFAULT_MAX_PAIRS_PER_WINDOW = 2000  # Increased for more data
DEFAULT_NEG_TO_POS_RATIO = 1   # 1:1 ratio for balanced dataset
DEFAULT_N_PROCESSES = 170      # Leave 6 CPUs for system overhead

# ============================================
# CORE FUNCTIONS
# ============================================

def compute_behavioral_similarity_fast(hist1, hist2):
    """
    Compute cosine similarity between two trade histories.
    
    Trade history shape: (n_trades, 6) where features are:
    [yield_spread, treasury_spread, log_par_traded, trade_type_1, trade_type_2, log_seconds_ago]
    """
    if len(hist1) < MIN_TRADE_HISTORY_LENGTH or len(hist2) < MIN_TRADE_HISTORY_LENGTH:
        return 0.0
    
    # Flatten histories to 1D vectors
    hist1_flat = hist1.flatten()
    hist2_flat = hist2.flatten()
    
    # L2 normalize with epsilon to prevent division by zero
    hist1_norm = hist1_flat / (np.linalg.norm(hist1_flat) + EPSILON)
    hist2_norm = hist2_flat / (np.linalg.norm(hist2_flat) + EPSILON)
    
    # Cosine similarity via dot product of normalized vectors
    return np.dot(hist1_norm, hist2_norm)

def get_trade_history(hist_str):
    """Parse trade history, handling various input formats"""
    if isinstance(hist_str, np.ndarray):
        return hist_str
    return np.array(hist_str) if hist_str is not None else np.array([])

def process_single_window(args):
    """
    Process bonds that traded within a single time window.
    Returns positive and negative pairs based on behavioral similarity.
    """
    window, window_df, time_window_seconds, min_similarity, max_pairs_per_window = args
    
    # Get unique CUSIPs in this time window
    cusip_groups = window_df.groupby('cusip').first()
    
    if len(cusip_groups) < 2:
        return [], []
    
    # Extract valid trade histories
    cusip_data = {}
    for cusip, row in cusip_groups.iterrows():
        hist = get_trade_history(row['trade_history'])
        if len(hist) >= MIN_TRADE_HISTORY_LENGTH:
            cusip_data[cusip] = hist
    
    if len(cusip_data) < 2:
        return [], []
    
    # Generate all possible pairs within this window
    window_pairs = []
    for cusip1, cusip2 in combinations(cusip_data.keys(), 2):
        hist1 = cusip_data[cusip1]
        hist2 = cusip_data[cusip2]
        sim = compute_behavioral_similarity_fast(hist1, hist2)
        window_pairs.append((cusip1, cusip2, sim))
    
    # Random sampling if too many pairs (for computational efficiency)
    if len(window_pairs) > max_pairs_per_window:
        window_pairs = random.sample(window_pairs, max_pairs_per_window)
    
    # Classify pairs based on similarity
    positive_pairs = []
    negative_pairs = []
    for cusip1, cusip2, sim in window_pairs:
        if sim > min_similarity:
            positive_pairs.append((cusip1, cusip2, sim))
        elif sim < NEGATIVE_SIM_THRESHOLD:
            negative_pairs.append((cusip1, cusip2, sim))
    
    return positive_pairs, negative_pairs

def generate_temporal_behavioral_pairs_parallel(
    df,
    time_window_seconds=DEFAULT_TIME_WINDOW,
    min_similarity=POSITIVE_SIM_THRESHOLD,
    max_pairs_per_window=DEFAULT_MAX_PAIRS_PER_WINDOW,
    n_processes=DEFAULT_N_PROCESSES
):
    """
    Main parallel processing function for pair generation.
    """
    print(f"[CONFIG] Using {n_processes} processes for pair generation")
    print(f"[CONFIG] Time window: {time_window_seconds} seconds")
    print(f"[CONFIG] Max pairs per window: {max_pairs_per_window}")
    
    # Ensure datetime column
    df['trade_dt'] = pd.to_datetime(df['trade_datetime'])
    
    # Create time windows
    df['time_window'] = df['trade_dt'].dt.floor(f'{time_window_seconds}s')
    
    # Get unique windows
    unique_windows = df['time_window'].unique()
    print(f"[INFO] Processing {len(unique_windows):,} time windows")
    
    # Prepare arguments for parallel processing
    window_args = [
        (window, df[df['time_window'] == window], time_window_seconds, 
         min_similarity, max_pairs_per_window)
        for window in unique_windows
    ]
    
    # Process windows in parallel with progress tracking
    print(f"[PROCESSING] Starting parallel pair generation...")
    start_time = time.time()
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_window, window_args, chunksize=10)
    
    # Combine results and deduplicate
    all_positive_pairs = []
    all_negative_pairs = []
    seen_pairs = set()
    
    for pos_pairs, neg_pairs in results:
        for cusip1, cusip2, sim in pos_pairs:
            pair_key = tuple(sorted([cusip1, cusip2]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                all_positive_pairs.append((cusip1, cusip2, sim))
        
        for cusip1, cusip2, sim in neg_pairs:
            pair_key = tuple(sorted([cusip1, cusip2]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                all_negative_pairs.append((cusip1, cusip2, sim))
    
    elapsed_time = time.time() - start_time
    print(f"[COMPLETED] Pair generation completed in {elapsed_time:.2f} seconds")
    print(f"[STATS] Generated {len(all_positive_pairs):,} positive pairs")
    print(f"[STATS] Generated {len(all_negative_pairs):,} negative pairs")
    
    return all_positive_pairs, all_negative_pairs

def create_training_dataset(positive_pairs, negative_pairs, neg_to_pos_ratio=DEFAULT_NEG_TO_POS_RATIO):
    """Create balanced training dataset"""
    n_positives = len(positive_pairs)
    n_negatives_needed = min(len(negative_pairs), int(n_positives * neg_to_pos_ratio))
    
    # Sample negative pairs if we have too many
    if len(negative_pairs) > n_negatives_needed:
        negative_pairs = random.sample(negative_pairs, n_negatives_needed)
        print(f"[BALANCE] Sampled {n_negatives_needed:,} negative pairs to match ratio {neg_to_pos_ratio}:1")
    
    # Create DataFrame
    all_pairs = []
    
    for cusip1, cusip2, sim in positive_pairs:
        all_pairs.append({
            'cusip1': cusip1,
            'cusip2': cusip2,
            'similarity': sim,
            'label': 1
        })
    
    for cusip1, cusip2, sim in negative_pairs:
        all_pairs.append({
            'cusip1': cusip1,
            'cusip2': cusip2,
            'similarity': sim,
            'label': 0
        })
    
    random.shuffle(all_pairs)
    return pd.DataFrame(all_pairs)

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Generate bond pairs for Bond2Vec training')
    parser.add_argument('--input', type=str, required=True, help='Input pickle file path')
    parser.add_argument('--output', type=str, default='bond_pairs.pkl', 
                       help='Output pickle file path (default: bond_pairs.pkl)')
    parser.add_argument('--days', type=int, default=None,
                       help='Process only last N days of data (default: use all data)')  # ADD THIS LINE
    parser.add_argument('--time-window', type=int, default=DEFAULT_TIME_WINDOW,
                       help=f'Time window in seconds (default: {DEFAULT_TIME_WINDOW})')
    parser.add_argument('--min-similarity', type=float, default=POSITIVE_SIM_THRESHOLD,
                       help=f'Minimum similarity for positive pairs (default: {POSITIVE_SIM_THRESHOLD})')
    parser.add_argument('--max-pairs-per-window', type=int, default=DEFAULT_MAX_PAIRS_PER_WINDOW,
                       help=f'Max pairs per time window (default: {DEFAULT_MAX_PAIRS_PER_WINDOW})')
    parser.add_argument('--neg-to-pos-ratio', type=float, default=DEFAULT_NEG_TO_POS_RATIO,
                       help=f'Negative to positive ratio (default: {DEFAULT_NEG_TO_POS_RATIO})')
    parser.add_argument('--n-processes', type=int, default=DEFAULT_N_PROCESSES,
                       help=f'Number of processes (default: {DEFAULT_N_PROCESSES})')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)
    
    print("="*60)
    print("BOND2VEC PAIR GENERATION")
    print("="*60)
    print(f"[SYSTEM] CPUs available: {cpu_count()}")
    print(f"[SYSTEM] Using {args.n_processes} processes")
    print(f"[INPUT] Loading data from {args.input}")
    
    # Load data
    start_load = time.time()
    import sys
    # Add compatibility for older pandas pickle files
    sys.modules['pandas.core.indexes.numeric'] = sys.modules.get('pandas.core.indexes.numeric', sys.modules['pandas.core.indexes.api'])

    with open(args.input, 'rb') as f:
        df = pickle.load(f)
    print(f"[LOADED] Data loaded in {time.time() - start_load:.2f} seconds")
    print(f"[DATA] Shape: {df.shape}")
    print(f"[DATA] Unique CUSIPs: {df['cusip'].nunique():,}")
    print(f"[DATA] Date range: {df['trade_datetime'].min()} to {df['trade_datetime'].max()}")
    
    # Filter to last 3 months
    df['trade_datetime'] = pd.to_datetime(df['trade_datetime'])
    three_months_ago = df['trade_datetime'].max() - pd.Timedelta(days=90)
    df_filtered = df[df['trade_datetime'] >= three_months_ago].copy()
    
    print(f"\n[FILTER] Processing last 3 months only")
    print(f"[FILTER] Date range: {df_filtered['trade_datetime'].min()} to {df_filtered['trade_datetime'].max()}")
    print(f"[FILTER] Filtered shape: {df_filtered.shape}")
    print(f"[FILTER] Filtered unique CUSIPs: {df_filtered['cusip'].nunique():,}")
    
    # Generate pairs
    positive_pairs, negative_pairs = generate_temporal_behavioral_pairs_parallel(
        df_filtered,  # Use filtered dataframe
        time_window_seconds=args.time_window,
        min_similarity=args.min_similarity,
        max_pairs_per_window=args.max_pairs_per_window,
        n_processes=args.n_processes
    )
    
    # Create balanced dataset
    print("\n[DATASET] Creating balanced training dataset...")
    pairs_df = create_training_dataset(
        positive_pairs, 
        negative_pairs, 
        neg_to_pos_ratio=args.neg_to_pos_ratio
    )
    
    # Save results
    print(f"\n[SAVE] Saving to {args.output}")
    pairs_df.to_pickle(args.output)
    
    # Final statistics
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"[FINAL] Total pairs: {len(pairs_df):,}")
    print(f"[FINAL] Positive ratio: {pairs_df['label'].mean():.2%}")
    print(f"[FINAL] Output file: {args.output}")
    print(f"[FINAL] File size: {Path(args.output).stat().st_size / 1e9:.2f} GB")
    
    # Detailed statistics
    pos_df = pairs_df[pairs_df['label']==1]
    neg_df = pairs_df[pairs_df['label']==0]
    
    print("\n[SIMILARITY STATS]")
    if len(pos_df) > 0:
        print(f"  Positive pairs - Mean: {pos_df['similarity'].mean():.3f} ± {pos_df['similarity'].std():.3f}")
        print(f"  Positive pairs - Range: [{pos_df['similarity'].min():.3f}, {pos_df['similarity'].max():.3f}]")
    
    if len(neg_df) > 0:
        print(f"  Negative pairs - Mean: {neg_df['similarity'].mean():.3f} ± {neg_df['similarity'].std():.3f}")
        print(f"  Negative pairs - Range: [{neg_df['similarity'].min():.3f}, {neg_df['similarity'].max():.3f}]")
    
    print("\n[SUCCESS] Pair generation completed successfully!")

if __name__ == "__main__":
    main()