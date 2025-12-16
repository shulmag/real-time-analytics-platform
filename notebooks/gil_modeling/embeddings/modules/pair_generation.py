"""
Pair Generation Module for Municipal Bond Embeddings
Generates training pairs with proper temporal alignment
"""

import sys

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import combinations
import random
import time
from typing import List, Tuple, Dict, Optional
from .bond_embedding_helpers import get_trade_history, compute_behavioral_similarity_fast

def process_single_window(args: Tuple) -> Tuple[List, List]:
    """
    Process a single time window to generate pairs
    
    Args:
        args: Tuple containing (window, window_df, time_window_seconds, min_similarity, max_pairs_per_window)
    
    Returns:
        Tuple of (positive_pairs, negative_pairs) with temporal metadata
    """
    window, window_df, time_window_seconds, min_similarity, max_pairs_per_window = args
    
    # Group by CUSIP and keep the LAST trade in the window for each CUSIP
    # This ensures we use the most recent state of each bond
    cusip_groups = window_df.sort_values('trade_datetime').groupby('cusip').last()
    
    if len(cusip_groups) < 2:
        return [], []
    
    # Prepare CUSIP data with histories AND keep track of the trade datetime
    cusip_data = {}
    for cusip, row in cusip_groups.iterrows():
        hist = get_trade_history(row['trade_history'])
        if len(hist) >= 5:  # Need sufficient history for meaningful similarity
            cusip_data[cusip] = {
                'history': hist,
                'trade_datetime': row['trade_datetime'],
                'rtrs_control_number': row.get('rtrs_control_number', None)
            }
    
    if len(cusip_data) < 2:
        return [], []
    
    # Generate all pairs within this window
    window_pairs = []
    for cusip1, cusip2 in combinations(cusip_data.keys(), 2):
        hist1 = cusip_data[cusip1]['history']
        hist2 = cusip_data[cusip2]['history']
        
        # Compute behavioral similarity using dot product
        sim = compute_behavioral_similarity_fast(hist1, hist2)
        
        # Store pair with temporal metadata
        window_pairs.append({
            'cusip1': cusip1,
            'cusip2': cusip2,
            'similarity': sim,
            'window_datetime': window,  # Time window for this pair
            'cusip1_datetime': cusip_data[cusip1]['trade_datetime'],
            'cusip2_datetime': cusip_data[cusip2]['trade_datetime'],
            'cusip1_rtrs': cusip_data[cusip1]['rtrs_control_number'],
            'cusip2_rtrs': cusip_data[cusip2]['rtrs_control_number']
        })
    
    # Cap pairs per window if needed
    if len(window_pairs) > max_pairs_per_window:
        window_pairs = random.sample(window_pairs, max_pairs_per_window)
    
    # Classify into positive and negative based on similarity
    positive_pairs = []
    negative_pairs = []
    
    for pair in window_pairs:
        if pair['similarity'] > min_similarity:
            pair['label'] = 1
            positive_pairs.append(pair)
        elif pair['similarity'] < 0.2:  # Clear negative pairs
            pair['label'] = 0
            negative_pairs.append(pair)
    
    return positive_pairs, negative_pairs

def generate_temporal_behavioral_pairs_parallel(
    df: pd.DataFrame,
    time_window_seconds: int = 30,
    min_similarity: float = 0.5,
    max_pairs_per_window: int = 1000,
    n_processes: Optional[int] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate pairs using parallel processing with temporal alignment
    
    Args:
        df: DataFrame with trade data including trade_history
        time_window_seconds: Size of time window for co-occurrence
        min_similarity: Minimum similarity for positive pairs
        max_pairs_per_window: Maximum pairs to generate per time window
        n_processes: Number of parallel processes
    
    Returns:
        Tuple of (positive_pairs, negative_pairs) with full metadata
    """
    if n_processes is None:
        n_processes = int(cpu_count() / 2)
    
    print(f"Using {n_processes} processes for pair generation")
    
    # Ensure datetime column
    df['trade_dt'] = pd.to_datetime(df['trade_datetime'])
    
    # Create time windows
    df['time_window'] = df['trade_dt'].dt.floor(f'{time_window_seconds}s')
    
    # Get unique windows
    unique_windows = df['time_window'].unique()
    print(f"Processing {len(unique_windows)} time windows")
    
    # Prepare arguments for parallel processing
    window_args = []
    for window in unique_windows:
        window_df = df[df['time_window'] == window]
        window_args.append((
            window, 
            window_df, 
            time_window_seconds, 
            min_similarity, 
            max_pairs_per_window
        ))
    
    # Process windows in parallel
    start_time = time.time()
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_window, window_args)
    
    # Combine results
    all_positive_pairs = []
    all_negative_pairs = []
    seen_pairs = set()
    
    for pos_pairs, neg_pairs in results:
        # Add positive pairs
        for pair in pos_pairs:
            # Create unique key for deduplication
            pair_key = tuple(sorted([pair['cusip1'], pair['cusip2']]))
            
            # Keep the pair from the earliest time window if duplicates exist
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                all_positive_pairs.append(pair)
        
        # Add negative pairs
        for pair in neg_pairs:
            pair_key = tuple(sorted([pair['cusip1'], pair['cusip2']]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                all_negative_pairs.append(pair)
    
    elapsed_time = time.time() - start_time
    print(f"Pair generation completed in {elapsed_time:.2f} seconds")
    
    return all_positive_pairs, all_negative_pairs

def create_training_dataset(
    positive_pairs: List[Dict], 
    negative_pairs: List[Dict], 
    neg_to_pos_ratio: int = 3
) -> pd.DataFrame:
    """
    Create balanced training dataset from positive and negative pairs
    
    Args:
        positive_pairs: List of positive pair dictionaries
        negative_pairs: List of negative pair dictionaries
        neg_to_pos_ratio: Ratio of negative to positive pairs
    
    Returns:
        DataFrame with balanced pairs and metadata
    """
    n_positives = len(positive_pairs)
    n_negatives_needed = min(len(negative_pairs), n_positives * neg_to_pos_ratio)
    
    # Sample negative pairs if we have too many
    if len(negative_pairs) > n_negatives_needed:
        negative_pairs = random.sample(negative_pairs, n_negatives_needed)
    
    # Combine all pairs
    all_pairs = positive_pairs + negative_pairs
    
    # Shuffle
    random.shuffle(all_pairs)
    
    # Create DataFrame
    pairs_df = pd.DataFrame(all_pairs)
    
    return pairs_df

def run_pair_generation_pipeline(
    df: pd.DataFrame,
    time_window_seconds: int = 30,
    min_similarity: float = 0.5,
    max_pairs_per_window: int = 500,
    output_path: str = 'embedding_pairs_with_metadata.pkl',
    n_processes: Optional[int] = None
) -> pd.DataFrame:
    """
    Main pipeline for pair generation with full temporal metadata
    
    Args:
        df: DataFrame with trade data
        min_similarity: Minimum similarity for positive pairs
        max_pairs_per_window: Max pairs per window
        output_path: Path to save pairs
        n_processes: Number of parallel processes
    
    Returns:
        DataFrame with training pairs
    """
    print(f"Data shape: {df.shape}")
    print(f"Unique CUSIPs: {df['cusip'].nunique()}")
    print(f"Date range: {df['trade_datetime'].min()} to {df['trade_datetime'].max()}")
    print(f"Time window: {time_window_seconds} seconds")
    
    # Generate pairs in parallel
    print("\nGenerating pairs with parallel processing...")
    positive_pairs, negative_pairs = generate_temporal_behavioral_pairs_parallel(
        df,
        time_window_seconds=time_window_seconds,
        min_similarity=min_similarity,
        max_pairs_per_window=max_pairs_per_window,
        n_processes=n_processes
    )
    
    print(f"\nGenerated:")
    print(f"  Positive pairs: {len(positive_pairs):,}")
    print(f"  Negative pairs: {len(negative_pairs):,}")
    
    # Create balanced training dataset
    pairs_df = create_training_dataset(positive_pairs, negative_pairs)
    
    print(f"\nFinal dataset shape: {pairs_df.shape}")
    print(f"Positive ratio: {pairs_df['label'].mean():.2%}")
    
    # Save with metadata
    pairs_df.to_pickle(output_path)
    print(f"\nSaved to {output_path}")
    
    # Print statistics
    if len(pairs_df) > 0:
        print("\nSimilarity statistics:")
        pos_df = pairs_df[pairs_df['label'] == 1]
        neg_df = pairs_df[pairs_df['label'] == 0]
        
        if len(pos_df) > 0:
            print(f"Positive pairs - Mean similarity: {pos_df['similarity'].mean():.3f}")
            print(f"Positive pairs - Std:  {pos_df['similarity'].std():.3f}")
        
        if len(neg_df) > 0:
            print(f"Negative pairs - Mean similarity: {neg_df['similarity'].mean():.3f}")
            print(f"Negative pairs - Std:  {neg_df['similarity'].std():.3f}")
        
        # Show temporal spread
        print("\nTemporal statistics:")
        print(f"Earliest pair: {pairs_df['window_datetime'].min()}")
        print(f"Latest pair: {pairs_df['window_datetime'].max()}")
        print(f"Number of unique time windows: {pairs_df['window_datetime'].nunique()}")
    
    return pairs_df