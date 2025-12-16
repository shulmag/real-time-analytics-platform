import os
import numpy as np
import pandas as pd
from google.cloud import bigquery
from datetime import datetime, timezone
from data_fetchers import fetch_all_etf_data, fetch_sp_index_data, MUNI_ETFS
from data_processors import calculate_etf_returns, calculate_yield_changes, combine_data
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'
FINNHUB_API_KEY = 'c516kliad3if5950qksg'

# Set your BigQuery project and table details
PROJECT_ID = "eng-reactor-287421"
DATASET_ID = "yield_curves_v2"
TABLE_ID = "etf_selection"

# 1. Data Processing Functions
def prepare_data(etf_data, target_col, etf_cols):
    """
    Prepare ETF and target data for training
    """
    # Convert ETF returns to basis points for better scaling
    X = etf_data[etf_cols] * 10000  
    y = target_col
    return X, y

# 2. Main Lasso Training Function
def train_model(combined_data, target_col, etf_cols, window_size=45, 
                alphas=[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0, 1, 5, 10, 20]):
    """
    Train Lasso model with rolling window approach
    """
    results_dict = {}
    meta_dict = {}

    for alpha in alphas:
        i = window_size
        data_length = len(combined_data)
        
        while i + 1 <= data_length:
            # Get training and test windows
            X_train = combined_data[etf_cols].iloc[(i-window_size):i]
            X_test = combined_data[etf_cols].iloc[i:i+1]
            y_train = combined_data[target_col].iloc[(i-window_size):i]
            y_test = combined_data[target_col].iloc[i:i+1]

            # Scale data and train model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            lasso = Lasso(alpha=alpha, max_iter=5000)
            lasso.fit(X_train_scaled, y_train)

            # Store results
            date_start = X_train.index[0].strftime('%Y-%m-%d')
            results_dict[date_start] = {
                f'coef_{col}': coef for col, coef in zip(etf_cols, lasso.coef_)
            }
            results_dict[date_start]['prediction'] = lasso.predict(X_test_scaled)[0]
            results_dict[date_start]['actual'] = y_test.iloc[0]
            
            i += 1

        # Calculate metrics
        result_df = pd.DataFrame(results_dict).T
        result_df['error'] = abs(result_df['prediction'] - result_df['actual'])
        mae = result_df['error'].mean()
        
        meta_dict[str(alpha)] = {'result_df': result_df, 'mae': mae}
    
    return pd.DataFrame(meta_dict).T

# 3. Extract Best Parameters
def extract_info(lasso_df):
    """
    Get optimal alpha and ETF relevance scores
    """
    best_alpha = lasso_df['mae'].idxmin()
    relevance_df = (lasso_df.loc[best_alpha]['result_df']
                   .filter(regex='coef')
                   .ne(0)
                   .sum()
                   .sort_values(ascending=False))
    
    return relevance_df, float(best_alpha)

# 4. Find Best ETF Combination
def find_best_etf_combination(combined_data, target_col, relevance_df, best_alpha):
    """
    Starts with the most relevant ETF (based on the relevance ranking).
    Iteratively tests models with the top 1 ETF, top 2 ETFs, ... up to all ETFs in order of importance.
    For each subset, it trains the model again (using train_model) but only with best_alpha and the selected subset of ETFs.
    Records the resulting MAE for each subset.
    """
    importance_order = list(relevance_df.index.str.replace('coef_', ''))
    results = {}
    
    for i in range(1, len(importance_order) + 1):
        print(f"Testing top {i} ETFs...")
        current_etfs = importance_order[:i]
        
        # Train model with current ETF subset
        lasso_results = train_model(
            combined_data=combined_data,
            target_col=target_col,
            etf_cols=current_etfs,
            alphas=[best_alpha]
        )
        
        results[i] = {
            'etfs': current_etfs,
            'mae': lasso_results['mae'].iloc[0]
        }
    
    return pd.DataFrame(results).T

# 5. Driver Function
def analyze_index(combined_data, target_col, etf_cols):
    """
    Complete analysis pipeline for one index
    """
    print(f"\nAnalyzing {target_col}")
    
    # Train initial model
    lasso_df = train_model(combined_data, target_col, etf_cols)
    
    # Get best parameters
    relevance_df, best_alpha = extract_info(lasso_df)
    print(f"Best alpha: {best_alpha}")
    
    # Find optimal ETF combination
    results = find_best_etf_combination(
        combined_data, target_col, relevance_df, best_alpha
    )
    
    # Get best combination
    best_combo = results.loc[results['mae'].idxmin()]
    print(f"\nBest combination: {best_combo['etfs']}")
    print(f"MAE: {best_combo['mae']:.6f}")
    
    return results

def update_bq(final_selections):
    # Initialize the BigQuery client
    client = bigquery.Client()

    # Define your table reference
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    # Convert DataFrame rows into a list of dictionaries matching the BQ schema
    # The schema expected is:
    # lasso_selection_date (DATE), index_name (STRING), etfs_list (STRING REPEATED), mae (FLOAT)

    current_date = datetime.now(timezone.utc).date().isoformat()
    optimal_etfs = []
    for _, row in final_selections.iterrows():
        record = {
            "lasso_selection_date": current_date,
            "index_name": row["sp_index"],
            "etfs_list": row["optimal_etfs"],  # This should be a list of strings
            "mae": row["mae"]
        }
        optimal_etfs.append(record)

    # Insert data into BigQuery
    errors = client.insert_rows_json(table=table_ref, json_rows=optimal_etfs)

    if errors:
        print(f"Failed to insert rows: {errors}")
        return f"Failed to insert rows: {errors}"
    else:
        print(f"Successfully inserted {len(optimal_etfs)} rows into {table_ref}.")
        return "ETFs selected and inserted into BigQuery successfully."

def main(args):
    # 1. Data Collection
    # =================
    print("Fetching ETF data...")
    etf_data = fetch_all_etf_data(FINNHUB_API_KEY)

    print("\nFetching S&P index data...")
    sp_data = fetch_sp_index_data()

    # 2. Data Preparation
    # ==================
    print("\nCalculating returns and changes...")
    etf_returns = calculate_etf_returns(etf_data)
    yield_changes = calculate_yield_changes(sp_data)

    print("\nCombining datasets...")
    combined_data = combine_data(etf_returns, yield_changes)

    results = {}
    for target_col in yield_changes.columns:
        results[target_col] = analyze_index(combined_data, target_col, MUNI_ETFS) 

    # Create final DataFrame showing only the best combination for each index

    final_selections = pd.DataFrame([
    {
        'sp_index': index_name.replace('ytw_', ''),  # Remove ytw_ here
        'optimal_etfs': result.loc[result['mae'].idxmin(), 'etfs'],
        'mae': result['mae'].min()
    } 
    for index_name, result in results.items()
    ])

    # Sort by MAE to see best performing combinations first
    final_selections = final_selections.sort_values('mae')
    update_bq(final_selections)
    return 'SUCCESS'

# main(None)
