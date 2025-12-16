import pandas as pd
import random
import matplotlib.pyplot as plt
import json
import requests
from google.cloud import bigquery
from collections import defaultdict
import io
import time
from datetime import datetime

EMAIL = 'gil@ficc.ai'
PASSWORD = 'aDwYwwL6'  # Note: In production, use environment variables for credentials
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'

query = """
SELECT msrb.*, ref.sp_long 
FROM `eng-reactor-287421.auxiliary_views_v2.msrb_final` msrb 
LEFT JOIN (
  SELECT cusip, sp_long,
         ROW_NUMBER() OVER (PARTITION BY cusip ORDER BY ref_valid_from_date DESC) as rn
  FROM `reference_data_v2.reference_data_flat`
) ref ON msrb.cusip = ref.cusip AND ref.rn = 1
WHERE DATE(trade_datetime) = "2025-05-22"
"""

def sql_to_df(query):
    start_time = time.time()
    client = bigquery.Client()
    df = client.query(query).to_dataframe()
    end_time = time.time()
    print(f"SQL query execution time: {end_time - start_time:.2f} seconds")
    return df

def ficc_prices(batch):
    url = 'https://api.ficc.ai/api/batchpricing'
    resp = requests.post(url, data={
        'username': EMAIL,
        'password': PASSWORD,
        'cusipList': [x['cusip'] for x in batch],
        'quantityList': [x['quantity'] for x in batch],
        'tradeTypeList': [x['side'] for x in batch]
    })
    resp.raise_for_status()
    return resp.json()

def ficc_historical_predictions(trades):
    start_time = time.time()
    url = 'https://api.ficc.ai/api/compliance'
    
    cusip_list = [trade['cusip'] for trade in trades]
    quantity_list = [int(float(trade['par_traded']) / 1000) for trade in trades]  # Divide by 1000 and convert to int
    trade_datetime_list = [trade['trade_datetime'] for trade in trades]
    user_price_list = [trade['dollar_price'] for trade in trades]  # Use actual trade price
    
    results = []
    
    trade_types = ['S', 'P']
    
    for trade_type in trade_types:
        trade_type_list = [trade_type] * len(trades)
        
        print(f"Making API call for {len(trades)} trades (trade_type: {trade_type})...")
        api_start_time = time.time()
        
        resp = requests.post(url, data={
            'username': EMAIL,
            'password': PASSWORD,
            'cusipList': cusip_list,
            'quantityList': quantity_list,
            'tradeTypeList': trade_type_list,
            'tradeDatetimeList': trade_datetime_list,
            'userPriceList': user_price_list
        })
        
        api_end_time = time.time()
        print(f"API call for trade_type {trade_type} took: {api_end_time - api_start_time:.2f} seconds")
                
        resp.raise_for_status()
        result = json.loads(resp.text)
        result2 = json.loads(result)
        df = pd.DataFrame(result2)
        df['api_trade_type'] = trade_type
        results.append(df)
    
    combined_df = pd.concat(results, ignore_index=True)
    end_time = time.time()
    print(f"Total ficc_historical_predictions execution time: {end_time - start_time:.2f} seconds")
    return combined_df

def main():
    # Start overall timer
    overall_start_time = time.time()
    
    try:
        data_start_time = time.time()
        df = pd.read_pickle('msrb_0610.pkl')[:6] # gs: limit to 500 trades to test

        data_end_time = time.time()
        print(f"Loaded {len(df)} trades from existing df.pkl in {data_end_time - data_start_time:.2f} seconds")
    except FileNotFoundError:
        print("Getting yesterday's MSRB trades with ratings from SQL...")
        df = sql_to_df(query)
        print(f"Found {len(df)} trades from yesterday")
        pickle_start_time = time.time()
        df.to_pickle('df.pkl')
        pickle_end_time = time.time()
        print(f"Saved df.pkl in {pickle_end_time - pickle_start_time:.2f} seconds")
        
    # Time the data preparation
    prep_start_time = time.time()
    trades = []
    for _, row in df.iterrows():
        trade = {
            'cusip': row['cusip'],
            'par_traded': str(row['par_traded']),
            'trade_datetime': row['trade_datetime'].strftime('%Y-%m-%dT%H:%M:%S') if pd.notna(row['trade_datetime']) else '',
            'dollar_price': str(row['dollar_price']),
            'trade_type': row['trade_type'],
            'maturity_date': row['maturity_date'].strftime('%Y-%m-%d') if pd.notna(row['maturity_date']) else '',
            'coupon': str(row['coupon']) if pd.notna(row['coupon']) else '',
            #'sp_long': row['sp_long'] if pd.notna(row['sp_long']) else ''
        }
        trades.append(trade)
    prep_end_time = time.time()
    print(f"Data preparation took: {prep_end_time - prep_start_time:.2f} seconds")
    
    print(f"Getting predictions for {len(trades)} trades...")
    predictions_df = ficc_historical_predictions(trades)   
    
    # Time the results processing
    processing_start_time = time.time()
    results = []
    for trade in trades:
        cusip = trade['cusip']
        
        # Get predictions for this trade
        trade_preds = predictions_df[predictions_df['cusip'] == cusip]
        
        bid_price = None
        offer_price = None
        
        bid_pred = trade_preds[trade_preds['api_trade_type'] == 'P']
        offer_pred = trade_preds[trade_preds['api_trade_type'] == 'S']
        
        if len(bid_pred) > 0 and bid_pred['price'].iloc[0] != -1:
            bid_price = bid_pred['price'].iloc[0]
        if len(offer_pred) > 0 and offer_pred['price'].iloc[0] != -1:
            offer_price = offer_pred['price'].iloc[0]
        
        # Determine side and predicted prices
        # 'Bid Side / Dealer-Purchase', 
        # 'Offered Side / Dealer-Sell', 
        if trade.get('trade_type') == 'S':
            predicted_price_for_trade = offer_price
            predicted_price_opposite = bid_price
        elif trade.get('trade_type') == 'P':
            predicted_price_for_trade = bid_price
            predicted_price_opposite = offer_price
        else:  # 'D' for Dealer
            predicted_price_for_trade = trade_preds['price'].iloc[0]
            predicted_price_opposite = None
        
        result = {
            'cusip': cusip,
            'trade datetime': trade.get('trade_datetime', ''),
            'Side of trade': trade.get('trade_type', ''),
            'amount': int(float(trade['par_traded'])),
            'msrb trade price': float(trade['dollar_price']),
            'ficc predicted price for trade': predicted_price_for_trade,
            'if customer trade, predicted price for opposite side': predicted_price_opposite,
            'predicted bid': bid_price,
            'predicted offer': offer_price,
            'maturity date': trade['maturity_date'],
            'coupon': trade['coupon'],
            # 'rating': trade['sp_long']
        }
        
        results.append(result)
    
    processing_end_time = time.time()
    print(f"Results processing took: {processing_end_time - processing_start_time:.2f} seconds")
    
    # Time the file saving
    save_start_time = time.time()
    results_df = pd.DataFrame(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"yesterday_msrb_analysis_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    save_end_time = time.time()
    print(f"File saving took: {save_end_time - save_start_time:.2f} seconds")
    print(f"Results saved to {filename}")
    
    print("\nSample results:")
    print(f"len results_df: {len(results_df)}")
    
    # Calculate and display total execution time
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    print(f"\n{'='*50}")
    print(f"TOTAL EXECUTION TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"{'='*50}")
        
    return results_df
    
if __name__ == '__main__':
    main()