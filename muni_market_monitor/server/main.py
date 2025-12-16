'''
'''

import os
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime, timedelta
import pytz
import requests
import json
import pickle

from google.cloud import storage

# If running locally, set your Google credentials path as needed.
# (You can remove this if your service is running in GCP where the
#  credentials are provided by default.)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'

app = FastAPI()

# Enable CORS for all origins (adjust as needed for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EASTERN = pytz.timezone('US/Eastern')
FICC_API_URL = "https://api.ficc.ai"

# For demonstration, these remain the same:
GCS_BUCKET_NAME = "market-monitor"
GCS_FILE_NAME = "compliance_prices.pkl"


def parse_response_to_df(response_text: str) -> pd.DataFrame:
    """
    Converts a JSON string (potentially nested) into a Pandas DataFrame.
    This helps keep the code DRY since both real-time and compliance
    endpoints return JSON in a similar structure.
    """
    # First parse the response text
    parsed = json.loads(response_text)
    # Sometimes the JSON might be a string-encoded JSON, so parse again if needed
    if isinstance(parsed, str):
        parsed = json.loads(parsed)

    # Build a DataFrame from dict-of-lists
    return pd.DataFrame({k: pd.Series(v) for k, v in parsed.items()})


def json_to_dataframe(json_result: dict) -> pd.DataFrame:
    """
    Helper: Convert a dict-of-dicts response to a pandas DataFrame.
    (You mentioned this in your code, so I'm leaving it as-is in case it's used elsewhere.)
    """
    return pd.DataFrame({key: pd.Series(value) for key, value in json_result.items()})


def get_holdings_data(limit: int = 15) -> pd.DataFrame:
    """
    Read first N lines of MUB holdings CSV and filter out rows where
    Asset Class is 'Money Market' or 'Cash', or Sector is 'Cash and/or Derivatives'.
    """
    # Read CSV, skipping the first 9 lines
    df = pd.read_csv('MUB_holdings.csv', skiprows=9)
    
    # Select only the columns we need
    df = df[['Name', 'Sector', 'Asset Class', 'CUSIP', 'Market Value']]
    
    # Exclude where 'Asset Class' is 'Money Market' or 'Cash'
    mask_asset = ~df['Asset Class'].isin(['Money Market', 'Cash'])
    # Exclude where 'Sector' matches 'Cash and/or Derivatives'
    mask_sector = ~df['Sector'].str.contains('Cash and/or Derivatives', case=False, na=False)
    
    # Combine both masks
    final_mask = mask_asset & mask_sector
    filtered = df[final_mask].head(limit)

    return filtered


def get_realtime_prices(cusips: list, access_token: str) -> pd.DataFrame:
    """
    Fetch real-time prices from the FICC API.
    """
    url = f"{FICC_API_URL}/api/batchpricing"
    
    # For each CUSIP, define a simple set of parameters
    quantities = [1000] * len(cusips)
    trade_types = ['D'] * len(cusips)
    
    form_data = {
        'access_token': access_token,
        'cusipList': cusips,
        'quantityList': quantities,
        'tradeTypeList': trade_types
    }
    
    # print("Realtime request data:", json.dumps(form_data, indent=2))
    
    response = requests.post(url, data=form_data)
    # Convert response JSON to a DataFrame
    df = parse_response_to_df(response.text)
    
    return df


def get_yesterday_compliance_prices(cusips: list, df_realtime: pd.DataFrame, access_token: str) -> pd.DataFrame:
    """
    Fetch yesterday's 4 PM compliance prices only if necessary:
      - If a stored compliance file exists in Cloud Storage and is from today, re-use it.
      - Otherwise, fetch new compliance prices with the same `access_token` 
        and store them in Cloud Storage.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_FILE_NAME)

    # Check if the file exists in Cloud Storage
    if blob.exists():
        temp_file = "/tmp/compliance_prices.pkl"
        blob.download_to_filename(temp_file)
        with open(temp_file, "rb") as f:
            stored_data = pickle.load(f)
        
        last_updated_date = stored_data.get("date")
        stored_cusips = stored_data.get("cusips", [])

        # If from today and same number of CUSIPs, re-use it
        if last_updated_date == datetime.now(EASTERN).date() and len(stored_cusips) == len(cusips):
            print("> Using cached compliance prices from Cloud Storage.")
            return stored_data["df"]

    print("> Fetching new compliance prices...")

    url = f"{FICC_API_URL}/api/compliance"
    
    # Prepare quantities and trade types
    quantities = [1000] * len(cusips)
    trade_types = ['D'] * len(cusips)
    
    # Use the real-time prices as userPrices
    user_prices = df_realtime['price'].tolist()
    
    # Yesterday at 4 PM
    now = datetime.now(EASTERN)
    yesterday = now - timedelta(days=1)
    yesterday_4pm = yesterday.replace(hour=16, minute=0, second=0, microsecond=0)
    yesterday_4pm_str = yesterday_4pm.strftime('%Y-%m-%d %H:%M:%S')
    
    form_data = {
        'access_token': access_token,
        'cusipList': cusips,
        'quantityList': quantities,
        'tradeTypeList': trade_types,
        'userPriceList': user_prices,
        'tradeDatetimeList': [yesterday_4pm_str] * len(cusips),
        'useCachedPricedFile': 'true'
    }
    
    # print(f"Form data: {form_data}")    
    response = requests.post(url, data=form_data)
    
    # print(f"Response: {response.text}")
    df_compliance = parse_response_to_df(response.text)

    # Store new compliance prices in Cloud Storage
    stored_data = {
        "date": datetime.now(EASTERN).date(),
        "cusips": cusips,
        "df": df_compliance
    }

    temp_file = "/tmp/compliance_prices.pkl"
    with open(temp_file, "wb") as f:
        pickle.dump(stored_data, f)
    
    blob.upload_from_filename(temp_file)
    print("> Stored new compliance prices in Cloud Storage.")

    return df_compliance


def combine_prices(df_realtime: pd.DataFrame, df_yesterday: pd.DataFrame) -> pd.DataFrame:
    """
    Combine real-time and yesterday's data into a single DataFrame.
    """
    if len(df_realtime) != len(df_yesterday):
        raise ValueError("Real-time and yesterday DataFrames have different row counts!")

    df_combined = pd.DataFrame({
        'cusip': df_realtime['cusip'],
        'quantity': df_realtime['quantity'],
        'trade_type': df_realtime['trade_type'],

        'price_realtime': df_realtime['price'],
        'price_yesterday': df_yesterday['price'],
        'price_delta': df_realtime['price'] - df_yesterday['price'],

        'ytw_realtime': df_realtime['ytw'],
        'ytw_yesterday': df_yesterday['ytw'],
        'ytw_delta': df_realtime['ytw'] - df_yesterday['ytw'],

        'coupon': df_realtime['coupon'],
        'security_description': df_realtime['security_description']
    })

    return df_combined


@app.get("/prices")
def get_prices(
    access_token: str = Query(..., description="Firebase Auth token from React"),
    limit: int = 60
):
    """
    Example endpoint that:
      1. Retrieves MUB holdings (top N lines, default 30).
      2. Fetches real-time prices from FICC (using the provided access_token).
      3. Fetches compliance (yesterday 4 PM) prices from FICC (using the same token).
      4. Combines them into a single DataFrame and returns JSON.
    """
    # 1) Get the holdings
    holdings = get_holdings_data(limit=limit)
    cusips = holdings['CUSIP'].tolist()
    print(f"\nProcessing {len(cusips)} CUSIPs from holdings:")
    print(f"CUSIPs: {cusips}")
    
    # 2) Get the real-time DataFrame (pass token)
    df_realtime = get_realtime_prices(cusips, access_token=access_token)
    
    # 3) Get compliance DataFrame (pass the same token)
    df_yesterday = get_yesterday_compliance_prices(cusips, df_realtime, access_token=access_token)
    
    # 4) Combine
    df_final = combine_prices(df_realtime, df_yesterday)
    print(df_final.to_markdown(index=False))
    
    # Return as JSON
    return df_final.to_dict("records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
