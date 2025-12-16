'''
Description: use the api to get a single price
'''

import requests
import pandas as pd
import json
from google.cloud import storage
import io

def ficc_prices_request(request_params):
    """
    Fetch real-time prices from the ficc.ai API.

    Args:
        request_params (dict): Query parameters for the API request.

    Returns:
        dict: Parsed API response or an error message.
    """

    url = "https://api.ficc.ai/api/price"
    try:
        # Create a new dict and transform quantity to amount
        api_params = request_params.copy()
        if 'quantity' in api_params:
            api_params['amount'] = api_params.pop('quantity')
        
        # Construct the query parameters as a string
        query_string = "&".join(f"{key}={value}" for key, value in api_params.items())
        full_url = f"{url}?{query_string}"
        print(f"Request URL: {full_url}")  # This will now show 'amount' instead of 'quantity'

        # Perform GET request
        response = requests.get(full_url)
        response.raise_for_status()  # Raise an error for HTTP errors
        
        # Parse JSON response
        data = response.json()
        print(f"API response: {data}")
        # Ensure the response is in the expected format
        if isinstance(data, list) and data:
            return data[0]  # Return the first object if it's a list
        else:
            return data
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    
def ficc_prices(cusips_quantities_tradetypes, username, password):
    """
    Fetch batch pricing data for municipal bonds from FICC API.

    Args:
        cusips_quantities_tradetypes (list): List of dicts with CUSIP, Quantity, and Trade Type.
        username (str): Username for FICC API.
        password (str): Password for FICC API.

    Returns:
        list: Parsed API response for each municipal bond, excluding those with invalid prices.
    """
    base_url = 'https://api.ficc.ai/api/batchpricing'
    try:
        cusip_list = [row['CUSIP'] for row in cusips_quantities_tradetypes]
        quantity_list = [row['Quantity'] for row in cusips_quantities_tradetypes]
        trade_type_list = [row['Trade Type'] for row in cusips_quantities_tradetypes]

        # API request payload
        payload = {
            'username': username,
            'password': password,
            'cusipList': cusip_list,
            'quantityList': quantity_list,
            'tradeTypeList': trade_type_list,
        }

        # Send POST request to FICC API
        response = requests.post(base_url, data=payload)
        response.raise_for_status()  # Raise an error for HTTP issues
        
        # Parse response JSON
        results = response.json()
        return results
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}

    
def read_top_holdings_from_gcs(bucket_name: str = "ficc_blockchain", file_name: str = "MUB_holdings.csv", num_holdings: int = 20) -> pd.DataFrame:
    """
    Read the top holdings from a CSV file stored in Google Cloud Storage.
    
    Args:
        bucket_name (str): Name of the GCS bucket (default: "ficc_blockchain")
        file_name (str): Name of the CSV file (default: "MUB_holdings.csv")
        num_holdings (int): Number of top holdings to read (default: 20)
        
    Returns:
        pd.DataFrame: DataFrame containing the holdings data
    """
    try:
        # Initialize the GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Download the CSV content as a string
        csv_content = blob.download_as_text()
        
        # Read the CSV content into a DataFrame
        holdings = pd.read_csv(
            io.StringIO(csv_content),
            skiprows=9  # Skip the metadata rows
        )
        
        # Select relevant columns and limit to top N holdings
        holdings = holdings[['Name', 'CUSIP', 'Market Value']]
        if num_holdings:
            holdings = holdings.head(num_holdings)
        
        return holdings
        
    except Exception as e:
        print(f"Error reading from GCS: {e}")
        return None


