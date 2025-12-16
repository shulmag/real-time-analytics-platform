"""
Description: Cloud Function for pricing MUB ETF holdings and 
storing the data using Soroban smart contracts on Stellar Mainnet.
"""

import os
import json
import requests
import pandas as pd
import io
import time
import random
from typing import Dict, Any
from google.cloud import storage, secretmanager
from pprint import pprint
from stellar_sdk import (
    Server, 
    TransactionBuilder,
    scval, 
    Address,
    Account, 
    Keypair,
    Network
)

from soroban_module import (
    init_soroban_server,
    check_connection,
    simulate,
    prepare,
    sign,
    send,
    read_price_data,
    FeeTracker
)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'
BASE_FEE = 10000  


#####################################################
#    [1] HELPER FUNCTIONS & GCP ACCESS
#####################################################

def access_secret_version(secret_id: str, project_id: str = 'eng-reactor-287421', version_id='latest'):
    """
    Access a secret from GCP Secret Manager.
    """
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(request={'name': name})
    return response.payload.data.decode('UTF-8')


def read_top_holdings_from_gcs(bucket_name: str, file_name: str, num_holdings=10) -> pd.DataFrame:
    """
    Read holdings from a CSV file stored in Google Cloud Storage.
    If num_holdings is None, reads all holdings.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        csv_content = blob.download_as_text()

        holdings = pd.read_csv(
            io.StringIO(csv_content),
            skiprows=9  # skip metadata rows if needed
        )

        holdings = holdings[['Name', 'CUSIP', 'Market Value']]
        
        if num_holdings is not None:
            holdings = holdings.head(num_holdings)
            
        # Add debug logging
        print(f"*** Read {len(holdings)} holdings from {file_name}")
        return holdings

    except Exception as e:
        print(f"Error reading from GCS: {e}")
        return pd.DataFrame()


def ficc_prices(cusips_quantities_tradetypes, username, password):
    """
    Fetch batch pricing data for municipal bonds from FICC API.
    The exact payload and response shape depends on your API.
    """
    base_url = 'https://api.ficc.ai/api/batchpricing'
    try:
        cusip_list = [row['CUSIP'] for row in cusips_quantities_tradetypes]
        quantity_list = [row['Quantity'] for row in cusips_quantities_tradetypes]
        trade_type_list = [row['Trade Type'] for row in cusips_quantities_tradetypes]

        payload = {
            'username': username,
            'password': password,
            'cusipList': cusip_list,
            'quantityList': quantity_list,
            'tradeTypeList': trade_type_list,
        }

        response = requests.post(base_url, data=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}


def check_account(public_key: str) -> bool:
    """
    Verify the Stellar account exists and is properly funded on Mainnet.
    """
    try:
        server = Server("https://horizon.stellar.org")
        account = server.load_account(public_key)
        print(f"Account found: {public_key}")
        print(f"Sequence: {account.sequence}")
        return True
    except Exception as e:
        print(f"Error checking account: {type(e).__name__}: {str(e)}")
        print("Make sure the account is properly funded.")
        return False

#####################################################
#   [2]  SOROBAN "ADD_PRICE" HELPER
#####################################################

def add_price_to_contract(
    soroban_context: dict,
    contract_id: str,
    public_key: str,
    secret_key: str,
    cusip: str,
    price: float,
    yield_value: float,
    trade_amount: int,
    trade_type: str,
    fee_tracker
) -> Dict[str, Any]:
    """
    Add a price record to the Soroban contract for a given CUSIP.
    """
    try:
        server = Server("https://horizon.stellar.org")
        source_account = server.load_account(public_key)

        # Prepare parameters for the transaction
        params = [
            scval.to_address(Address(public_key)),  # Caller (Address)
            scval.to_string(cusip),                # CUSIP (String)
            scval.to_int128(int(price * 1000)),    # Price in millicents (i128)
            scval.to_int128(int(yield_value * 1000)),  # Yield in millibps (i128)
            scval.to_int128(trade_amount),         # Trade amount (i128)
            scval.to_string(trade_type),          # Trade type (String)
        ]

        print(f"[add_price_to_contract] Building transaction for CUSIP={cusip}")

        # Build the transaction with higher initial fee
        transaction = (
            TransactionBuilder(
                source_account=source_account,
                network_passphrase=Network.PUBLIC_NETWORK_PASSPHRASE,  # Using PUBLIC network instead of TESTNET
                base_fee=BASE_FEE  
            )
            .set_timeout(600)    # Doubled timeout
            .append_invoke_contract_function_op(
                contract_id=contract_id,
                function_name="add_price",
                parameters=params
            )
            .build()
        )

        # Simulate the transaction
        simulation_result = simulate(soroban_context, transaction)
        print(f"[add_price_to_contract] Simulation result: {simulation_result}")

        # Prepare the transaction
        prepared_transaction = prepare(soroban_context, transaction, simulation_result)
        
        # Extract fees and add to tracker
        fee_info = {
            'base_fee': int(str(prepared_transaction.transaction.fee).split('[')[-1].split(']')[0]),  # Extract the number from the string
            'resource_fee': int(str(prepared_transaction.transaction.soroban_data.resource_fee).split('int64=')[-1].split(']')[0])
                if hasattr(prepared_transaction.transaction, 'soroban_data') 
                and hasattr(prepared_transaction.transaction.soroban_data, 'resource_fee') 
                else 0,
        }
        fee_info['total_fee'] = fee_info['base_fee'] + fee_info['resource_fee']        
        # Add to tracker
        fee_tracker.add_transaction_fees(fee_info)
        # Sign the transaction
        signed_transaction = sign(prepared_transaction, secret_key)

        # Send the transaction
        response = send(soroban_context, signed_transaction)

        # Handle PENDING and TRY_AGAIN_LATER
        if response.status == "PENDING":
            final_status = poll_transaction_status(soroban_context, response.hash)
            if final_status == "SUCCESS":
                return {"status": "success", "cusip": cusip, "transaction_hash": response.hash}
            return {"status": "error", "cusip": cusip, "message": f"Transaction failed with status: {final_status}"}

        elif response.status == "TRY_AGAIN_LATER":
            retry_response = retry_transaction(soroban_context, signed_transaction)
            return {"status": retry_response.status, "cusip": cusip, "hash": retry_response.hash}

        print(f"[add_price_to_contract] Transaction sent. Result: {response}")

        # Check transaction status
        if response.status == "SUCCESS":
            return {
                "status": "success",
                "cusip": cusip,
                "price": price,
                "yield_value": yield_value,
                "transaction_hash": response.hash,
            }
        else:
            return {
                "status": "error",
                "cusip": cusip,
                "message": f"Transaction failed with status: {response.status}",
                "hash": response.hash,
            }

    except Exception as e:
        print(f"Error adding price for CUSIP {cusip}: {str(e)}")
        return {"status": "error", "message": str(e), "cusip": cusip}

def poll_transaction_status(soroban_context, transaction_hash, max_retries=8, initial_delay=2):
    """
    Polls the status of a transaction with exponential backoff.
    """
    import time

    for attempt in range(max_retries):
        try:
            response = soroban_context["soroban_server"].get_transaction_status(transaction_hash)
            status = response.get("status")
            
            if status in ["SUCCESS", "FAILED"]:
                return status
                
            # Exponential backoff with randomization
            delay = initial_delay * (2 ** attempt) + (random.uniform(0, 1))
            print(f"[poll_transaction_status] Attempt {attempt+1}/{max_retries}: Status = {status}, waiting {delay:.2f}s")
            time.sleep(delay)
            
        except Exception as e:
            print(f"Error polling transaction status: {str(e)}")
            time.sleep(initial_delay)
            
    return "TIMEOUT"

def retry_transaction(soroban_context, signed_transaction, max_retries=5, initial_delay=3):
    """
    Retries a signed transaction with exponential backoff and higher fees.
    """
    import time
    import random

    for attempt in range(max_retries):
        try:
            # Increase fee with each retry
            fee_multiplier = 1.5 ** attempt
            signed_transaction.transaction.fee = int(100000 * fee_multiplier)
            
            response = send(soroban_context, signed_transaction)
            if response.status != "TRY_AGAIN_LATER":
                return response
                
            # Exponential backoff with jitter
            delay = initial_delay * (2 ** attempt) + (random.uniform(0, 1))
            print(f"[retry_transaction] Attempt {attempt+1}/{max_retries}: Status = TRY_AGAIN_LATER, "
                  f"fee = {signed_transaction.transaction.fee}, waiting {delay:.2f}s")
            time.sleep(delay)
            
        except Exception as e:
            print(f"Error retrying transaction: {str(e)}")
            time.sleep(initial_delay)
            
    return {"status": "FAILED"}


def check_blockchain_data(max_holdings):
    """Check what data is actually stored on the blockchain for the first N holdings"""
    try:
        # Initialize Soroban and get secrets
        rpc_url = "https://soroban-rpc.mainnet.stellar.gateway.fm"  # Updated to mainnet RPC
        soroban_context = init_soroban_server(rpc_url)
        
        public_key = access_secret_version("stellar_public_key")
        secret_key = access_secret_version("stellar_secret_key")
        contract_id = access_secret_version("soroban_contract_id")
        
        # Read holdings from CSV with limit
        bucket_name = "ficc_blockchain"
        file_name = "MUB_holdings.csv"
        holdings = read_top_holdings_from_gcs(bucket_name, file_name, num_holdings=20)
        print(f"Holdings: {holdings}")
        if holdings.empty:
            return {"error": "No holdings data found"}
            
        print(f"Checking {len(holdings)} holdings on blockchain...")
        results = []
        for _, row in holdings.iterrows():
            cusip = row["CUSIP"]
            try:
                data = read_price_data(
                    soroban_context, 
                    contract_id, 
                    cusip,
                    public_key,
                    secret_key
                )
                results.append({
                    "CUSIP": cusip,
                    "Name": row["Name"],
                    "Blockchain Data": data
                })
            except Exception as e:
                results.append({
                    "CUSIP": cusip,
                    "Name": row["Name"],
                    "Error": str(e)
                })
        
        # Pretty print the results
        print("\n=== Results ===")
        pprint(results)  # Pretty print for structured output
                
    except Exception as e:
        return {"error": f"Failed to check blockchain data: {str(e)}"}

#####################################################
#   [3]  MAIN FUNCTION (Cloud Function entrypoint)
#####################################################

def main(request):
    """
    Cloud Function that processes MUB holdings, fetches prices,
    and stores them on the blockchain via Soroban.
    """
    # At the start of main():
    fee_tracker = FeeTracker()
    try:
        #######################
        # 1) Soroban Setup
        #######################

        # Update to use mainnet RPC URL
        rpc_url = "https://soroban-rpc.mainnet.stellar.gateway.fm"
        soroban_context = init_soroban_server(rpc_url)

        # Check Soroban RPC connection
        print("Checking Soroban RPC connection...")
        if not check_connection(soroban_context):
            return {"error": "Unable to connect to Soroban RPC endpoint"}, 500

        # 2) Load secrets
        public_key = access_secret_version("stellar_public_key")
        secret_key = access_secret_version("stellar_secret_key")
        username   = access_secret_version("stellar_ficc_username")
        password   = access_secret_version("stellar_ficc_password")

        # 3) Check Stellar account
        if not check_account(public_key):
            return {"error": "Stellar account not found or not properly funded"}, 500

        # 4) Contract ID
        CONTRACT_ID = access_secret_version("soroban_contract_id")
        print(f"Using contract ID: {CONTRACT_ID}")

        #######################
        # 5) Holdings & Pricing
        #######################
        bucket_name = "ficc_blockchain"
        file_name = "MUB_holdings.csv"

        holdings = read_top_holdings_from_gcs(bucket_name, file_name,num_holdings=20)

        if holdings.empty:
            return {"error": "No holdings data found"}, 500

        # Prepare data for FICC API
        pricing_request_data = [
            {"CUSIP": row["CUSIP"], "Quantity": 1000, "Trade Type": "D"}
            for _, row in holdings.iterrows()
        ]
        
        # Call the external FICC pricing API
        pricing_results = ficc_prices(pricing_request_data, username, password)
        
        # if "error" in pricing_results:
        #     return {"error": f"FICC API Error: {pricing_results['error']}"}, 500
        print(f"Pricing results: {pricing_results}")

        #######################
        # 6) Storing on Soroban
        #######################
        results = []

        # Ensure pricing_results is a dictionary by parsing JSON if needed
        if isinstance(pricing_results, str):
            try:
                pricing_results = json.loads(pricing_results)
                print("Debug: Parsed pricing_results as dictionary.")
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse pricing_results as JSON: {e}")

        # Validate and parse pricing_results
        if not isinstance(pricing_results, dict):
            raise ValueError("Invalid structure: pricing_results is not a dictionary.")

        if "cusip" not in pricing_results or not isinstance(pricing_results["cusip"], dict):
            raise ValueError("Invalid structure: 'cusip' field is missing or not a dictionary.")

        # Iterate over all indices in pricing_results["cusip"]
        for idx in pricing_results["cusip"].keys():  # Ensure idx is treated as a string
            try:
                # Debugging: Print the current index being processed
                print(f"Debug: Processing index {idx}")

                # Access data using the string index
                cusip_str = pricing_results["cusip"][idx]
                price_val = pricing_results["price"][idx]
                ytw_val = pricing_results["ytw"][idx]

                # Debugging: Print the values being processed
                print(f"  CUSIP: {cusip_str}, Price: {price_val}, YTW: {ytw_val}")

                # Skip invalid prices
                if price_val == -1:
                    results.append({
                        "CUSIP": cusip_str,
                        "status": "skipped",
                        "reason": "Invalid price"
                    })
                    continue

                # Call the Soroban "add_price" contract function
                resp = add_price_to_contract(
                    soroban_context=soroban_context,
                    contract_id=CONTRACT_ID,
                    public_key=public_key,
                    secret_key=secret_key,
                    cusip=cusip_str,
                    price=price_val,
                    yield_value=ytw_val,
                    trade_amount=1000,
                    trade_type="D",  # 'D' for Dealer
                    fee_tracker=fee_tracker
                )
                results.append(resp)
                # Add delay between transactions
                time.sleep(5 + random.uniform(0, 2))  # 5-7 second delay between CUSIPs
                
            except Exception as e:
                print(f"Error processing CUSIP at index {idx}: {str(e)}")
                results.append({
                    "CUSIP": pricing_results["cusip"].get(idx, "Unknown"),
                    "status": "error",
                    "reason": str(e)
                })
        
        fees_str = fee_tracker.print_summary()
        print(fees_str)
        return {"status": "completed", "results": results,"fees":fees_str}, 200

    except Exception as e:
        error_message = f"Error processing MUB holdings: {str(e)}"
        print(error_message)
        return {"error": error_message}, 500

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        print("\nChecking blockchain data...")
        check_blockchain_data(10)  # Check first 10 holdings
        
    else:
        # Normal price update flow
        test_request = {"message": "Test function logic"}
        response = main(test_request)
        print(json.dumps(response[0], indent=2))