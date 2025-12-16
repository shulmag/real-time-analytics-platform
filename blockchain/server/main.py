'''
Description: Flask server for managing municipal bond price data using Soroban contracts
'''

import os
import time
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from google.cloud import secretmanager, bigquery
from soroban_service import (
    init_soroban_server,
    check_connection,
    get_price_history,       # Updated read-only function
    get_all_latest_prices,   # Updated read-only function
    simulate_add_price,      # (Used by /prepare_price_update)
    prepare_transaction_xdr
)

from ficc_api import ficc_prices_request, ficc_prices, read_top_holdings_from_gcs
from daily_cusip_stats import get_most_active_seasoned_cusip
from xdr_parser import parse_latest_prices_xdr, parse_price_history_xdr

from stellar_sdk import TransactionEnvelope, Network
from stellar_sdk.soroban_server import SendTransactionStatus

# ---------------------------
# GCP Credentials Setup
# ---------------------------
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'

def access_secret_version(secret_id: str, project_id: str = 'eng-reactor-287421', version_id='latest'):
    """Access the secret version from GCP Secret Manager"""
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
    response = secretmanager.SecretManagerServiceClient().access_secret_version(
        request={'name': name}
    )
    return response.payload.data.decode('UTF-8')

def get_ficc_credentials():
    """
    Helper function to DRY up reading the FICC username/password.
    """
    return {
        'username': access_secret_version('stellar_ficc_username'),
        'password': access_secret_version('stellar_ficc_password')
    }

# ---------------------------
# Flask App Configuration
# ---------------------------
app = Flask(__name__)
CORS(app)

NETWORK_URL = "https://soroban-rpc.mainnet.stellar.gateway.fm" #  "https://soroban-testnet.stellar.org"
CONTRACT_ID = access_secret_version("soroban_contract_id")
ADMIN_SECRET = access_secret_version("stellar_secret_key")
PUBLIC_KEY = access_secret_version("stellar_public_key")

# Initialize Soroban connection
soroban_context = init_soroban_server(NETWORK_URL)

# Add BigQuery setup
bigquery_client = bigquery.Client()
ORACLE_TABLE_ID = "eng-reactor-287421.api_calls_tracker.muni_oracle_activity"

def log_oracle_activity(transaction_data):
    """
    Log oracle activity to BigQuery.
    
    Args:
        transaction_data (dict): Data about the transaction to log
    """
    try:
        rows_to_insert = [transaction_data]
        errors = bigquery_client.insert_rows_json(ORACLE_TABLE_ID, rows_to_insert)
        
        if errors:
            print(f"Encountered errors while inserting to BigQuery: {errors}")
            return False
        else:
            print(f"Successfully logged oracle activity for transaction: {transaction_data.get('transaction_id')}")
            return True
    except Exception as e:
        print(f"Error logging to BigQuery: {str(e)}")
        return False

def extract_transaction_data(transaction_envelope, response, price_data=None):
    """
    Extract data from a transaction envelope for logging
    """
    try:
        # Check success status correctly for Soroban's SendTransactionStatus
        success = False
        if hasattr(response, 'status'):
            # For Soroban SDK 0.9.0+, status is an enum SendTransactionStatus
            from stellar_sdk.soroban_server import SendTransactionStatus
            if isinstance(response.status, SendTransactionStatus):
                success = response.status == SendTransactionStatus.SUCCESS
            else:
                # Handle string representation
                success = str(response.status) == "SUCCESS" or str(response.status) == "SendTransactionStatus.SUCCESS"
        
        # Start with data we can extract directly
        transaction_data = {
            "transaction_id": response.hash,
            "user_address": transaction_envelope.transaction.source.account_id,
            "source_ip": request.remote_addr,
            "timestamp": time.time(),
            "success": success,
            "fee_paid": transaction_envelope.transaction.fee
        }
        
        # Print incoming price data for debugging
        print("Received price_data:", price_data)
        
        # Use price data from frontend if available
        if price_data and isinstance(price_data, dict):
            # Check for both field name options (yield and yield_value)
            yield_value = None
            if 'yield_value' in price_data:
                yield_value = float(price_data.get('yield_value', 0))
            elif 'yield' in price_data:
                yield_value = float(price_data.get('yield', 0))
                
            transaction_data.update({
                "cusip": price_data.get('cusip'),
                "price": float(price_data.get('price', 0)) if price_data.get('price') else None,
                "yield_value": yield_value,
                "trade_amount": int(price_data.get('trade_amount', 0)) if price_data.get('trade_amount') else None,
                "trade_type": price_data.get('trade_type')
            })
        
        # Print the data for debugging
        print("Extracted transaction data:", transaction_data)
        
        return transaction_data
        
    except Exception as e:
        print(f"Error extracting transaction data: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "transaction_id": response.hash if response else None,
            "user_address": None,
            "timestamp": time.time(),
            "success": False,
            "source_ip": request.remote_addr
        }

# ---------------------------
# Health Check Endpoint
# ---------------------------
@app.route('/health', methods=['GET'])
def health_check():
    """
    Check if the server and Soroban connection are healthy.
    """
    try:
        soroban_status = check_connection(soroban_context)
        return jsonify({
            "status": "healthy" if soroban_status else "degraded",
            "soroban_connection": soroban_status
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

# ---------------------------
# GET Price (Read-Only)
# ---------------------------
@app.route('/get_price/<cusip>', methods=['GET'])
def handle_get_price(cusip):
    """Fetch price history for a specific CUSIP (read-only)."""
    try:
        result = get_price_history(
            context=soroban_context,
            contract_id=CONTRACT_ID,
            cusip=cusip,
            public_key=PUBLIC_KEY
        )
        if result['status'] == 'error':
            return jsonify({"error": result['message']}), 404

        # Parse the history
        history_xdr = result['history']
        parsed_history = parse_price_history_xdr(history_xdr)

        return jsonify({
            "status": "success",
            "cusip": cusip,
            "history": parsed_history
        })

    except Exception as e:
        print(f"Error in handle_get_price: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Latest Prices (Read-Only)
# ---------------------------
@app.route('/latest_prices', methods=['GET'])
def get_latest_prices():
    """
    Get latest prices for all tracked CUSIPs (read-only).
    Called by the BlockchainPrices component in the front-end.
    """
    try:
        limit = request.args.get('limit', type=int, default=None)
        result = get_all_latest_prices(
            context=soroban_context,
            contract_id=CONTRACT_ID,
            public_key=PUBLIC_KEY,
            limit=limit
        )

        if result['status'] == 'error':
            return jsonify({"error": result['message']}), 500

        # Get the raw string representation
        raw_xdr = str(result["prices"])  # Ensure we get the string representation
        
        # Parse it
        parsed_prices = parse_latest_prices_xdr(raw_xdr)
        if not parsed_prices:
            print("Warning: No prices were parsed from XDR")
            print(f"Raw XDR: {raw_xdr[:200]}...")  # Print first 200 chars for debugging
            
        # Replace the raw data with parsed version
        result["prices"] = parsed_prices

        # Sort in descending order by timestamp
        result['prices'] = sorted(
            result['prices'],
            key=lambda p: p['timestamp'],
            reverse=True
)

        return jsonify(result)

    except Exception as e:
        print(f"Error in get_latest_prices: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Ticker (Read-Only)
# ---------------------------
@app.route('/ticker', methods=['GET'])
def get_ticker():
    try:
        print("Starting ticker data retrieval")
        # Get MUB holdings from GCS
        bucket_name = "ficc_blockchain"
        file_name = "MUB_holdings.csv"
        top_holdings = read_top_holdings_from_gcs(bucket_name, file_name)

        print(f"Retrieved {len(top_holdings)} top holdings from GCS")
        
        if top_holdings is None or top_holdings.empty:
            return jsonify({"error": "Failed to read holdings data from GCS"}), 500

        # Limit the number of holdings to process
        top_holdings = top_holdings.head(20)  # Only process first 5
        print(f"Limiting to first 5 holdings: {top_holdings['CUSIP'].tolist()}")
        
        # Fetch data for each CUSIP individually
        ticker_data = []
        
        for _, row in top_holdings.iterrows():
            cusip = row["CUSIP"]
            print(f"Processing CUSIP: {cusip}")
            
            ticker_entry = {
                "name": row["Name"],
                "cusip": cusip,
                "marketValue": float(
                    row["Market Value"].replace('$', '').replace(',', '')
                ) if isinstance(row["Market Value"], str) else float(row["Market Value"])
            }
            
            # Try to get price data for this specific CUSIP
            try:
                price_result = get_price_history(
                    context=soroban_context,
                    contract_id=CONTRACT_ID,
                    cusip=cusip,
                    public_key=PUBLIC_KEY
                )
                
                if price_result['status'] == 'success' and price_result.get('history'):
                    # Parse the history
                    history_xdr = price_result['history']
                    parsed_history = parse_price_history_xdr(history_xdr)
                    
                    # Use the latest price entry (first one - they're sorted newest first)
                    if parsed_history and len(parsed_history) > 0:
                        latest_price = parsed_history[0]
                        print(f"Found data for {cusip}: {latest_price}")
                        
                        if 'price' in latest_price:
                            ticker_entry["price"] = latest_price['price']
                        
                        if 'yield_value' in latest_price:
                            ticker_entry["yieldToWorst"] = latest_price['yield_value']
                            
                        if 'trade_amount' in latest_price:
                            ticker_entry["tradeAmount"] = latest_price['trade_amount']
                            
                        if 'trade_type' in latest_price:
                            ticker_entry["tradeType"] = latest_price['trade_type']
                            
                        if 'timestamp' in latest_price:
                            ticker_entry["lastUpdate"] = latest_price['timestamp']
            except Exception as e:
                print(f"Error fetching price for CUSIP {cusip}: {e}")
                # Continue with next CUSIP
            
            # Add to results whether we got price data or not
            ticker_data.append(ticker_entry)

        # Sort by market value descending
        ticker_data.sort(key=lambda x: x['marketValue'], reverse=True)

        response = {
            "status": "success",
            "count": len(ticker_data),
            "timestamp": int(time.time()),
            "ticker": ticker_data
        }
        
        print(f"Returning {len(ticker_data)} ticker entries")
        return jsonify(response)

    except Exception as e:
        print(f"Error in get_ticker: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Prepare Price Update
# ---------------------------
@app.route('/prepare_price_update', methods=['POST'])
def prepare_price_update():
    """
    Prepare transaction for frontend signing and simulate fees.
    Called by the new front-end code in handleGetAndUpdatePrice().
    """
    try:
        data = request.json

        request_params = {
            'username': access_secret_version('stellar_ficc_username'),
            'password': access_secret_version('stellar_ficc_password'),
            'cusip': data['cusip'],
            'amount': int(data['trade_amount']),
            'tradeType': data['trade_type']
        }

        # Call FICC API
        real_time_price = ficc_prices_request(request_params)


        # -------------------------------
        # 1) Check for explicit error
        # -------------------------------
        if 'error' in real_time_price:
            # If the FICC API response has an "error" key, return it directly.
            return jsonify({
                "status": "error",
                "error": real_time_price['error']  # actual FICC error message
            }), 200

        # -----------------------------------------
        # 2) Check for valid price data in the API response
        # -----------------------------------------
        if isinstance(real_time_price, list) and len(real_time_price) > 0:
            # A list with at least one price object
            price_data = real_time_price[0]
        elif isinstance(real_time_price, dict) and 'price' in real_time_price:
            # A single dict with "price"
            price_data = real_time_price
        else:
            # Nothing we recognize => Unexpected format
            error_msg = f"Unexpected FICC API response format: {real_time_price}"
            print(error_msg)
            return jsonify({
                "status": "error",
                "error": error_msg
            }), 500

        # Extract numeric values
        price = float(price_data.get('price', 0))
        ytw = float(price_data.get('ficc_ytw', 0))

        # Simple validation
        if price <= 0:
            return jsonify({
                "status": "error",
                "error": "Invalid price received from FICC"
            }), 500

        # -----------------------------------
        # Simulate fees (no error so far)
        # -----------------------------------
        fees = simulate_add_price(
            context=soroban_context,
            contract_id=CONTRACT_ID,
            public_key=data['public_key'],
            cusip=data['cusip'],
            price=price,
            yield_value=ytw,
            trade_amount=int(data['trade_amount']),
            trade_type=data['trade_type']
        )
        total_xlm = fees['total_fee'] / 10000000  # Convert stroops to XLM

        # -----------------------------------
        # Prepare final transaction XDR
        # -----------------------------------
        prepared_xdr = prepare_transaction_xdr(
            context=soroban_context,
            contract_id=CONTRACT_ID,
            public_key=data['public_key'],
            cusip=data['cusip'],
            price=price,
            yield_value=ytw,
            trade_amount=int(data['trade_amount']),
            trade_type=data['trade_type'],
            fee=fees['total_fee']
        )
        
        return jsonify({
            "status": "success",
            "transaction_xdr": prepared_xdr,
            "price_data": {
                "price": price,
                "yield": ytw
            },
            "estimated_fees": {
                "base_fee_xlm": fees['base_fee'] / 10000000,
                "resource_fee_xlm": fees['resource_fee'] / 10000000,
                "total_xlm": total_xlm,
                "approximate_usd": round(total_xlm * 0.33, 2)  # Rough conversion
            }
        })

    except Exception as e:
        error_msg = str(e)
        print(f"Error in prepare_price_update: {error_msg}")
        return jsonify({
            "status": "error",
            "error": error_msg
        }), 500


# ---------------------------
# Submit Transaction
# ---------------------------
@app.route('/submit_transaction', methods=['POST'])
def submit_transaction():
    """
    Submit a signed transaction to the Soroban network.
    Called by the new front-end code in handleTransactionConfirm().
    """
    try:
        data = request.json
        print("Received signed transaction data:", data)
        
        if not data or 'signed_xdr' not in data:
            return jsonify({"error": "No signed transaction provided"}), 400

        try:
            # Extract price data for logging
            price_data = data.get('price_data')
            
            # Parse the envelope from XDR
            transaction_envelope = TransactionEnvelope.from_xdr(
                data['signed_xdr'], 
                network_passphrase=Network.PUBLIC_NETWORK_PASSPHRASE
            )
            
            # Submit the transaction
            response = soroban_context["soroban_server"].send_transaction(
                transaction_envelope
            )
            print("Submit response:", response)

            # Convert enum to string for JSON serialization
            status = str(response.status) if response.status else "ERROR"
            
            # Create transaction data for logging
            from google.cloud import bigquery
            
            try:
                bigquery_client = bigquery.Client()
                ORACLE_TABLE_ID = "eng-reactor-287421.api_calls_tracker.muni_oracle_activity"
                
                # Assume most transactions will eventually succeed
                # This is just for logging purposes
                is_success = status == "PENDING" or status == "SUCCESS"
                
                # Prepare the transaction data
                transaction_data = {
                    "transaction_id": response.hash,
                    "user_address": transaction_envelope.transaction.source.account_id,
                    "timestamp": time.time(),
                    "source_ip": request.remote_addr,
                    "success": is_success,
                    "fee_paid": transaction_envelope.transaction.fee
                }
                
                # Add price data if available
                if price_data:
                    transaction_data.update({
                        "cusip": price_data.get('cusip'),
                        "price": price_data.get('price'),
                        "yield_value": price_data.get('yield') or price_data.get('yield_value'),
                        "trade_amount": price_data.get('trade_amount'),
                        "trade_type": price_data.get('trade_type')
                    })
                
                # Log to BigQuery
                rows_to_insert = [transaction_data]
                errors = bigquery_client.insert_rows_json(ORACLE_TABLE_ID, rows_to_insert)
                
                if errors:
                    print(f"BigQuery insert errors: {errors}")
                else:
                    print(f"Successfully logged transaction to BigQuery: {response.hash}")
                    
            except Exception as e:
                print(f"Error logging to BigQuery: {e}")
                # Continue even if logging fails
            
            # Return response to client
            return jsonify({
                "status": "success",
                "transaction_hash": response.hash,
                "result": status,
                "ledger": response.latest_ledger
            })

        except Exception as e:
            print("Error submitting to Soroban:")
            import traceback
            print(traceback.format_exc())
            return jsonify({"error": f"Failed to submit to Soroban: {str(e)}"}), 500

    except Exception as e:
        print("Error in submit_transaction:")
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
@app.route('/most_active_cusip', methods=['GET'])
def handle_most_active_cusip():
    """
    Get the most actively traded seasoned CUSIP from yesterday.
    """
    try:
        stats_df = get_most_active_seasoned_cusip()
        print(stats_df)
        if stats_df.empty:
            return jsonify({
                "status": "error",
                "message": "No data found for yesterday"
            }), 404
        
        # Extract the first (and only) row as a dictionary
        stats = stats_df.iloc[0].to_dict()
        
        return jsonify({
            "status": "success",
            "data": {
                "cusip": stats.get("most_actively_traded_seasoned_cusip"),
                "description": stats.get("description"),
                "trade_count": int(stats.get("number_of_trades", 0))
            },
            "timestamp": int(time.time())
        })
        
    except Exception as e:
        print(f"Error in handle_most_active_cusip: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Failed to get most active CUSIP: {str(e)}"
        }), 500
    

# ---------------------------
# Run Flask App
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5001)
