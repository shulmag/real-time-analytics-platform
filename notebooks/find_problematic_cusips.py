import os
import requests
import time
import json
from google.cloud import bigquery
from google.cloud import bigquery_storage

# Set credentials path
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'

# Initialize BigQuery clients
bq_client = bigquery.Client()
bqstorage_client = bigquery_storage.BigQueryReadClient()

# Define global timeout (in seconds)
API_TIMEOUT = 9  # 9 second timeout for faster testing

# Define helper function to execute SQL queries
def sqltodf(sql, bq_client):
    '''Taken mostly from `ficc.utils.auxiliary_functions.py`. Decided not to import the function because there are lots of package dependencies in `ficc_python`. 
    Added `bqstorage_client` optional argument for speed.'''
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe(bqstorage_client=bigquery_storage.BigQueryReadClient())  # optional argument inside `.to_dataframe(...)` is used to significantly speed up the procedure by parallelizing the reads and supporting gRPC-based communication

# Fill out the EMAIL and PASSWORD variables below
EMAIL = 'gil@ficc.ai'
PASSWORD = 'TODO'

# API function
def ficc_prices(cusips_quantities_tradetypes):
    base_url = 'https://api.ficc.ai/api/batchpricing'
    cusip_list = [row['CUSIP'] for row in cusips_quantities_tradetypes]
    quantity_list = [row['Quantity'] for row in cusips_quantities_tradetypes]
    trade_type_list = [row['Trade Type'] for row in cusips_quantities_tradetypes]
    
    try:
        # Use timeout to stop waiting after API_TIMEOUT seconds
        response = requests.post(
            base_url, 
            data={
                'username': EMAIL, 
                'password': PASSWORD, 
                'cusipList': cusip_list, 
                'quantityList': quantity_list, 
                'tradeTypeList': trade_type_list
            },
            timeout=API_TIMEOUT
        )
        
        # Check if response status is successful
        if response.status_code != 200:
            print(f"API returned status code {response.status_code}")
            return {"error": f"API returned status code {response.status_code}"}
            
        try:
            # Try to parse as JSON - any valid response is considered successful
            if isinstance(response.text, str) and response.text.strip():
                try:
                    result = json.loads(response.text)
                    return {"success": True, "data": result}
                except:
                    # If that fails, use the built-in json method
                    result = response.json()
                    return {"success": True, "data": result}
            else:
                # Empty response
                return {"error": "Empty response"}
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {response.text[:100]}...")
            return {"error": "Failed to parse JSON response"}
    except requests.exceptions.Timeout:
        print(f"API call timed out after {API_TIMEOUT} seconds")
        return {"error": "API timeout", "timeout": True}
    except Exception as e:
        print(f"Exception in API call: {str(e)}")
        return {"error": str(e)}

# Get all CUSIPs in the specified range
def get_cusips():
    sql = """
    SELECT DISTINCT cusip 
    FROM `reference_data_v2.reference_data_flat` 
    WHERE cusip >= '649907NY4' AND cusip <= '64990ASV8' AND outstanding_indicator 
    ORDER BY cusip
    """
    df = sqltodf(sql, bq_client)
    return df['cusip'].tolist()

# Test if a batch of CUSIPs works
def test_cusips(cusips):
    if not cusips:
        print("No CUSIPs to test")
        return True, []
        
    print(f"Testing {len(cusips)} CUSIPs...")
    cusips_quantities_tradetypes = [{'CUSIP': cusip, 'Quantity': 1000, 'Trade Type': 'D'} for cusip in cusips]
    
    result = ficc_prices(cusips_quantities_tradetypes)
    
    # For debugging
    print(f"API response type: {type(result)}")
    
    # The only real errors are timeouts or explicit errors
    if isinstance(result, dict) and 'error' in result:
        if result.get("timeout", False):
            print(f"API timed out - these CUSIPs might be problematic")
        else:
            print(f"Got error: {result.get('error', 'Unknown error')}")
        
        # If we're testing a single CUSIP, it's definitely problematic
        if len(cusips) == 1:
            return False, cusips
        
        # For multiple CUSIPs, we need to test further to find which one is problematic
        return False, cusips
    
    # Any other response is considered successful
    print("API call successful")
    return True, []

# Find all problematic CUSIPs via binary search
def find_problematic_cusips_binary(cusips):
    print(f"Searching for problematic CUSIPs among {len(cusips)} CUSIPs using binary search")
    
    # Base case: single CUSIP
    if len(cusips) == 1:
        success, problematic = test_cusips(cusips)
        return problematic  # Will be empty list if not problematic
    
    # Base case: empty list
    if not cusips:
        return []
    
    # Test the entire batch first
    success, _ = test_cusips(cusips)
    
    # If successful, no problematic CUSIPs
    if success:
        print("All CUSIPs in this batch work fine")
        return []
    
    # Split and recurse - binary search
    mid = len(cusips) // 2
    first_half = cusips[:mid]
    second_half = cusips[mid:]
    
    print(f"Dividing into batches of {len(first_half)} and {len(second_half)} CUSIPs")
    
    # Recursively find problematic CUSIPs in each half
    problematic_first = find_problematic_cusips_binary(first_half)
    problematic_second = find_problematic_cusips_binary(second_half)
    
    # Combine results
    return problematic_first + problematic_second

# Verify a list of potentially problematic CUSIPs
def verify_problematic_cusips(cusips):
    confirmed_problematic = []
    print(f"\nVerifying {len(cusips)} potentially problematic CUSIPs...")
    
    for i, cusip in enumerate(cusips):
        print(f"Testing CUSIP {i+1}/{len(cusips)}: {cusip}")
        success, problematic = test_cusips([cusip])
        if not success:
            print(f"Confirmed: {cusip} is problematic")
            confirmed_problematic.append(cusip)
        else:
            print(f"{cusip} works fine individually")
        time.sleep(0.5)  # Small delay to avoid rate limiting
        
    return confirmed_problematic

def main():
    start_time = time.time()
    print(f"Starting search at {time.strftime('%H:%M:%S')}")
    
    # Get all CUSIPs in the specified range
    all_cusips = get_cusips()
    print(f"Found {len(all_cusips)} CUSIPs in the specified range")
    
    # Specifically check 649907T85 first (known problematic)
    print("\nTesting known problematic CUSIP: 649907T85")
    success, _ = test_cusips(['649907T85'])
    if not success:
        print("Confirmed 649907T85 is problematic")
    else:
        print("649907T85 appears to work fine - check our understanding of 'problematic'")
    
    # Use binary search to find problematic CUSIPs efficiently
    potentially_problematic = find_problematic_cusips_binary(all_cusips)
    
    if potentially_problematic:
        print(f"\nFound {len(potentially_problematic)} potentially problematic CUSIPs:")
        for cusip in potentially_problematic:
            print(f"- {cusip}")
        
        # Final verification of each problematic CUSIP
        confirmed_problematic = verify_problematic_cusips(potentially_problematic)
        
        if confirmed_problematic:
            print(f"\nConfirmed {len(confirmed_problematic)} problematic CUSIPs:")
            for cusip in confirmed_problematic:
                print(f"- {cusip}")
            
            # Save to file
            with open('problematic_cusips.txt', 'w') as f:
                for cusip in confirmed_problematic:
                    f.write(f"{cusip}\n")
            print(f"Saved list to problematic_cusips.txt")
        else:
            print("\nNo problematic CUSIPs confirmed.")
    else:
        print("\nNo problematic CUSIPs found.")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Finished at {time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()