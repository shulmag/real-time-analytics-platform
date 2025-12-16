"""
Version 2 of Investment Grade Bond Spread Analyzer - Simplified Version
Tracks daily spreads for IG bonds with 4.5-5.5, and 9.5-10.5 years maturity
At 1000 and 100 quantities. 
"""
import pandas as pd
from google.cloud import storage, bigquery
import os
from datetime import datetime, timedelta
import re
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
from pytz import timezone

# # Set credentials
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/hadassahlurbur/repos/ficc/creds.json'

# Calendar and timezone constants
class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    """Custom US Federal Holiday calendar that includes Good Friday"""
    rules = USFederalHolidayCalendar.rules + [GoodFriday]

BUSINESS_DAY = CustomBusinessDay(calendar=USHolidayCalendarWithGoodFriday())
EASTERN = timezone('US/Eastern')

def extract_datetime_from_filename(filename):
    """Extract datetime from filename"""
    pattern = r'priced_(\d{4}-\d{2}-\d{2})--(\d{2}-\d{2}-\d{2})_'
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2).replace('-', ':')
        datetime_str = f"{date_str} {time_str}"
        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    return None

def get_last_file_for_date(storage_client, date):
    """Get the last (most recent) pricing file for a given date"""
    date_str = date.strftime('%Y-%m-%d')
    bucket = storage_client.bucket('large_batch_pricing')
    prefix = f"{date_str}/"
    
    # List all files for this date
    blobs = list(bucket.list_blobs(prefix=prefix))
    csv_files = [b for b in blobs if b.name.endswith('.csv') and 'priced_' in b.name]
    
    if not csv_files:
        return None
    
    # Sort by filename (which contains timestamp) and get the last one
    csv_files.sort(key=lambda x: x.name)
    return f"gs://large_batch_pricing/{csv_files[-1].name}"


def calculate_spreads_for_file(gcs_path, reference_df, storage_client):
    """Calculate spreads for a single file"""
    # Parse GCS path
    if gcs_path.startswith('gs://'):
        gcs_path = gcs_path[5:]
    
    parts = gcs_path.split('/', 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    
    # Read pricing file
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_text()
    pricing_df = pd.read_csv(pd.io.common.StringIO(content))
    
    # Get common CUSIPs
    common_cusips = set(pricing_df['cusip'].unique()).intersection(set(reference_df['cusip'].unique()))
    
    if not common_cusips:
        return pd.DataFrame()
    
    # Filter to common CUSIPs
    pricing_filtered = pricing_df[pricing_df['cusip'].isin(common_cusips)].copy()
    
    # Pivot to get bid and offer prices
    bid_df = pricing_filtered[pricing_filtered['trade_type'] == 'Bid Side'][['cusip', 'price']]
    offer_df = pricing_filtered[pricing_filtered['trade_type'] == 'Offered Side'][['cusip', 'price']]
    
    # Merge and calculate spreads
    spreads_df = pd.merge(
        bid_df.rename(columns={'price': 'bid_price'}),
        offer_df.rename(columns={'price': 'offer_price'}),
        on='cusip',
        how='inner'
    )
    
    spreads_df['spread'] = spreads_df['offer_price'] - spreads_df['bid_price']
    spreads_df['mid_price'] = (spreads_df['bid_price'] + spreads_df['offer_price']) / 2
    spreads_df['spread_pct'] = (spreads_df['spread'] / spreads_df['mid_price']) * 100
    
    return spreads_df

def get_last_n_business_days(n=10):
    ''' Gets the last 10 business days, including today. '''
    today = datetime.now(EASTERN).date()
    business_days = [today]  # Start with today
    current_date = today
    
    while len(business_days) < n:
        current_date = (pd.Timestamp(current_date) - BUSINESS_DAY).date()
        business_days.append(current_date)
    return business_days


def get_ig_bonds(bq_client, maturity):

    low_bound, high_bound = maturity.split('-')

    """Get IG bonds with specified maturity range"""
    query = f"""
    SELECT
      cusip,
      maturity_date,
      sp_long,
      DATE_DIFF(maturity_date, CURRENT_DATE(), DAY) / 365.25 AS years_to_maturity
    FROM
      `reference_data_v2.reference_data_flat`
    WHERE
      sp_long IN ('AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-')
      AND ref_valid_to_date > CURRENT_TIMESTAMP()
      AND maturity_date > CURRENT_DATE()
      AND DATE_DIFF(maturity_date, CURRENT_DATE(), DAY) / 365.25 BETWEEN {low_bound} AND {high_bound}
    """
    return bq_client.query(query).to_dataframe()


def ficc_prices(cusips_quantities_tradetypes):
    """
    Fetch batch pricing data for municipal bonds from FICC API.

    Args:
        cusips_quantities_tradetypes (list): List of dicts with CUSIP, Quantity, and Trade Type.

    Returns:
        list: Parsed API response for each municipal bond, excluding those with invalid prices.
    """
    import requests
    import json

    username = 'cf_test@ficc.ai'
    password = 'Luiy7864W1%'

    # username = 'eng@ficc.ai'
    # password = '1137Work!'
    base_url = 'https://api.ficc.ai/api/batchpricing'

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
    # Parse response
    raw = response.json()
    if isinstance(raw, str):
        data = json.loads(raw)  # double-encoded
    else:
        data = raw
    # Convert dict of dicts -> list of dicts
    n = len(next(iter(data.values())))
    records = [{k: data[k][str(i)] for k in data} for i in range(n)]
    return pd.DataFrame(records)
    
def get_existing_data(bq_client, business_days, maturity, quantity):
    """Get existing data for the specified business days"""

    date_list = "', '".join([d.strftime('%Y-%m-%d') for d in business_days])
    query = (
        "SELECT "
        "date_et as date, "
        "CONCAT('$', ROUND(avg_spread, 3)) as AvgSpreadDollar, "
        "CONCAT(ROUND(avg_spread_pct, 3), '%') as AvgSpreadPercent, "
        "num_cusips as NumCUSIPs "
        "FROM `eng-reactor-287421.analytics_data_source.daily_dollar_spread_averages` "
        f"WHERE date_et IN ('{date_list}') "
        "AND rating_category = 'IG' "
        f"AND maturity_range = '{maturity}' "
        f"AND quantity = {quantity} "
        "ORDER BY date DESC"
    )
    
    df = bq_client.query(query).to_dataframe()
    existing_dates = set(df['date'].tolist()) if not df.empty else set()
    
    return df, existing_dates


def process_missing_days_historical_file(bq_client, storage_client, missing_dates, maturity):
    """Process any missing days from the provided business days"""
    errors = []
    print(f"Processing {len(missing_dates)} missing days: {missing_dates}")
    
    # Get reference data only when needed
    print("Fetching IG bond reference data...")
    reference_df = get_ig_bonds(bq_client, maturity)
    print(f"Found {len(reference_df)} IG bonds with {maturity} years maturity")
    
    processed_count = 0
    for date in missing_dates:
        # Get the last file for this date
        gcs_path = get_last_file_for_date(storage_client, date)
        
        if not gcs_path:
            print(f"No files found for {date}")
            continue
        
        try:
            # Extract datetime from filename
            filename = gcs_path.split('/')[-1]
            #Note: While this date extracted from the investortools file is listed in UTC it's actually already in ET
            file_datetime = extract_datetime_from_filename(filename)
            file_datetime_et = file_datetime.replace(tzinfo=EASTERN)

            et_date = file_datetime.date()
            
            if not file_datetime:
                print(f"Could not extract datetime from {filename}")
                continue
            
            # Calculate spreads
            spreads_df = calculate_spreads_for_file(gcs_path, reference_df, storage_client)
            
            if spreads_df.empty:
                print(f"No spreads calculated for {filename}")
                continue
            
            # Calculate averages
            avg_spread = spreads_df['spread'].mean()
            avg_spread_pct = spreads_df['spread_pct'].mean()
            num_cusips = len(spreads_df)
            
            # Save to BigQuery
            results_df = pd.DataFrame([{
                'date_et': et_date,
                'datetime': file_datetime_et,
                'avg_spread': avg_spread,
                'avg_spread_pct': avg_spread_pct,
                'rating_category': 'IG',
                'maturity_range': maturity,
                'num_cusips': num_cusips,
                'source': gcs_path,
                'quantity': 1000
            }])

            # table_id = 'eng-reactor-287421.hadassah_tests.backup_hadassah_dollar_spread_averages_v3'
            table_id = 'eng-reactor-287421.analytics_data_source.daily_dollar_spread_averages'
            job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
            
            job = bq_client.load_table_from_dataframe(results_df, table_id, job_config=job_config)
            job.result()
            
            print(f"Processed {date}: {num_cusips} CUSIPs, avg spread ${avg_spread:.3f}")
            processed_count += 1
            
        except Exception as e:
            errors.append((date, str(e)))
            continue
    if errors:
        msg = "; ".join([f"{d}: {m}" for d,m in errors])
        raise RuntimeError(f"Historical backfill had errors: {msg}")
    return processed_count

def process_missing_days_batch_pricing(bq_client, quantity, source, maturity):
    from pytz import utc

    # Get reference data only when needed
    print("Fetching IG bond reference data...")
    reference_df = get_ig_bonds(bq_client, maturity)
    print(f"Found {len(reference_df)} IG bonds with {maturity} years maturity")

    # Keep consistent with UTC zone from investortools files
    eastern_now = datetime.now(EASTERN) 
    utc_now = eastern_now.astimezone(utc) 

    # Drop fractional seconds
    eastern_now = eastern_now.replace(microsecond=0)
    date_et = eastern_now.date()

    processed_count = 0

    # Construct the pricing request
    cusips = reference_df['cusip'].tolist()
    payload_bid = [{'CUSIP': cusip, 'Quantity': quantity, 'Trade Type': 'P'} for cusip in cusips]
    payload_ask = [{'CUSIP': cusip, 'Quantity': quantity, 'Trade Type': 'S'} for cusip in cusips]
    # Call FICC API
    bid_df = ficc_prices(payload_bid)
    ask_df = ficc_prices(payload_ask)
    merged = pd.merge(bid_df, ask_df, on='cusip', suffixes=('_bid', '_ask'))
    # Compute spread
    merged['spread'] = merged['price_ask'] - merged['price_bid']
    merged['spread_pct'] = (merged['spread'] / merged['price_bid']) * 100
    # Compute averages
    avg_spread = merged['spread'].mean()
    avg_spread_pct = merged['spread_pct'].mean()
    num_cusips = len(merged)

    # Save to BigQuery
    results_df = pd.DataFrame([{
        'date_et': date_et,
        'datetime': eastern_now,
        'avg_spread': avg_spread,
        'avg_spread_pct': avg_spread_pct,
        'rating_category': 'IG',
        'maturity_range': maturity,
        'num_cusips': num_cusips,
        'source': source,
        'quantity': quantity
    }])

    # table_id = 'eng-reactor-287421.hadassah_tests.backup_hadassah_dollar_spread_averages_v3'
    table_id = 'eng-reactor-287421.analytics_data_source.daily_dollar_spread_averages'
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
    
    job = bq_client.load_table_from_dataframe(results_df, table_id, job_config=job_config)
    job.result()
    
    print(f"Processed {eastern_now}: {num_cusips} CUSIPs, avg spread ${avg_spread:.3f}")
    processed_count += 1
    
    return processed_count

def get_spread_data():
    bq_client = bigquery.Client(project='eng-reactor-287421')
    storage_client = storage.Client(project='eng-reactor-287421')

    business_days = get_last_n_business_days(10)

    maturities = ['4.5-5.5', '9.5-10.5']
    quantities = [1000, 100]

    last_df = pd.DataFrame()
    today_et = datetime.now(EASTERN).date()

    # Calculate spreads for each maturity, at each quantity
    for maturity in maturities:
        for quantity in quantities:

            # Only quantity 1000 spreads can be calculated using the investortools files in GCP
            if quantity == 1000:
                source = 'gcp_file'
                print(f"Processing maturity={maturity}, qty={quantity}, source={source}")

                existing_df, existing_dates = get_existing_data(bq_client, business_days, maturity, quantity)
                last_df = existing_df  # keep a snapshot even if we don't add rows

                missing_dates = [d for d in business_days if d not in existing_dates]
                if not missing_dates:
                    print(f"All days already processed for 1000; maturity {maturity} skipping backfill.")
                    continue

                processed = process_missing_days_historical_file(
                    bq_client, storage_client, missing_dates, maturity
                )
                
                if processed > 0:
                    print(f"Added {processed} new days (1000). Refreshing snapshot…")
                    last_df, _ = get_existing_data(bq_client, business_days, maturity, quantity)

            else:
                source = 'ficc_batch_pricing'
                print(f"Processing maturity={maturity}, qty={quantity}, source={source}")

                # For non-1000, only check/return the most recent business day
                existing_df, existing_dates = get_existing_data(bq_client, [business_days[0]], maturity, quantity)
                last_df = existing_df

                # We are missing a date if the most recent business day is not in the table
                missing_dates = (business_days[0] not in existing_dates)

                if not missing_dates:
                    print(f"All days already processed for {quantity}; maturity {maturity} skipping backfill.")
                    continue

                processed = process_missing_days_batch_pricing(
                    bq_client, quantity, source, maturity
                )

                if processed > 0:
                    print(f"Inserted today for non-1000. Refreshing today snapshot…")
                    last_df, _ = get_existing_data(bq_client, [business_days[0]], maturity, quantity)

    return last_df


if __name__ == '__main__':
    get_spread_data()