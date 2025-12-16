"""
Investment Grade Bond Spread Analyzer - Simplified Version
Tracks daily spreads for IG bonds with 4.5-5.5 years maturity
"""
import pandas as pd
from google.cloud import storage, bigquery
import os
from datetime import datetime, timedelta
import re
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
from pytz import timezone

# Set credentials
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'

# Calendar and timezone constants
class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    """Custom US Federal Holiday calendar that includes Good Friday"""
    rules = USFederalHolidayCalendar.rules + [GoodFriday]

BUSINESS_DAY = CustomBusinessDay(calendar=USHolidayCalendarWithGoodFriday())
EASTERN = timezone('US/Eastern')


def get_last_n_business_days(n=10):
    """Get the last n business days from today"""
    today = datetime.now(EASTERN).date()
    business_days = []
    current_date = today
    
    while len(business_days) < n:
        current_date = (pd.Timestamp(current_date) - BUSINESS_DAY).date()
        business_days.append(current_date)
    
    return business_days


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


def get_ig_bonds(bq_client):
    """Get IG bonds with 4.5-5.5 years maturity"""
    query = """
    SELECT
      cusip,
      maturity_date,
      sp_long,
      DATE_DIFF(maturity_date, CURRENT_DATE(), DAY) / 365.25 AS years_to_maturity
    FROM
      `reference_data_v2.reference_data_flat`
    WHERE
      sp_long IN ('AAA', 'AA', 'A', 'BBB')
      AND ref_valid_to_date > CURRENT_TIMESTAMP()
      AND maturity_date > CURRENT_DATE()
      AND DATE_DIFF(maturity_date, CURRENT_DATE(), DAY) / 365.25 BETWEEN 4.5 AND 5.5
    """
    return bq_client.query(query).to_dataframe()


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


def get_existing_data(bq_client, business_days):
    """Get existing data for the specified business days"""
    date_list = "', '".join([d.strftime('%Y-%m-%d') for d in business_days])
    
    query = (
        "SELECT "
        "file_datetime as DateTime, "
        "DATE(file_datetime) as date, "
        "CONCAT('$', ROUND(avg_spread, 3)) as AvgSpreadDollar, "
        "CONCAT(ROUND(avg_spread_pct, 3), '%') as AvgSpreadPercent, "
        "num_cusips as NumCUSIPs, "
        "REGEXP_EXTRACT(source_file, r'[^/]+$') as SourceFile "
        "FROM `eng-reactor-287421.spreads.dollar_spread_averages` "
        f"WHERE DATE(file_datetime) IN ('{date_list}') "
        "AND rating_category = 'IG' "
        "AND maturity_range = '4.5-5.5' "
        "ORDER BY file_datetime DESC"
    )
    
    df = bq_client.query(query).to_dataframe()
    existing_dates = set(df['date'].tolist()) if not df.empty else set()
    
    # Drop the date column as it's not needed in the final output
    if not df.empty:
        df = df.drop('date', axis=1)
    
    return df, existing_dates


def process_missing_days(bq_client, storage_client, business_days, existing_dates):
    """Process any missing days from the provided business days"""
    # Find missing dates
    missing_dates = [d for d in business_days if d not in existing_dates]
    
    if not missing_dates:
        print("All days already processed")
        return 0
    
    print(f"Processing {len(missing_dates)} missing days: {missing_dates}")
    
    # Get reference data only when needed
    print("Fetching IG bond reference data...")
    reference_df = get_ig_bonds(bq_client)
    print(f"Found {len(reference_df)} IG bonds with 4.5-5.5 years maturity")
    
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
            file_datetime = extract_datetime_from_filename(filename)
            
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
                'file_datetime': file_datetime,
                'avg_spread': avg_spread,
                'avg_spread_pct': avg_spread_pct,
                'rating_category': 'IG',
                'maturity_range': '4.5-5.5',
                'num_cusips': num_cusips,
                'source_file': gcs_path
            }])
            
            table_id = 'eng-reactor-287421.spreads.dollar_spread_averages'
            job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
            
            job = bq_client.load_table_from_dataframe(results_df, table_id, job_config=job_config)
            job.result()
            
            print(f"Processed {date}: {num_cusips} CUSIPs, avg spread ${avg_spread:.3f}")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {date}: {str(e)}")
            continue
    
    return processed_count


def get_spread_data():
    """Main function to get spread data - ensures last 10 business days are populated"""
    # Initialize clients
    bq_client = bigquery.Client(project='eng-reactor-287421')
    storage_client = storage.Client(project='eng-reactor-287421')
    
    # Get last 10 business days
    business_days = get_last_n_business_days(30)
    
    # Get existing data and check which dates we have
    existing_df, existing_dates = get_existing_data(bq_client, business_days)
    
    # Find missing dates
    missing_dates = [d for d in business_days if d not in existing_dates]
    
    # If no missing dates, return what we have
    if not missing_dates:
        print("All days already processed")
        return existing_df
    
    # Process missing days
    processed = process_missing_days(bq_client, storage_client, business_days, existing_dates)
    
    if processed > 0:
        print(f"Added {processed} new days to the database")
        # Re-query only if we added new data
        existing_df, _ = get_existing_data(bq_client, business_days)
    
    return existing_df


if __name__ == "__main__":
    # Test the function
    df = get_spread_data()
    print("\nCurrent spread data:")
    print(df.to_markdown())