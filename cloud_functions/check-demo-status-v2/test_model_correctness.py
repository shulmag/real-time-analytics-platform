'''

Description: checking accuracy in average bid-ask spread per file in Google Bucket 'large_batch_pricing'

The idea: For every file that is added, it calculates the average spread and compares that to the historical averages using Z-score.
          If the Z-score is less than or equal to the threshold, which is calculated as 3 standard deviations away from the mean, then the average spread is accurate and stored inside BigQuery 'yoshi_test_avg_spread'. 
          If otherwise, its considered wide, and sends an alert message
          
          NOTE: get_last_file_for_date(), and get_ig_bonds() are exactly copied from https://github.com/Ficc-ai/ficc/blob/dev/analytics/cloud_function/spreads.py
'''

from datetime import datetime
from io import TextIOWrapper
from auxiliary_variables import REQUEST_URL, USERNAME, PASSWORD
from google.cloud import bigquery, storage
import pandas as pd
import numpy as np

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/tsuyoshikameda/Git/ficc/cloud_functions/daily-muni-market-recap/creds.json'

'''
Adds files to a csv file. Useful when testing
'''
# def get_all_files(storage_client, date):
#     date_str = date.strftime('%Y-%m-%d')
#     bucket = storage_client.bucket('large_batch_pricing')
#     prefix = f"{date_str}/"
    
#     # List all files for this date
#     blobs = list(bucket.list_blobs(prefix=prefix))
#     csv_files = [b for b in blobs if b.name.endswith('.csv') and 'priced_' in b.name]
#     return csv_files

# used to calculate avg spread of each file
# def calculate_avg_spread():
#     storage_client = storage.Client()
#     date_str = "2025-08-11"
#     date = datetime.strptime(date_str, "%Y-%m-%d").date()      
#     gcs_path = get_all_files(storage_client, date)
#     bq_client = bigquery.Client(project='eng-reactor-287421')
#     reference_df = get_ig_bonds(bq_client)

#     avg_list = []
#     for path in gcs_path:
#         url = f"gs://large_batch_pricing/{path.name}"
#         response = calculate_spreads_for_file(url, reference_df, storage_client)
#         avg_list.append({'date': date, 'file': path.name, 'avg_spread': response['spread'].mean()})

#     df = pd.DataFrame(avg_list)
#     csv_file_path = 'avg_spreads_history.csv'
#     df.to_csv(csv_file_path, mode='a', header=not pd.io.common.file_exists(csv_file_path), index=False, sep='\t')

def calculate_spreads_for_file(gcs_path, reference_df, storage_client, chunksize=100_000):

    # Normalize gs://bucket/path
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]
    bucket_name, blob_name = gcs_path.split("/", 1)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Only the columns we actually need
    usecols = ["cusip", "trade_type", "price"]
    dtype   = {"cusip": "string", "trade_type": "category", "price": "float32"}

    ig = set(reference_df["cusip"].astype(str))

    pending = {}

    # just collect spreads, then build a tiny DF at the end
    spreads = []

    with blob.open("rb") as fh:
        text_fh = TextIOWrapper(fh, encoding="utf-8")
        for chunk in pd.read_csv(text_fh, usecols=usecols, dtype=dtype, chunksize=chunksize):
            chunk = chunk[chunk["cusip"].isin(ig)]
            if chunk.empty:
                continue

            tt = chunk["trade_type"].astype(str).str.upper()

            bid_rows = chunk[tt.isin(["P", "BID SIDE", "BID"])]
            off_rows = chunk[tt.isin(["S", "OFFERED SIDE", "OFFER"])]

            # Consume bids
            for cusip, price in zip(bid_rows["cusip"], bid_rows["price"]):
                c = str(cusip)
                d = pending.setdefault(c, {"bid": None, "offer": None})
                d["bid"] = float(price)
                if d["offer"] is not None:
                    spreads.append(d["offer"] - d["bid"])
                    pending.pop(c, None)

            # Consume offers
            for cusip, price in zip(off_rows["cusip"], off_rows["price"]):
                c = str(cusip)
                d = pending.setdefault(c, {"bid": None, "offer": None})
                d["offer"] = float(price)
                if d["bid"] is not None:
                    spreads.append(d["offer"] - d["bid"])
                    pending.pop(c, None)

    # Return a tiny DF compatible with your existing caller
    if not spreads:
        return pd.DataFrame(columns=["spread"], dtype="float32")

    print(len(spreads))
    return pd.DataFrame({"spread": np.array(spreads, dtype=np.float32)})

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

def load_history_from_bq(bq_client: bigquery.Client) -> pd.DataFrame:
    query = f"""
        SELECT date, file, avg_spread
        FROM `eng-reactor-287421.yoshi_test_avg_spread.avg_spread_history`
    """
    return bq_client.query(query).result().to_dataframe(create_bqstorage_client=True)

def already_in_bq(bq_client: bigquery.Client, date_obj, gcs_path: str) -> bool:
    query = """
        SELECT 1
        FROM `eng-reactor-287421.yoshi_test_avg_spread.avg_spread_history`
        WHERE date = @date
          AND file = @file
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("date", "DATE", date_obj),
            bigquery.ScalarQueryParameter("file", "STRING", gcs_path.replace("gs://large_batch_pricing/", "")),
        ]
    )
    result = bq_client.query(query, job_config=job_config).result()
    return result.total_rows > 0

def add_today_bq(bq_client: bigquery.Client, date_obj, gcs_path: str, avg_spread_today: float) -> None:
    table_id = "eng-reactor-287421.yoshi_test_avg_spread.avg_spread_history"
    row = {
        "date": date_obj.isoformat(),
        "file": gcs_path.replace("gs://large_batch_pricing/", ""),
        "avg_spread": float(avg_spread_today),
    }
    errors = bq_client.insert_rows_json(table_id, [row])
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")

def compare_avg_spread():
    storage_client = storage.Client()
    date = datetime.today().date()
    gcs_path = get_last_file_for_date(storage_client, date)
    if not gcs_path:
        print(f"No file found for date {date:%Y-%m-%d}")
        return "none"
    
    print("Reading from: ", gcs_path)
    bq_client = bigquery.Client(project='eng-reactor-287421')
    if already_in_bq(bq_client, date, gcs_path):
        print(f"File {gcs_path} already processed for {date}; skipping.")
        return
    reference_df = get_ig_bonds(bq_client)

    spreads_df = calculate_spreads_for_file(gcs_path, reference_df, storage_client)
    avg_spread_today = spreads_df['spread'].mean()

    historical_data = load_history_from_bq(bq_client)
    avg_spread_history = historical_data['avg_spread'].mean()
    std_spread_history = historical_data['avg_spread'].std()
    print(f"Average spread for today ({date}): ", avg_spread_today)
    print("Historical average spread: ", avg_spread_history)
    print("Historical standard deviation of the spreads: ", std_spread_history)
    z_score = (avg_spread_today - avg_spread_history) / std_spread_history
    print(f"Z-score: ", z_score)
    threshold = avg_spread_history + (3 * std_spread_history)
    print(f"Threshold: ", threshold)

    if(z_score <= threshold) :
        print("Since z-score <= threshold, this spread is valid")
        print("Adding to history...")
        add_today_bq(bq_client, date, gcs_path, avg_spread_today)
        print(f"Done!")

    else : 
        raise AssertionError(
            "Error: avgerage spread is too wide"
            f"date={date}, file={gcs_path}, avg_today={avg_spread_today:.6f}, mean={avg_spread_history:.6f}, standard deviation={std_spread_history:.6f}"
        )

# ----- For Testing ----
# if __name__ == "__main__":
    # test_spread()
    # check_wide_spread()
    # calculate_avg_spread()
    # compare_avg_spread()