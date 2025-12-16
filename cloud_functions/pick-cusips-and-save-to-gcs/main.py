'''
'''

#This cloud function will be periodicalled called from the cloud scheduler to pick various cusips of different categories from bq and save them in the bucket "cusip_pool_for_testing".
#They will be saved in the form of csv's with titles showing what kind of cusips they are. More queries will be added to this function that handles more criteria. 

from google.cloud import bigquery, storage
import pandas as pd
import os

def pick_cusips_and_save_to_gcs(request):
    try:
        bq_client = bigquery.Client()
        storage_client = storage.Client()

        short_maturity_query = """
            SELECT DISTINCT cusip,
                   DATE_DIFF(maturity_date, trade_date, DAY) AS diff,
                   trade_date
            FROM auxiliary_views_v2.trade_history_same_issue_5_yr_mat_bucket_1_materialized
            WHERE DATE_DIFF(maturity_date, trade_date, DAY) > 40
              AND DATE_DIFF(maturity_date, trade_date, DAY) < 60
              AND outstanding_amount > 500000
              AND trade_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
                                 AND DATE_SUB(CURRENT_DATE(), INTERVAL 0 DAY)
            ORDER BY trade_date DESC, diff DESC
            LIMIT 3
        """
        #The above query returns 3 cusips that had a maturity date of 60 days in the past week. So at the time this run they would have a maturity of 53-60 days.
        short_maturity_df = bq_client.query(short_maturity_query).to_dataframe()

        if short_maturity_df.empty:
            return jsonify({"status": "WARNING", "message": "No CUSIPs found."}), 200

        local_path = "/tmp/cusips.csv"
        short_maturity_df.to_csv(local_path, index=False)

        bucket_name = "cusip_pool_for_testing"
        blob = storage_client.bucket(bucket_name).blob("short_maturity.csv")
        blob.upload_from_filename(local_path)

        os.remove(local_path)

        return (f"Short maturity cusips successfully uploaded to bucket {bucket-name}", 200)

    except Exception as e:
        return (f"ERROR: {str(e)}", 500)
