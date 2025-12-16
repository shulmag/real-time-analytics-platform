'''
'''


def picking_and_saving_cusips_to_gcs(request):
    from google.cloud import bigquery, storage
    import pandas as pd

    bq_client = bigquery.Client()
    storage_client = storage.Client()

    query = """
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
    df = bq_client.query(query).to_dataframe()

    local_path = "/tmp/cusips.csv"
    df.to_csv(local_path, index=False)

    bucket_name = "cusip_pool_for_testing"
    blob = storage_client.bucket(bucket_name).blob("cusips.csv")
    blob.upload_from_filename(local_path)

    return "Uploaded to GCS"
