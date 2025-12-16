# -*- coding: utf-8 -*-

from google.cloud import bigquery
import os
import pandas as pd
import pickle
import redis
import smtplib, ssl
from email.mime.text import MIMEText
from google.cloud import secretmanager

bq_client = bigquery.Client()
job_config = bigquery.job.QueryJobConfig(allow_large_results=True)
redis_host = os.environ.get('REDISHOST', '10.122.9.59')
redis_port = int(os.environ.get('REDISPORT', 6379))
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)

# The below query uses windowing to select for each cusip only the reference data and trade data for the most recent trade.
query = "select * from ( select cusip as c, *, row_number() over(partition by cusip order by publish_datetime desc) seq FROM `eng-reactor-287421.auxiliary_views.trade_history_with_ref_today` ) d where seq = 1"


def sqltodf(sql,limit = ""):
    if limit != "": 
        limit = f" WHERE trade_date < DATE({limit})"
    bqr = bq_client.query(sql + limit).result()
    return bqr.to_dataframe()


def upload_data_to_redis(key, value):
    value = pickle.dumps(value,protocol=pickle.HIGHEST_PROTOCOL)
    redis_client.set(key, value)

def main(args):
    df = sqltodf(query)

    for _,row in df.iterrows():
        key = row['cusip']
        upload_data_to_redis(key,row)

