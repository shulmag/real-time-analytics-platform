import os
import redis
from google.cloud import bigquery
from data.process_data import process_data
import pickle5 as pickle
from datetime import datetime, timedelta
import pandas as pd
from datetime import datetime

redis_client = redis.Redis(host='10.54.92.117', port=6379, db=0)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/shayaan/ahmad_creds.json'

bq_client = bigquery.Client()


SEQUENCE_LENGTH = 5
NUM_FEATURES = 6

def main():
    data = pickle.loads(redis_client.get('64971XQM3')).to_frame().T
    data['settlement_date'] = pd.to_datetime(datetime.now() + timedelta(days=2), format='%Y-%m-%d')
    data['trade_date'] = pd.to_datetime(datetime.now(), format='%Y-%m-%d')
    data['par_traded'] = 100000
    data['trade_type'] = 'D'
    data['trade_datetime'] = pd.to_datetime(datetime.now(), format='%Y-%m-%d %H:%M:%S')
    data['transaction_type'] = 'I'
    data = process_data(data,
                    pd.to_datetime(datetime.now(), format='%Y-%m-%d %H:%M'),
                    bq_client,
                    SEQUENCE_LENGTH,
                    NUM_FEATURES,
                    'FICC',
                    remove_short_maturity=True,
                    min_trades_in_history=0,
                    process_ratings=False)
    print(data)


if __name__ == "__main__":
    main()