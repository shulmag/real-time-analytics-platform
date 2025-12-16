'''
'''
import os
import redis
from pytz import timezone

from google.cloud import storage


USERNAME = 'investortools_cf@ficc.ai'
PASSWORD = 'investortools'

EASTERN = timezone('US/Eastern')

API_URL = 'https://server-investortools-3ukzrmokpq-uc.a.run.app'

QUANTITY = 500
TRADE_TYPE = 'S'
# Upper and lower bounds for what the user may enter for the quantity; taken directly from `app_engine/demo/server/modules/auxiliary_variables.py`
QUANTITY_LOWER_BOUND = 5    # in thousands
QUANTITY_UPPER_BOUND = 10000    # in thousands

UNIQUE_QUANTITIES_FOR_INVESTOR_TOOLS = [1000]
UNIQUE_TRADE_TYPES_FOR_INVESTOR_TOOLS = ['S', 'P']

COLUMNS_TO_KEEP = ['cusip', 'trade_type', 'quantity', 'price', 'ytw', 'yield_to_worst_date']

MAX_NUMBER_OF_CUSIPS_PER_BATCH = 2000    # chose this value to match `LARGE_BATCH_SIZE` in `app_engine/demo/server/modules/finance.py` so to not have multiple nested API calls

MULTIPROCESSING = True

INPUT_CSV_FILENAME = 'files/100k.csv'
GOOGLE_CLOUD_BUCKET = 'large_batch_pricing'
LOCAL_NOT_OUTSTANDING_CUSIPS_FILENAME = 'not_outstanding_CUSIPs.txt'
NOT_OUTSTANDING_CUSIPS_PICKLE_FILENAME = 'not_outstanding_CUSIPs.pkl'

REFERENCE_DATA_REDIS_CLIENT = redis.Redis(host='10.108.4.36', port=6379, db=0)    # use read endpoint since use case is read-only allowing for lower latency and to not accidentally corrupt the redis by attempting to write to it

TESTING = False

if TESTING: os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/user/ficc/mitas_creds.json'
STORAGE_CLIENT = storage.Client()    # this needs to be done after the credentials are created

# SFTP credentials
SFTP_HOST = 'sftp.ficc.ai'
SFTP_USERNAME = 'investortools'
SFTP_PASSWORD = 'ipts0520'
