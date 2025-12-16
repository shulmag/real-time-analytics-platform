'''
'''
from pytz import timezone


USERNAME = None    # TODO: fill in
PASSWORD = None    # TODO: fill in

CSV_FILEPATH = None    # TODO: fill in

EASTERN = timezone('US/Eastern')

API_URL_LIST = ['https://api.ficc.ai']
NUM_SERVERS = len(API_URL_LIST)

UNIQUE_QUANTITIES = [50, 100, 500, 1000]
UNIQUE_TRADE_TYPES = ['P']

COLUMNS_TO_KEEP = ['cusip', 'trade_type', 'quantity', 'price', 'ytw', 'yield_to_worst_date']

MAX_NUMBER_OF_CUSIPS_PER_BATCH = 2000
MAX_ASYNC_CALLS_PER_SERVER = 200

PRINT_RETRY_MESSAGES = False
