'''
'''
from pytz import timezone


USERNAME = ''    # TODO: fill in
PASSWORD = ''    # TODO: fill in

CSV_FILEPATH = ''    # TODO: fill in

EASTERN = timezone('US/Eastern')

API_URL = 'https://api.ficc.ai'

UNIQUE_QUANTITIES = [50, 100, 500, 1000]
UNIQUE_TRADE_TYPES = ['P']

COLUMNS_TO_KEEP = ['cusip', 'trade_type', 'quantity', 'price', 'ytw', 'yield_to_worst_date', 'user_price', 'bid_ask_price_delta', 'compliance_rating', 'trade_datetime']

MAX_NUMBER_OF_CUSIPS_PER_BATCH = 2000
