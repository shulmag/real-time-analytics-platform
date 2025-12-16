'''
Description: General purpose variables used through the server code.
'''
from collections import defaultdict

from datetime import datetime, timedelta
import redis
from pytz import timezone

from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday

from google.cloud import bigquery, storage    # , logging
from google.auth.exceptions import DefaultCredentialsError

from modules.ficc.utils.calendars import get_all_trading_holidays
from modules.ficc.utils.gcp_storage_functions import download_pickle_file
from modules.ficc.utils.auxiliary_variables import CATEGORICAL_FEATURES, CATEGORICAL_FEATURES_DOLLAR_PRICE
from modules.ficc.utils.auxiliary_functions import run_multiple_times_before_failing


# # removed the logging client since this is a potential suspect for latency during container initialization
# # set up logging client; https://cloud.google.com/logging/docs/setup/python
# logging_client = logging.Client()
# logging_client.setup_logging()

API_URL = 'https://server-batch-pricing-and-compliance-964018767272.us-central1.run.app'    # http://localhost:5001
SERVER_URL_MAP = defaultdict(lambda: [API_URL])
SERVER_URL_MAP.update({'investortools': ['https://server-investortools-964018767272.us-central1.run.app'], 
                       'vanguard': ['https://server-vanguard-0-964018767272.us-central1.run.app', 'https://server-vanguard-1-964018767272.us-central1.run.app', 'https://server-vanguard-2-964018767272.us-central1.run.app']})

USE_SIMILAR_TRADES_MODEL_FOR_YIELD_SPREAD_PREDICTIONS_FOR_POINT_IN_TIME_PRICING = True    # for 50k unique CUSIPs and 4 different quantities (200k line items), with 8 CPUs, pricing without similar trades takes approximately 1.75 hours and requires 100 GB of memory, whereas pricing with similar trades takes approximately 3 hours and requires 120 GB of memory

PROJECT_ID = 'eng-reactor-287421'
AUXILIARY_VIEWS_DATASET = f'{PROJECT_ID}.auxiliary_views_v2'

DEFAULT_QUANTITY = 500    # NOTE: this needs to match `defaultQuantity` in `src/components/pricing.jsx`
DEFAULT_TRADE_TYPE = 'S'    # NOTE: this needs to match `defaultTradeType` in `src/components/pricing.jsx`


@run_multiple_times_before_failing((DefaultCredentialsError,))    # transient error that occurs during times of high volume usage
def get_bq_client():
    return bigquery.Client()
bq_client = get_bq_client()


@run_multiple_times_before_failing((DefaultCredentialsError,))    # transient error that occurs during times of high volume usage
def get_storage_client():
    return storage.Client()
storage_client = get_storage_client()


REFERENCE_DATA_REDIS_CLIENT = redis.Redis(host='10.108.4.36', port=6379, db=0)    # use read endpoint since use case is read-only allowing for lower latency and to not accidentally corrupt the redis by attempting to write to it
TRADE_HISTORY_REDIS_CLIENT = redis.Redis(host='10.75.46.229', port=6379, db=0)    # use read endpoint since use case is read-only allowing for lower latency and to not accidentally corrupt the redis by attempting to write to it
SIMILAR_TRADE_HISTORY_REDIS_CLIENT = redis.Redis(host='10.117.191.181', port=6379, db=0)    # use read endpoint since use case is read-only allowing for lower latency and to not accidentally corrupt the redis by attempting to write to it

get_holidays = lambda: get_all_trading_holidays()
get_treasury_rate_df = lambda: download_pickle_file(storage_client, 'treasury_rate_df', 'treasury_rate_df.pkl')    # create function instead of loading as a global variable so we can get the most updated value every time we price a CUSIP; updated daily by the cloud function `daily_treasury_yield`
SEQUENCE_LENGTH = 5    # number of trades in the same CUSIP trade history for yield spread model
SEQUENCE_LENGTH_DOLLAR_PRICE = 2    # number of trades in the same CUSIP trade history for dollar price model
MAX_NUM_TRADES_IN_HISTORY_TO_DISPLAY_ON_UI = 32    # maximum number of trades that are displayed on the frontend (which is assumed to be much larger than the number of trades used in the model); if this number is very large, there could be CUSIPs with a large number of recent trades that would take a long time to process and it would look like the individual pricing component is slow, but it is really just processing these trades to dispaly on the frontend
NUM_FEATURES = 6    # number of features per trade in the history

EASTERN = timezone('US/Eastern')    # all datetime objects must use ET time zone

# Variables for date functions
ONE_DAY = timedelta(days=1)
TWO_DAYS = timedelta(days=2)


class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]
current_year = datetime.now(tz=EASTERN).year
holidays_in_last_year_and_next_year = set(USHolidayCalendarWithGoodFriday().holidays(start=f'{current_year - 1}-01-01',end=f'{current_year + 1}-12-31'))


# Variables for string formatting of datetime objects
YEAR_MONTH_DAY = '%Y-%m-%d'
HOUR_MIN = '%H:%M'
HOUR_MIN_SEC = '%H:%M:%S'
MONTH_DAY_YEAR = '%m-%d-%Y'
YEAR_MONTH_DAY_HOUR_MIN_SEC = YEAR_MONTH_DAY + ' ' + HOUR_MIN_SEC
datetime_display_format = MONTH_DAY_YEAR + ' ' + HOUR_MIN_SEC

DATE_FROM_WHICH_ALL_PAST_TRADES_ARE_STORED_IN_REDIS = datetime(2024, 6, 27)    # date at which the following Jira task was resolved: https://ficcai.atlassian.net/browse/FA-2107
DATE_FROM_WHICH_PAST_REFERENCE_DATA_IS_STORED_IN_REDIS = datetime(2024, 10, 2)    # date noted in the following Jira task: https://ficcai.atlassian.net/browse/FA-1250
DATE_FROM_WHICH_MODELS_TRAINED_WITH_NEW_REFERENCE_DATA_IS_IN_PRODUCTION = datetime(2024, 12, 29)    # use v2 models after this date; date of the most recent comment in the following Jira task: https://ficcai.atlassian.net/browse/FA-2534

NUM_CHARS_IN_YEAR_MONTH_DAY = 10    # 4 characters for the year, 2 characters for the month, 2 characters for the day, and two additional characters for the separators

# Predictors list for the neural network
BASE_PREDS = ['trade_history_input', 'target_attention_input', 'NON_CAT_AND_BINARY_FEATURES']
NN_PRED_LIST = ['similar_trade_history_input'] + BASE_PREDS + CATEGORICAL_FEATURES
NN_PRED_LIST_DOLLAR_PRICE = BASE_PREDS + CATEGORICAL_FEATURES_DOLLAR_PRICE
TRADE_MAPPING = {'D': [0, 0], 'S': [0, 1], 'P': [1, 0]}
NUMERICAL_ERROR = -1    # value used to fill in a numerical field when there is an error
DOLLAR_PRICE_MODEL_DISPLAY_TEXT = {'missing_or_negative_yields': 'We do not provide an evaluated yield since previous MSRB reported yields for this CUSIP are missing or negative.', 
                                   'adjustable_rate_coupon': 'For adjustable rate coupon, we do not yet display yield. Yield to conversion date coming soon!', 
                                   'maturing_soon': 'CUSIP is maturing very soon or has already matured so we only provide a dollar price.', 
                                   'defaulted': 'CUSIP has defaulted so we only provide a dollar price.', 
                                   'high_yield_in_history': 'MSRB reported yields for this CUSIP are abnormally high (greater than 10%), so we only provide a dollar price.'}    # must have the same key and second item in value as `dollarPriceModelDisplayText` in `src/components/pricing.jsx`

# Precision of price and yield values
DISPLAY_PRECISION = 3
LOGGING_PRECISION = 5

# Used to create the scheme for logging
LOGGING_FEATURES = {'user': 'string', 
                    'api_call': 'BOOLEAN', 
                    'time': 'timestamp', 
                    'cusip': 'string', 
                    'direction': 'string', 
                    'quantity': 'integer', 
                    'ficc_price': 'numeric', 
                    'ficc_ytw': 'numeric', 
                    'yield_spread': 'numeric', 
                    'ficc_ycl': 'numeric', 
                    'calc_date': 'string', 
                    'daily_schoonover_report': 'BOOLEAN', 
                    'real_time_yield_curve': 'BOOLEAN', 
                    'batch': 'BOOLEAN', 
                    'show_similar_bonds': 'BOOLEAN', 
                    'error': 'BOOLEAN', 
                    'model_used': 'string'}
RECENT_FEATURES = ['yield_spread', 'yield_spread2', 'yield_spread3', 'yield_spread4', 'yield_spread5']

REFERENCE_DATA_FEATURES = ['coupon',
                           'cusip',
                           'ref_valid_from_date',
                           'ref_valid_to_date',
                           'incorporated_state_code',
                           'organization_primary_name',
                           'instrument_primary_name',
                           'issue_key',
                           'issue_text',
                           'conduit_obligor_name',
                           'is_called',
                           'is_callable',
                           'is_escrowed_or_pre_refunded',
                           'first_call_date',
                           'call_date_notice',
                           'callable_at_cav',
                           'par_price',
                           'call_defeased',
                           'call_timing',
                           'call_timing_in_part',
                           'extraordinary_make_whole_call',
                           'extraordinary_redemption',
                           'make_whole_call',
                           'next_call_date',
                           'next_call_price',
                           'call_redemption_id',
                           'first_optional_redemption_code',
                           'second_optional_redemption_code',
                           'third_optional_redemption_code',
                           'first_mandatory_redemption_code',
                           'second_mandatory_redemption_code',
                           'third_mandatory_redemption_code',
                           'par_call_date',
                           'par_call_price',
                           'maximum_call_notice_period',
                           'called_redemption_type',
                           'muni_issue_type',
                           'refund_date',
                           'refund_price',
                           'redemption_cav_flag',
                           'max_notification_days',
                           'min_notification_days',
                           'next_put_date',
                           'put_end_date',
                           'put_feature_price',
                           'put_frequency',
                           'put_start_date',
                           'put_type',
                           'maturity_date',
                           'sp_long',
                           'sp_stand_alone',
                           'sp_icr_school',
                           'sp_prelim_long',
                           'sp_outlook_long',
                           'sp_watch_long',
                           'sp_Short_Rating',
                           'sp_Credit_Watch_Short_Rating',
                           'sp_Recovery_Long_Rating',
                           'moodys_long',
                           'moodys_short',
                           'moodys_Issue_Long_Rating',
                           'moodys_Issue_Short_Rating',
                           'moodys_Credit_Watch_Long_Rating',
                           'moodys_Credit_Watch_Short_Rating',
                           'moodys_Enhanced_Long_Rating',
                           'moodys_Enhanced_Short_Rating',
                           'moodys_Credit_Watch_Long_Outlook_Rating',
                           'has_sink_schedule',
                           'next_sink_date',
                           'sink_indicator',
                           'sink_amount_type_text',
                           'sink_amount_type_type',
                           'sink_frequency',
                           'sink_defeased',
                           'additional_next_sink_date',
                           'sink_amount_type',
                           'additional_sink_frequency',
                           'min_amount_outstanding',
                           'max_amount_outstanding',
                           'default_exists',
                           'has_unexpired_lines_of_credit',
                           'years_to_loc_expiration',
                           'escrow_exists',
                           'escrow_obligation_percent',
                           'escrow_obligation_agent',
                           'escrow_obligation_type',
                           'child_linkage_exists',
                           'put_exists',
                           'floating_rate_exists',
                           'bond_insurance_exists',
                           'is_general_obligation',
                           'has_zero_coupons',
                           'delivery_date',
                           'issue_price',
                           'primary_market_settlement_date',
                           'issue_date',
                           'outstanding_indicator',
                           'federal_tax_status',
                           'maturity_amount',
                           'available_denom',
                           'denom_increment_amount',
                           'min_denom_amount',
                           'accrual_date',
                           'bond_insurance',
                           'coupon_type',
                           'current_coupon_rate',
                           'daycount_basis_type',
                           'debt_type',
                           'default_indicator',
                           'first_coupon_date',
                           'interest_payment_frequency',
                           'issue_amount',
                           'last_period_accrues_from_date',
                           'next_coupon_payment_date',
                           'odd_first_coupon_date',
                           'orig_principal_amount',
                           'original_yield',
                           'outstanding_amount',
                           'previous_coupon_payment_date',
                           'sale_type',
                           'settlement_type',
                           'additional_project_txt',
                           'asset_claim_code',
                           'additional_state_code',
                           'backed_underlying_security_id',
                           'bank_qualified',
                           'capital_type',
                           'conditional_call_date',
                           'conditional_call_price',
                           'designated_termination_date',
                           'DTCC_status',
                           'first_execution_date',
                           'formal_award_date',
                           'maturity_description_code',
                           'muni_security_type',
                           'mtg_insurance',
                           'orig_cusip_status',
                           'orig_instrument_enhancement_type',
                           'other_enhancement_type',
                           'other_enhancement_company',
                           'pac_bond_indicator',
                           'project_name',
                           'purpose_class',
                           'purpose_sub_class',
                           'refunding_issue_key',
                           'refunding_dated_date',
                           'sale_date',
                           'sec_regulation',
                           'secured',
                           'series_name',
                           'sink_fund_redemption_method',
                           'state_tax_status',
                           'tax_credit_frequency',
                           'tax_credit_percent',
                           'use_of_proceeds',
                           'use_of_proceeds_supplementary',
                           # 'material_event_history',    # this feature doubles the query cost and is not used in the product
                           # 'default_event_history',    # removed by Developer 2023-05-25
                           # 'most_recent_event',
                           # 'event_exists',
                           'series_id',
                           'security_description',
                           # 'recent',    # removed by Developer 2024-08-30 since we use a separate redis for the trade history data
                           ]
REFERENCE_DATA_FEATURE_TO_INDEX = {feature: idx for idx, feature in enumerate(REFERENCE_DATA_FEATURES)}

# Features used in the final batch pricing csv
FEATURES_FOR_OUTPUT_CSV = ['cusip', 'quantity', 'trade_type', 'ytw', 'price', 'yield_to_worst_date', 'coupon', 'security_description', 'maturity_date', 'error_message']
ADDITIONAL_FEATURES_FOR_COMPLIANCE_CSV = ['user_price', 'bid_ask_price_delta', 'compliance_rating', 'trade_datetime']
TRADE_TYPE_CODE_TO_TEXT = {'D': 'Inter-Dealer', 'P': 'Bid Side', 'S': 'Offered Side'}    # used to display a human-readable trade type in the output csv for batch pricing; NOTE: this is WET and needs to match those on the front end (perform cmd+f on `const tradeType = ` in `pricing.jsx` and make sure the values here match those; 'Bid Side' used to be 'Purchase from Customer' and 'Offered Side' used to be 'Sale to Customer'), and automated testing (find the variable with the same name in `check-demo-status-v2/auxiliary_variables.py`)

FEATURES_NECESSARY_FOR_CREATING_SIMILAR_TRADE_HISTORY_GROUPS = ['issue_key', 'maturity_date', 'coupon']

# Upper and lower bounds for what the user may enter for the quantity; NOTE: this is WET and needs to match those on the front end, perform cmd+f on `required min=` in both `pricing.jsx` and `tabsCusipSearchForm.jsx`
QUANTITY_LOWER_BOUND = 5    # in thousands
QUANTITY_UPPER_BOUND = 10000    # in thousands

REDEMPTION_VALUE_AT_MATURITY = 100    # hard coded for price at maturity

PRICE_BOTH_DIRECTIONS_TO_CORRECT_INVERSION = True    # boolean that indicates whether we manually correct for inversions by pricing both the customer sell direction and the customer buy direction and choose the correct value such that there is no price inversion
LARGE_BATCH_SIZE = 2000 if PRICE_BOTH_DIRECTIONS_TO_CORRECT_INVERSION else 5000
SINGLE_FUNCTION_CALL_BATCH_SIZE = 100    # if the batch size is lesser than or equal to this, then we do not use parallelization due to unnecessary overhead

USE_CACHE_FOR_GET_DATA_FOR_SINGLE_CUSIP = True

X_AXIS_LABEL_FOR_YIELD_CURVE = 'x'    # FIXME: change 'x' to a more descriptive name
Y_AXIS_LABEL_FOR_YIELD_CURVE = 'yield'
ACCEPTABLE_DISPLAY_TYPES_FOR_YIELD_CURVE = ('plot', 'table')
