import os
from datetime import datetime

from pytz import timezone
import redis

from google.cloud import logging


MULTIPROCESSING = False
FILENAME_ADDENDUM = '_from_fast_redis_update_v2'

# set up logging client; https://cloud.google.com/logging/docs/setup/python
logging_client = logging.Client()
logging_client.setup_logging()

_redis_port = int(os.environ.get('REDISPORT', 6379))    # shared across all redis instances
_reference_data_redis_host = os.environ.get('REDISHOST', '192.168.240.19')    # using the read endpoint to not accidentally corrupt the redis by attempting to write to it
reference_data_redis_client = redis.Redis(host=_reference_data_redis_host, port=_redis_port, db=0)
_trade_history_redis_host = os.environ.get('REDISHOST', '192.168.240.3')
trade_history_redis_client = redis.Redis(host=_trade_history_redis_host, port=_redis_port, db=0)
_similar_trade_history_redis_host = os.environ.get('REDISHOST', '192.168.240.11')
similar_trade_history_redis_client = redis.Redis(host=_similar_trade_history_redis_host, port=_redis_port, db=0)

EASTERN = timezone('US/Eastern')

YEAR_MONTH_DAY = '%Y-%m-%d'
HOUR_MIN_SEC = '%H:%M:%S'

DATETIME_FAR_INTO_FUTURE = datetime(2100, 1, 1, 0, 0, 0, 0)

MAX_NUM_TRADES_IN_HISTORY = 32
FEATURES_FOR_EACH_TRADE_IN_HISTORY = {'msrb_valid_from_date': 'DATETIME', 
                                      'msrb_valid_to_date': 'DATETIME', 
                                      'rtrs_control_number': 'INTEGER', 
                                      'trade_datetime': 'DATETIME', 
                                      'publish_datetime': 'DATETIME', 
                                      'yield': 'FLOAT', 
                                      'dollar_price': 'FLOAT', 
                                      'par_traded': 'NUMERIC', 
                                      'trade_type': 'STRING', 
                                      'is_non_transaction_based_compensation': 'BOOLEAN', 
                                      'is_lop_or_takedown': 'BOOLEAN', 
                                      'brokers_broker': 'STRING', 
                                      'is_alternative_trading_system': 'BOOLEAN', 
                                      'is_weighted_average_price': 'BOOLEAN', 
                                      'settlement_date': 'DATE', 
                                      'calc_date': 'DATE', 
                                      'calc_day_cat': 'INTEGER', 
                                      'maturity_date': 'DATE', 
                                      'next_call_date': 'DATE', 
                                      'par_call_date': 'DATE', 
                                      'refund_date': 'DATE', 
                                      'transaction_type': 'STRING', 
                                      'sequence_number': 'INTEGER'}

MAX_NUM_TRADES_IN_SIMILAR_TRADE_HISTORY = 64    # wanted a value larger than 32 in case the last many trades were from the same CUSIP and so chose the next power of 2

MAX_NUM_DAYS_FOR_TRADES_IN_HISTORY = 720    # number of days for which if there are more than `MAX_NUM_TRADES_IN_HISTORY` for a particular CUSIP, then we keep only those that are within `MAX_NUM_DAYS_FOR_TRADES_IN_HISTORY` from the current datetime for the trade history redis; similar logic for the similar trade history redis; this value should be the same as `MAX_NUM_DAYS_FOR_REFERENCE_DATA_POINT_IN_TIME` in `cloud_functions/reference_data_redis_update/main.py`
MAX_TRADES_USED_IN_MODEL = 5    # yield spread with similar trades model uses 5 trades in the history, dollar price model uses 2 trades in the history, and so this value is max(2, 5) = 5

NUM_OF_MONTHS_IN_YEAR = 12
NUM_OF_WEEKS_IN_YEAR = 52
NUM_OF_DAYS_IN_YEAR = 360

COUPON_FREQUENCY_DICT = {0: 'Unknown',
                         1: 'Semiannually',
                         2: 'Monthly',
                         3: 'Annually',
                         4: 'Weekly',
                         5: 'Quarterly',
                         6: 'Every 2 years',
                         7: 'Every 3 years',
                         8: 'Every 4 years',
                         9: 'Every 5 years',
                         10: 'Every 7 years',
                         11: 'Every 8 years',
                         12: 'Biweekly',
                         13: 'Changeable',
                         14: 'Daily',
                         15: 'Term mode',
                         16: 'Interest at maturity',
                         17: 'Bimonthly',
                         18: 'Every 13 weeks',
                         19: 'Irregular',
                         20: 'Every 28 days',
                         21: 'Every 35 days',
                         22: 'Every 26 weeks',
                         23: 'Not Applicable',
                         24: 'Tied to prime',
                         25: 'One time',
                         26: 'Every 10 years',
                         27: 'Frequency to be determined',
                         28: 'Mandatory put',
                         29: 'Every 52 weeks',
                         30: 'When interest adjusts-commercial paper',
                         31: 'Zero coupon',
                         32: 'Certain years only',
                         33: 'Under certain circumstances',
                         34: 'Every 15 years',
                         35: 'Custom',
                         36: 'Single Interest Payment'}

LARGE_NUMBER = 1e6

COUPON_FREQUENCY_TYPE = {'Unknown': LARGE_NUMBER,
                         'Semiannually': 2,
                         'Monthly': 12,
                         'Annually': 1,
                         'Weekly': 52,
                         'Quarterly': 4,
                         'Every 2 years': 0.5,
                         'Every 3 years': 1/3,
                         'Every 4 years': 0.25,
                         'Every 5 years': 0.2,
                         'Every 7 years': 1/7,
                         'Every 8 years': 1/8,
                         'Biweekly':  26,
                         'Changeable': 44,
                         'Daily': 360,
                         'Interest at maturity': 0,
                         'Not Applicable': LARGE_NUMBER}

# identical to `app_engine/demo/server/modules/auxiliary_functions.py::REFERENCE_DATA_FEATURES`
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

PROJECT_ID = 'eng-reactor-287421'
LOCATION = 'US'

ALL_TRADE_MESSAGES_FILENAME = 'all_trade_messages'
MSRB_INTRADAY_FILES_BUCKET_NAME = 'msrb_intraday_real_time_trade_files'

LOGGING_PRECISION = 3    # choosing to log 3 digits after the decimal point since price values are only 3 digits after the decimal point

DATA_TYPE_DICT = {'upload_date': 'date', 
                  'message_type': 'string', 
                  'sequence_number': 'integer', 
                  'rtrs_control_number': 'integer', 
                  'trade_type': 'string', 
                  'transaction_type': 'string', 
                  'cusip': 'string', 
                  'security_description': 'string', 
                  'dated_date': 'date', 
                  'coupon': 'numeric', 
                  'maturity_date': 'date', 
                  'when_issued': 'boolean', 
                  'assumed_settlement_date': 'date', 
                  'trade_date': 'date', 
                  'time_of_trade': 'time', 
                  'settlement_date': 'date', 
                  'par_traded': 'numeric', 
                  'dollar_price': 'float', 
                  'yield': 'float', 
                  'brokers_broker': 'string', 
                  'is_weighted_average_price': 'boolean', 
                  'is_lop_or_takedown': 'boolean', 
                  'publish_date': 'date', 
                  'publish_time': 'time', 
                  'version': 'numeric', 
                  'unable_to_verify_dollar_price': 'boolean', 
                  'is_alternative_trading_system': 'boolean', 
                  'is_non_transaction_based_compensation': 'boolean', 
                  'is_trade_with_a_par_amount_over_5MM': 'boolean'}

future_processing_file_name = 'trades_for_future_processing_v2.pkl'
future_processing_file_bucket_name = 'fast_trade_history_redis_update_files'
