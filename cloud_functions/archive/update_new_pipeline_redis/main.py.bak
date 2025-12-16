# -*- coding: utf-8 -*-

from google.cloud import bigquery
import os
import pandas as pd
import pickle5 as pickle
import redis
import smtplib, ssl
from email.mime.text import MIMEText
from google.cloud import secretmanager



bq_client = bigquery.Client()
job_config = bigquery.job.QueryJobConfig(allow_large_results=True)
redis_host = os.environ.get('REDISHOST', '10.14.140.37')
redis_port = int(os.environ.get('REDISPORT', 6379))
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)

def sqltodf(sql,limit = ""):
    if limit != "": 
        limit = f" WHERE trade_date < DATE({limit})"
    bqr = bq_client.query(sql + limit).result()
    return bqr.to_dataframe()


#The following selects the latest reference data for a given cusip.
query = """
SELECT
  coupon,
  cusip,
  ref_valid_from_date,
  ref_valid_to_date,
  incorporated_state_code,
  organization_primary_name,
  instrument_primary_name,
  issue_key,
  issue_text,
  conduit_obligor_name,
  is_called,
  is_callable,
  is_escrowed_or_pre_refunded,
  first_call_date,
  call_date_notice,
  callable_at_cav,
  par_price,
  call_defeased,
  call_timing,
  call_timing_in_part,
  extraordinary_make_whole_call,
  extraordinary_redemption,
  make_whole_call,
  next_call_date,
  next_call_price,
  call_redemption_id,
  first_optional_redemption_code,
  second_optional_redemption_code,
  third_optional_redemption_code,
  first_mandatory_redemption_code,
  second_mandatory_redemption_code,
  third_mandatory_redemption_code,
  par_call_date,
  par_call_price,
  maximum_call_notice_period,
  called_redemption_type,
  muni_issue_type,
  refund_date,
  refund_price,
  redemption_cav_flag,
  max_notification_days,
  min_notification_days,
  next_put_date,
  put_end_date,
  put_feature_price,
  put_frequency,
  put_start_date,
  put_type,
  maturity_date,
  sp_long,
  sp_stand_alone,
  sp_icr_school,
  sp_prelim_long,
  sp_outlook_long,
  sp_watch_long,
  sp_Short_Rating,
  sp_Credit_Watch_Short_Rating,
  sp_Recovery_Long_Rating,
  moodys_long,
  moodys_short,
  moodys_Issue_Long_Rating,
  moodys_Issue_Short_Rating,
  moodys_Credit_Watch_Long_Rating,
  moodys_Credit_Watch_Short_Rating,
  moodys_Enhanced_Long_Rating,
  moodys_Enhanced_Short_Rating,
  moodys_Credit_Watch_Long_Outlook_Rating,
  has_sink_schedule,
  next_sink_date,
  sink_indicator,
  sink_amount_type_text,
  sink_amount_type_type,
  sink_frequency,
  sink_defeased,
  additional_next_sink_date,
  sink_amount_type,
  additional_sink_frequency,
  min_amount_outstanding,
  max_amount_outstanding,
  default_exists,
  has_unexpired_lines_of_credit,
  years_to_loc_expiration,
  escrow_exists,
  escrow_obligation_percent,
  escrow_obligation_agent,
  escrow_obligation_type,
  child_linkage_exists,
  put_exists,
  floating_rate_exists,
  bond_insurance_exists,
  is_general_obligation,
  has_zero_coupons,
  delivery_date,
  issue_price,
  primary_market_settlement_date,
  issue_date,
  outstanding_indicator,
  federal_tax_status,
  maturity_amount,
  available_denom,
  denom_increment_amount,
  min_denom_amount,
  accrual_date,
  bond_insurance,
  coupon_type,
  current_coupon_rate,
  daycount_basis_type,
  debt_type,
  default_indicator,
  first_coupon_date,
  interest_payment_frequency,
  issue_amount,
  last_period_accrues_from_date,
  next_coupon_payment_date,
  odd_first_coupon_date,
  orig_principal_amount,
  original_yield,
  outstanding_amount,
  previous_coupon_payment_date,
  sale_type,
  settlement_type,
  additional_project_txt,
  asset_claim_code,
  additional_state_code,
  backed_underlying_security_id,
  bank_qualified,
  capital_type,
  conditional_call_date,
  conditional_call_price,
  designated_termination_date,
  DTCC_status,
  first_execution_date,
  formal_award_date,
  maturity_description_code,
  muni_security_type,
  mtg_insurance,
  orig_cusip_status,
  orig_instrument_enhancement_type,
  other_enhancement_type,
  other_enhancement_company,
  pac_bond_indicator,
  project_name,
  purpose_class,
  purpose_sub_class,
  refunding_issue_key,
  refunding_dated_date,
  sale_date,
  sec_regulation,
  secured,
  series_name,
  sink_fund_redemption_method,
  state_tax_status,
  tax_credit_frequency,
  tax_credit_percent,
  use_of_proceeds,
  use_of_proceeds_supplementary,
  -- material_event_history, These array functions double the query cost and are not used in the demo.
  -- default_event_history,  Removed by Developer-25-2023
  -- most_recent_event,
  -- event_exists,
  series_id,
  security_description,
  recent
FROM
  `auxiliary_views.trade_history_latest_ref_data_minimal_exclusions`
   where date(recent[safe_offset(0)].msrb_valid_from_date) = current_date"""

def upload_data_to_redis(key, value):
    value = pickle.dumps(value,protocol=pickle.HIGHEST_PROTOCOL)
    redis_client.set(key, value)

def main(args):
    df = sqltodf(query)
    num_update_cusips = len(df)

    print(f"""This is the number of updated cusips """)
    print(len(df))

    for _,row in df.iterrows():
        key = row['cusip']
        upload_data_to_redis(key,row)
    return "Upload successful!"


