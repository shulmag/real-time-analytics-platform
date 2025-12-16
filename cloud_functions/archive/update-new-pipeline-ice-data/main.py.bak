import time
from datetime import timedelta
import pickle
import redis

from google.cloud import bigquery


BQ_CLIENT = bigquery.Client()
job_config = bigquery.job.QueryJobConfig()

REFERENCE_DATA_REDIS_CLIENT = redis.Redis(host='10.14.140.37', port=6379, db=0)

features = ['coupon',
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
            'recent',
            ]


def get_new_reference_data_query():
    '''Return a query to select the latest reference data for a given cusip. The condition 
    '(date(ref_valid_from_date) = current_date("America/New_York") OR date(ref_valid_from_date) = "2010-01-01")' 
    will retrieve newly published reference data that is either the first record (hence "2010-01-01", 
    a ficc.ai convention) or which has been published on the current date.'''
    return f'''SELECT {", ".join(features)} 
               FROM eng-reactor-287421.auxiliary_views.trade_history_latest_ref_data_minimal_exclusions 
               WHERE (date(ref_valid_from_date) = current_date("America/New_York") OR date(ref_valid_from_date) = "2010-01-01")'''


def sqltodf(sql, limit=''):
    if limit != '': limit = f' WHERE trade_date < DATE({limit})'
    bqr = BQ_CLIENT.query(sql + limit).result()
    return bqr.to_dataframe()


def upload_data_to_redis(key, unpickled_value):
    REFERENCE_DATA_REDIS_CLIENT.set(key, pickle.dumps(unpickled_value))


def main(args):
    new_reference_data_query = get_new_reference_data_query()
    print(f'Query to get new reference data:\n{new_reference_data_query}')
    new_reference_data = sqltodf(new_reference_data_query)
    cusips = new_reference_data['cusip'].tolist()
    print(f'''Updating the reference data for {len(cusips)} cusips: {sorted(cusips)}''')

    start_time = time.time()
    for _, row in new_reference_data.iterrows():    # TODO: use parallelization if `len(cusips)` is large
        cusip = row['cusip']
        upload_data_to_redis(cusip, row)
    end_time = time.time()
    print(f'Update complete. Execution time: {timedelta(seconds=end_time - start_time)}')
    return 'SUCCESS'
