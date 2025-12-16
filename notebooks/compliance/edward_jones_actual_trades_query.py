'''
Description: The BigQuery table `eng-reactor-287421.jesse_tests.ej_data_for_compliance` was created by running the notebook: 
             `ficc/data_pipeline/define_views_notebooks/create_trade_history_with_reference_data_unified.ipynb` with the line 
             `AND par_traded >= 10000` removed. The notebook needs to be re-run up until this cell (the one that defines `trans` 
             and starts with `trans = mkview`). Once the edward_jones table has been created it is essential that the restriction 
             be added back in and the notebook is re-run before the materialized_trade_history query is run at 11pm ET. The process was the following: 
             (1) We took the csv we received from Edward Jones and converted the `SIDE` and `CLNT_TY_DESC` to `msrb_trade_type`. Specifically: `msrb_trade_type` 
                 is ‘D’ for all trades in which the `CLNT_TY_DESC` is ‘Dealer’. For trades where the `CLNT_TY_DESC` is ‘Client’, if the `SIDE` is ‘Sell’ the 
                 `msrb_trade_type` is ‘P’ and the if the `SIDE` is ‘Buy‘, the msrb_trade_type is ‘S’. The result is this table: `eng-reactor-287421.jesse_tests.ej_trades_processed`
             (2) We then take five columns: `cusip`, `quantity`, `msrb_trade_type`, `price`, and `trade_datetime` and attempt to match them to the msrb data, i.e. to find a 
                 `rtrs_control_number` that matches the trade. In some cases there are multiple matches, so we take the first. Once we have unique `rtrs_control_number`s, we then 
                 can find the right trades in materialized_trade_history which results in this table: `eng-reactor-287421.jesse_tests.ej_data_for_compliance`
'''
import os
import sys
import pandas as pd

from google.cloud import bigquery


server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'app_engine', 'demo', 'server'))    # get the directory containing the 'app_engine/demo/server' package
sys.path.append(server_dir)    # add the directory to sys.path

from modules.get_creds import get_creds
from modules.ficc.utils.auxiliary_functions import sqltodf


get_creds()
bq_client = bigquery.Client()


PICKLE_FILE_NAME = 'ej_compliance_trades.pkl'    # if this pickle file does not exist locally, then look in Google Cloud Storage bucket for the file: `gs://edward_jones_compliance_2025-01-28/ej_compliance_trades.pkl`
OUTPUT_FILE_NAME = 'compliance_edward_jones.csv'

COLUMNS_TO_RENAME_PRIOR_TO_RUNNING_COMPLIANCE = {'SIDE': 'compliance_side'}    # `demo/server/modules/compliance.py` needs the variable name to be `compliance_side`
COLUMN_VALUES_TO_TRANSFORM_PRIOR_TO_RUNNING_COMPLIANCE = {'compliance_side': {'Buy': 'S', 'Sell': 'P'}}

EXTRA_COLUMNS_FOR_OUTPUT = ['FI_TRD_TRAN_ID', 
                            'CLNT_TY_DESC', 
                            'compliance_side']    # corresponds to `SIDE` but has been renamed downstream

# if the BigQuery table `eng-reactor-287421.jesse_tests.ej_data_for_compliance` is not present, then look in the Google Cloud Storage bucket for the file: `gs://edward_jones_compliance_2025-01-28/ej_data_for_compliance_2025-01-28`
QUERY = '''SELECT
    ej_quantity AS quantity,
    ej_trade_type AS trade_type,
    ej_user_price AS user_price,
    CLNT_TY_DESC,
    SIDE,
    FI_TRD_TRAN_ID,
    price,
    MSRB_maturity_date,
    MSRB_coupon_rate,
    transaction_type,
    security_description,
    dated_date,
    calc_price,
    price_to_next_call,
    price_to_par_call,
    price_to_maturity,
    calc_date,
    price_delta,
    issue_key,
    series_id,
    obligor_id,
    calc_day_cat,
    first_published_datetime,
    MSRB_INST_ORDR_DESC,
    MSRB_valid_from_date,
    MSRB_valid_to_date,
    settlement_date,
    par_traded,
    trade_datetime,
    publish_datetime,
    upload_date,
    sequence_number,
    rtrs_control_number,
    when_issued,
    assumed_settlement_date,
    trade_date,
    time_of_trade,
    dollar_price,
    yield,
    brokers_broker,
    is_weighted_average_price,
    is_lop_or_takedown,
    publish_date,
    publish_time,
    version,
    unable_to_verify_dollar_price,
    is_alternative_trading_system,
    is_non_transaction_based_compensation,
    is_trade_with_a_par_amount_over_5MM,
    cusip_with_large_conversion_deltas,
    coupon,
    ice_file_date,
    id,
    cusip,
    ref_valid_from_date,
    ref_valid_to_date,
    incorporated_state_code,
    organization_primary_name,
    ice_organization_id,
    instrument_primary_name,
    issue_text,
    conduit_obligor_name,
    is_called,
    is_callable,
    is_escrowed_or_pre_refunded,
    refund_date,
    refund_price,
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
    sp_Short_Rating,
    sp_Credit_Watch_Short_Rating,
    sp_Recovery_Long_Rating,
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
    event_exists,
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
    next_reset_date,
    first_variable_reset_date,
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
    other_accrual_date,
    orig_principal_amount,
    orig_avg_life_date,
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
    sp_long_integer,
    rating_upgrade,
    rating_downgrade,
    min_sp_rating_this_year,
    max_sp_rating_this_year,
    rating_downgrade_to_junk,
    sp_long,
    sp_stand_alone,
    sp_icr_school,
    sp_prelim_long,
    sp_outlook_long,
    sp_watch_long,
    most_recent_event_1_days_ago,
    most_recent_default_event,
    days_since_most_recent_event,
    days_since_most_recent_default_event,
    de_minimis_threshold,
    price_yield_reporting_error,
    seq_num,
    recent,
    recent_5_year_mat AS recent_similar
FROM
    `eng-reactor-287421.jesse_tests.ej_data_for_compliance`'''


def get_edward_jones_df() -> pd.DataFrame:
    if os.path.exists(PICKLE_FILE_NAME):
        df = pd.read_pickle(PICKLE_FILE_NAME)
    else:
        print('Could not find pickle file. Consider looking in Google Cloud Storage bucket for the file: `gs://edward_jones_compliance_2025-01-28/ej_compliance_trades.pkl`')
        df = sqltodf(QUERY)
        df.to_pickle(PICKLE_FILE_NAME)

    interdealer = df['CLNT_TY_DESC'] == 'Dealer'
    df.loc[interdealer, 'trade_type'] = 'D'    # use `D` for the trade type so the price we use is the inter-dealer price

    df = df.rename(columns=COLUMNS_TO_RENAME_PRIOR_TO_RUNNING_COMPLIANCE)
    columns_set = set(df.columns)
    for column, transform_map in COLUMN_VALUES_TO_TRANSFORM_PRIOR_TO_RUNNING_COMPLIANCE.items():
        if column in columns_set:
            df[column] = df[column].map(transform_map)    # used to choose the direction we move when creating the thresholds for the compliance rating
    return df


def prepare_priced_df_for_edward_jones(df: pd.DataFrame) -> pd.DataFrame:
    columns_set = set(df.columns)
    for column, transform_map in COLUMN_VALUES_TO_TRANSFORM_PRIOR_TO_RUNNING_COMPLIANCE.items():
        if column in columns_set:
            flipped_transform_map = {new_value: old_value for old_value, new_value in transform_map.items()}
            df[column] = df[column].map(flipped_transform_map)    # used to choose the direction we move when creating the thresholds for the compliance rating
    new_column_to_old_column = {new_column: old_column for old_column, new_column in COLUMNS_TO_RENAME_PRIOR_TO_RUNNING_COMPLIANCE.items() if new_column in columns_set}
    df = df.rename(columns=new_column_to_old_column)
    return df
