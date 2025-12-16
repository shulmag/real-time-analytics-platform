'''
'''
import os

import pandas as pd


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/user/ficc/ficc/mitas_creds.json'    # filepath for creds file on local machine


from main import typecast_for_bigquery, upload_calculation_date_and_price_to_bigquery


pd.options.mode.chained_assignment = None    # default='warn'; suppresses `SettingWithCopyWarning` in Pandas


test_df = [{'rtrs_control_number': 2023081811496200, 'trade_datetime': '2023-08-18T17:10:02', 'cusip': '13077EJD3', 'calc_price': 86.597, 'price_to_next_call': 94.707, 'price_to_par_call': 94.707, 'price_to_maturity': 86.597, 'calc_date': '2050-09-01', 'next_call_date': '2030-09-01', 'par_call_date': '2030-09-01', 'maturity_date': '2050-09-01', 'refund_date': None, 'price_delta': 0.002, 'publish_datetime': '2023-08-18T17:10:26', 'when_issued': False, 'calc_day_cat': 2, 'issue_key': 1179120, 'sequence_number': 46928, 'par_traded': 20000, 'series_name': '2020', 'series_id': 1179120, 'msrb_valid_to_date': '2100-01-01T00:00:00', 'msrb_valid_from_date': '2023-08-18T17:10:26', 'brokers_broker': None, 'assumed_settlement_date': None, 'unable_to_verify_dollar_price': False}]
successful_rows = []
failure_rows = []
for idx, row in enumerate(test_df):
    test_df = pd.DataFrame.from_dict([row])
    test_df = typecast_for_bigquery(test_df, {'issue_key': 'Int64'})
    try:
        upload_calculation_date_and_price_to_bigquery(test_df)
        successful_rows.append(idx)
    except Exception as e:
        # print(f'row {idx} had exception: {e}')
        failure_rows.append(idx)

print(f'successful_rows: {successful_rows}')
print(f'failure_rows: {failure_rows}')