'''
'''
dedup_calc_date = '''CREATE OR REPLACE TABLE `auxiliary_views_v2.calculation_date_and_price_v2` AS
                     SELECT * FROM (SELECT ROW_NUMBER() OVER (PARTITION BY rtrs_control_number ORDER BY publish_datetime DESC, sequence_number DESC) AS row_num, * EXCEPT (row_num)    -- chose the name row_num instead of row_number because row_number is a reserved word in SQL
                                    FROM `auxiliary_views_v2.calculation_date_and_price_v2`) subquery
                     WHERE row_num = 1'''

dedup_msrb_trade_messages = '''CREATE OR REPLACE TABLE `MSRB.msrb_trade_messages` AS
                               SELECT DISTINCT * FROM `MSRB.msrb_trade_messages`'''

create_materialized_trade_history = '''CREATE OR REPLACE TABLE `auxiliary_views_v2.trade_history_same_issue_5_yr_mat_bucket_1_materialized`
                                        PARTITION BY DATE(trade_date)
                                        AS
                                        SELECT * FROM `eng-reactor-287421.auxiliary_views_v2.trade_history_ref_data_same_issue_5_yr_mat_bucket_1_coupon`'''



QUERIES = {'dedup_calc_date': dedup_calc_date, 
           'dedup_msrb_trade_messages': dedup_msrb_trade_messages, 
           'create_materialized_trade_history': create_materialized_trade_history}
