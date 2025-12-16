'''
Create Date: 2025-01-03
'''
DAILY_STATS_QUERY = """
    WITH trades AS (
        SELECT
            *,
            DATE_DIFF(trade_date, dated_date, YEAR) AS years_since_dated_date,
            DATE_DIFF(maturity_date, CURRENT_DATE('US/Eastern'), YEAR) AS years_to_maturity
        FROM
            `eng-reactor-287421.auxiliary_views_v2.msrb_final`
        WHERE
            trade_date BETWEEN DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 29 DAY) AND CURRENT_DATE('US/Eastern')
            AND msrb_valid_to_date > CURRENT_DATETIME('US/Eastern')
    ),
    big_trades_over_5mm AS (
        SELECT
            trade_date,
            COUNT(DISTINCT IF(trade_type = 'P', rtrs_control_number, NULL))
            AS customer_sell_trades_over_5_MM,
            COUNT(DISTINCT IF(trade_type = 'D', rtrs_control_number, NULL))
            AS dealer_trades_over_5_MM,
            COUNT(DISTINCT IF(trade_type = 'S', rtrs_control_number, NULL))
            AS customer_buy_trades_over_5_MM
        FROM `MSRB.msrb_trade_messages`
        WHERE
            is_trade_with_a_par_amount_over_5MM
            -- keep the same 30-day filter so the numbers line up
            AND trade_date BETWEEN DATE_SUB(CURRENT_DATE('US/Eastern'), INTERVAL 29 DAY)
                            AND CURRENT_DATE('US/Eastern')
        GROUP BY trade_date
        ),
    most_active_new_issue AS (
        SELECT
            trade_date,
            ARRAY_AGG(STRUCT(cusip, security_description) ORDER BY cnt DESC LIMIT 1)[OFFSET(0)] AS most_active
        FROM (
            SELECT
                trade_date,
                cusip,
                security_description,
                COUNT(*) AS cnt
            FROM
                trades
            WHERE
                years_since_dated_date < 1
            GROUP BY
                trade_date, cusip, security_description
        )
        GROUP BY
            trade_date
    ),
    most_active_seasoned AS (
        SELECT
            trade_date,
            ARRAY_AGG(STRUCT(cusip, security_description) ORDER BY cnt DESC LIMIT 1)[OFFSET(0)] AS most_active
        FROM (
            SELECT
                trade_date,
                cusip,
                security_description,
                COUNT(*) AS cnt
            FROM
                trades
            WHERE
                years_since_dated_date >= 1
            GROUP BY
                trade_date, cusip, security_description
        )
        GROUP BY
            trade_date
    ),
    trades_year AS (
        SELECT
            trade_date,
            COUNT(DISTINCT rtrs_control_number) AS daily_trade_count
        FROM
            `eng-reactor-287421.auxiliary_views_v2.msrb_final`
        WHERE
            EXTRACT(YEAR FROM trade_date) = EXTRACT(YEAR FROM CURRENT_DATE('US/Eastern'))
        GROUP BY
            trade_date
    ),
    cumulative_trades AS (
        SELECT
            trade_date,
            SUM(daily_trade_count) OVER (ORDER BY trade_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS num_trades_this_year
        FROM
            trades_year
    )
    SELECT
        t.trade_date,
        COUNT(DISTINCT t.rtrs_control_number) AS trade_count,
        ROUND(AVG(t.par_traded)) AS average_par_traded,
        SUM(t.par_traded) AS total_trade_volume,
        COUNT(DISTINCT CASE WHEN t.trade_type = 'D' THEN t.rtrs_control_number END) AS num_dealer_trades,
        COUNT(DISTINCT CASE WHEN t.trade_type = 'S' THEN t.rtrs_control_number END) AS num_customer_buy,
        COUNT(DISTINCT CASE WHEN t.trade_type = 'P' THEN t.rtrs_control_number END) AS num_customer_sell,
        o.customer_sell_trades_over_5_MM,
        o.dealer_trades_over_5_MM,
        o.customer_buy_trades_over_5_MM,
        COUNT(DISTINCT CASE WHEN t.upload_date = t.trade_date AND t.transaction_type <> 'I' THEN t.rtrs_control_number END) AS modified_trades,
        COUNT(DISTINCT CASE WHEN t.upload_date = t.trade_date AND t.transaction_type = 'C' THEN t.rtrs_control_number END) AS cancelled_trades,
        ct.num_trades_this_year,
        new_issue.most_active.cusip AS most_actively_traded_new_issue,
        new_issue.most_active.security_description AS description,
        seasoned.most_active.cusip AS most_actively_traded_seasoned_cusip,
        seasoned.most_active.security_description AS description_2,
        ROUND(AVG(CASE WHEN t.coupon = 3 AND t.years_to_maturity BETWEEN 4 AND 6 THEN t.dollar_price END), 3) AS avg_price_for_3_percent_coupon_5_yr,
        ROUND(AVG(CASE WHEN t.coupon = 3 AND t.years_to_maturity BETWEEN 4 AND 6 THEN t.yield END), 2) AS avg_yield_for_3_percent_coupon_5_yr,
        ROUND(AVG(CASE WHEN t.coupon = 3 AND t.years_to_maturity BETWEEN 9 AND 11 THEN t.dollar_price END), 3) AS avg_price_for_3_percent_coupon_10_yr,
        ROUND(AVG(CASE WHEN t.coupon = 3 AND t.years_to_maturity BETWEEN 9 AND 11 THEN t.yield END), 2) AS avg_yield_for_3_percent_coupon_10_yr,
        ROUND(AVG(CASE WHEN t.coupon = 3 AND t.years_to_maturity BETWEEN 19 AND 21 THEN t.dollar_price END), 3) AS avg_price_for_3_percent_coupon_20_yr,
        ROUND(AVG(CASE WHEN t.coupon = 3 AND t.years_to_maturity BETWEEN 19 AND 21 THEN t.yield END), 2) AS avg_yield_for_3_percent_coupon_20_yr,
        ROUND(AVG(CASE WHEN t.coupon = 4 AND t.years_to_maturity BETWEEN 4 AND 6 THEN t.dollar_price END), 3) AS avg_price_for_4_percent_coupon_5_yr,
        ROUND(AVG(CASE WHEN t.coupon = 4 AND t.years_to_maturity BETWEEN 4 AND 6 THEN t.yield END), 2) AS avg_yield_for_4_percent_coupon_5_yr,
        ROUND(AVG(CASE WHEN t.coupon = 4 AND t.years_to_maturity BETWEEN 9 AND 11 THEN t.dollar_price END), 3) AS avg_price_for_4_percent_coupon_10_yr,
        ROUND(AVG(CASE WHEN t.coupon = 4 AND t.years_to_maturity BETWEEN 9 AND 11 THEN t.yield END), 2) AS avg_yield_for_4_percent_coupon_10_yr,
        ROUND(AVG(CASE WHEN t.coupon = 4 AND t.years_to_maturity BETWEEN 19 AND 21 THEN t.dollar_price END), 3) AS avg_price_for_4_percent_coupon_20_yr,
        ROUND(AVG(CASE WHEN t.coupon = 4 AND t.years_to_maturity BETWEEN 19 AND 21 THEN t.yield END), 2) AS avg_yield_for_4_percent_coupon_20_yr,
        ROUND(AVG(CASE WHEN t.coupon = 5 AND t.years_to_maturity BETWEEN 4 AND 6 THEN t.dollar_price END), 3) AS avg_price_for_5_percent_coupon_5_yr,
        ROUND(AVG(CASE WHEN t.coupon = 5 AND t.years_to_maturity BETWEEN 4 AND 6 THEN t.yield END), 2) AS avg_yield_for_5_percent_coupon_5_yr,
        ROUND(AVG(CASE WHEN t.coupon = 5 AND t.years_to_maturity BETWEEN 9 AND 11 THEN t.dollar_price END), 3) AS avg_price_for_5_percent_coupon_10_yr,
        ROUND(AVG(CASE WHEN t.coupon = 5 AND t.years_to_maturity BETWEEN 9 AND 11 THEN t.yield END), 2) AS avg_yield_for_5_percent_coupon_10_yr,
        ROUND(AVG(CASE WHEN t.coupon = 5 AND t.years_to_maturity BETWEEN 19 AND 21 THEN t.dollar_price END), 3) AS avg_price_for_5_percent_coupon_20_yr,
        ROUND(AVG(CASE WHEN t.coupon = 5 AND t.years_to_maturity BETWEEN 19 AND 21 THEN t.yield END), 2) AS avg_yield_for_5_percent_coupon_20_yr
    FROM
        trades t
    LEFT JOIN big_trades_over_5mm o ON t.trade_date = o.trade_date
    LEFT JOIN most_active_new_issue new_issue ON t.trade_date = new_issue.trade_date
    LEFT JOIN most_active_seasoned seasoned ON t.trade_date = seasoned.trade_date
    LEFT JOIN cumulative_trades ct ON t.trade_date = ct.trade_date
    GROUP BY
        t.trade_date,
        o.customer_sell_trades_over_5_MM,
        o.dealer_trades_over_5_MM,
        o.customer_buy_trades_over_5_MM,
        ct.num_trades_this_year,
        new_issue.most_active.cusip,
        new_issue.most_active.security_description,
        seasoned.most_active.cusip,
        seasoned.most_active.security_description
    ORDER BY
        t.trade_date;
    """
