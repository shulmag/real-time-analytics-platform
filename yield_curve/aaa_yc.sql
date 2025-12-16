-- AAA Yield Curve Construction from MSRB Trade Data
-- This query creates bucketed yields for AAA-rated municipal bonds

DECLARE as_of_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP();

WITH todays_trades AS (
  SELECT
    t.cusip,
    t.yield,
    t.dollar_price,
    t.par_traded,
    t.trade_date,
    t.time_of_trade,
    t.maturity_date,
    t.coupon,
    -- Reference data fields
    rd.sp_long,
    rd.next_call_date,
    rd.current_coupon_rate,
    rd.min_amount_outstanding,
    rd.instrument_primary_name,
    rd.incorporated_state_code,
    -- Calculate years to maturity or call (whichever is sooner)
    CASE 
      WHEN rd.next_call_date IS NOT NULL AND rd.next_call_date < t.maturity_date 
      THEN DATE_DIFF(rd.next_call_date, CURRENT_DATE(), DAY) / 365.25
      ELSE DATE_DIFF(t.maturity_date, CURRENT_DATE(), DAY) / 365.25
    END AS years_to_worst
  FROM `eng-reactor-287421.MSRB.msrb_trade_messages` t
  JOIN `eng-reactor-287421.reference_data_v2.reference_data_flat` rd 
    ON t.cusip = rd.cusip
  WHERE 
    -- Time window: just today
    t.trade_date >= CURRENT_DATE() -- DATE_SUB(CURRENT_DATE(), INTERVAL 5 DAY)
    -- which trade type to include? 
    AND t.trade_type IN ('S', 'P','D')
    -- Minimum trade size $250k
    -- AND t.par_traded >= 250000
    -- Minimum outstanding $25MM
    -- AND rd.min_amount_outstanding >= 25000000
    -- AAA rating requirement (S&P only for simplicity)
    AND rd.sp_long = 'AAA'
    -- Non-callable (or callable beyond our longest bucket)
    AND (rd.next_call_date IS NULL OR rd.next_call_date > DATE_ADD(CURRENT_DATE(), INTERVAL 30 YEAR))
    -- 5% coupon bonds only (using trade data coupon field)
    AND t.coupon = 5.0
    -- General Obligation bonds only
    AND UPPER(rd.instrument_primary_name) LIKE '%GENERAL OBLIGATION%'
    -- Exclude when-issued trades
    AND t.when_issued = false
    -- Valid yield (not null or zero)
    AND t.yield IS NOT NULL 
    AND t.yield > 0
),

-- Calculate volume-weighted average yield by CUSIP for the period
issue_yields AS (
  SELECT
    cusip,
    MAX(sp_long) AS sp_rating,
    MAX(incorporated_state_code) AS state,
    MAX(instrument_primary_name) AS bond_description,
    MAX(min_amount_outstanding) AS par_outstanding,
    -- Volume-weighted average yield
    SUM(yield * par_traded) / SUM(par_traded) AS ytw_issue,
    -- Average years to worst
    AVG(years_to_worst) AS avg_years_to_worst,
    -- Trade statistics
    COUNT(*) AS trade_count,
    SUM(par_traded) AS total_volume,
    MAX(trade_date) AS last_trade_date,
    MAX(time_of_trade) AS last_trade_time
  FROM todays_trades
  GROUP BY cusip
),

-- Assign to maturity buckets
bucketed AS (
  SELECT
    CASE
      WHEN avg_years_to_worst BETWEEN 0.5 AND 1.5 THEN '1Y'
      WHEN avg_years_to_worst BETWEEN 1.5 AND 2.5 THEN '2Y'
      WHEN avg_years_to_worst BETWEEN 2.5 AND 3.5 THEN '3Y'
      WHEN avg_years_to_worst BETWEEN 3.5 AND 4.5 THEN '4Y'
      WHEN avg_years_to_worst BETWEEN 4.5 AND 5.5 THEN '5Y'
      WHEN avg_years_to_worst BETWEEN 6.5 AND 7.5 THEN '7Y'
      WHEN avg_years_to_worst BETWEEN 9.5 AND 10.5 THEN '10Y'
      WHEN avg_years_to_worst BETWEEN 14.5 AND 15.5 THEN '15Y'
      WHEN avg_years_to_worst BETWEEN 19.5 AND 20.5 THEN '20Y'
      WHEN avg_years_to_worst BETWEEN 24.5 AND 35.0 THEN '30Y'
      ELSE NULL
    END AS bucket,
    cusip,
    ytw_issue,
    par_outstanding,
    trade_count,
    total_volume,
    last_trade_date,
    last_trade_time,
    sp_rating,
    state
  FROM issue_yields
  WHERE avg_years_to_worst > 0  -- Exclude already matured bonds
)

-- Final aggregation by bucket
SELECT
  bucket,
  -- Cap-weighted average yield
  ROUND(SUM(ytw_issue * par_outstanding) / SUM(par_outstanding), 3) AS bucket_yield,
  -- Statistics
  COUNT(DISTINCT cusip) AS n_issues,
  COUNT(DISTINCT state) AS n_states,
  SUM(trade_count) AS total_trades,
  ROUND(SUM(total_volume) / 1000000, 1) AS total_volume_mm,
  ROUND(SUM(par_outstanding) / 1000000, 1) AS total_outstanding_mm,
  -- Data quality indicators
  MAX(last_trade_date) AS most_recent_trade_date,
  MAX(last_trade_time) AS most_recent_trade_time,
  ROUND(STDDEV(ytw_issue), 3) AS yield_stddev,
  ROUND(MIN(ytw_issue), 3) AS min_yield,
  ROUND(MAX(ytw_issue), 3) AS max_yield,
  -- Timestamp
  CURRENT_TIMESTAMP() AS calc_ts
FROM bucketed
WHERE bucket IS NOT NULL
GROUP BY bucket
ORDER BY 
  CASE bucket
    WHEN '1Y' THEN 1
    WHEN '2Y' THEN 2
    WHEN '3Y' THEN 3
    WHEN '4Y' THEN 4
    WHEN '5Y' THEN 5
    WHEN '7Y' THEN 7
    WHEN '10Y' THEN 10
    WHEN '15Y' THEN 15
    WHEN '20Y' THEN 20
    WHEN '30Y' THEN 30
  END;