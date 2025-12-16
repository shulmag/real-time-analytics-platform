/* Test to make sure it only runs on weekdays */
DECLARE should_run BOOL;
SET should_run = (
  SELECT
    EXTRACT(DAYOFWEEK FROM dt_edt) BETWEEN 2 AND 6
/*    AND
    EXTRACT(HOUR FROM dt_edt) BETWEEN 9 AND 17 */
  FROM (
    SELECT DATETIME(CURRENT_TIMESTAMP(), 'America/New_York') AS dt_edt
  )
);

IF should_run THEN

  /* recap_daily — overwrite ALL fields for today with final values */
  MERGE `eng-reactor-287421.muni_market_statistics.recap_daily` AS T
  USING (
    WITH
      today AS (
        SELECT *
        FROM `eng-reactor-287421.auxiliary_views_v2.msrb_final`
        WHERE trade_date = CURRENT_DATE('America/New_York')
      ),
      ytd AS (
        SELECT *
        FROM `eng-reactor-287421.auxiliary_views_v2.msrb_final`
        WHERE trade_date BETWEEN DATE_TRUNC(CURRENT_DATE('America/New_York'), YEAR)
                             AND CURRENT_DATE('America/New_York')
      )
    SELECT
      CURRENT_DATE('America/New_York') AS as_of_date,

      /* overall (today) */
      COUNT(*) AS total_trades,
      COALESCE(SUM(par_traded), 0) AS total_volume,
      SAFE_DIVIDE(SUM(par_traded), NULLIF(COUNT(*), 0)) AS avg_par,

      /* YTD overall */
      (SELECT COUNT(*) FROM ytd) AS total_trades_this_year,

      /* per-type D */
      COUNTIF(trade_type = 'D') AS trades_D,
      COALESCE(SUM(IF(trade_type = 'D', par_traded, 0)), 0) AS volume_D,
      SAFE_DIVIDE(SUM(IF(trade_type = 'D', par_traded, 0)),
                  NULLIF(COUNTIF(trade_type = 'D'), 0)) AS avg_par_D,

      /* per-type P */
      COUNTIF(trade_type = 'P') AS trades_P,
      COALESCE(SUM(IF(trade_type = 'P', par_traded, 0)), 0) AS volume_P,
      SAFE_DIVIDE(SUM(IF(trade_type = 'P', par_traded, 0)),
                  NULLIF(COUNTIF(trade_type = 'P'), 0)) AS avg_par_P,

      /* per-type S */
      COUNTIF(trade_type = 'S') AS trades_S,
      COALESCE(SUM(IF(trade_type = 'S', par_traded, 0)), 0) AS volume_S,
      SAFE_DIVIDE(SUM(IF(trade_type = 'S', par_traded, 0)),
                  NULLIF(COUNTIF(trade_type = 'S'), 0)) AS avg_par_S,

      /* big trades ≥ $5MM */
      COUNTIF(trade_type = 'D' AND par_traded >= 5000000) AS big_trades_over_5mm_dealer,
      COUNTIF(trade_type = 'P' AND par_traded >= 5000000) AS big_trades_over_5mm_buy,
      COUNTIF(trade_type = 'S' AND par_traded >= 5000000) AS big_trades_over_5mm_sell
    FROM today
  ) AS S
  ON T.as_of_date = S.as_of_date
  WHEN MATCHED THEN
    UPDATE SET
      total_trades = S.total_trades,
      total_volume = S.total_volume,
      avg_par = S.avg_par,
      total_trades_this_year = S.total_trades_this_year,
      trades_D = S.trades_D,
      volume_D = S.volume_D,
      avg_par_D = S.avg_par_D,
      trades_P = S.trades_P,
      volume_P = S.volume_P,
      avg_par_P = S.avg_par_P,
      trades_S = S.trades_S,
      volume_S = S.volume_S,
      avg_par_S = S.avg_par_S,
      big_trades_over_5mm_dealer = S.big_trades_over_5mm_dealer,
      big_trades_over_5mm_buy = S.big_trades_over_5mm_buy,
      big_trades_over_5mm_sell = S.big_trades_over_5mm_sell
  WHEN NOT MATCHED THEN
    INSERT (
      as_of_date,
      total_trades,
      total_volume,
      avg_par,
      total_trades_this_year,
      trades_D, volume_D, avg_par_D,
      trades_P, volume_P, avg_par_P,
      trades_S, volume_S, avg_par_S,
      big_trades_over_5mm_dealer, big_trades_over_5mm_buy, big_trades_over_5mm_sell
    )
    VALUES (
      S.as_of_date,
      S.total_trades,
      S.total_volume,
      S.avg_par,
      S.total_trades_this_year,
      S.trades_D, S.volume_D, S.avg_par_D,
      S.trades_P, S.volume_P, S.avg_par_P,
      S.trades_S, S.volume_S, S.avg_par_S,
      S.big_trades_over_5mm_dealer, S.big_trades_over_5mm_buy, S.big_trades_over_5mm_sell
    );

ELSE

END IF;