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

MERGE `eng-reactor-287421.muni_market_statistics.recap_top_issues` T
USING (
  WITH today AS (
    SELECT *
    FROM `eng-reactor-287421.auxiliary_views_v2.msrb_final`
    WHERE trade_date = CURRENT_DATE('US/Eastern')
  ),
  grouped AS (
    SELECT
      CURRENT_DATE('US/Eastern') AS as_of_date,
      IF(DATE_DIFF(trade_date, dated_date, YEAR) = 0, 'new', 'seasoned') AS issue_type,
      cusip,
      ANY_VALUE(security_description) AS security_description,
      COUNT(*) AS trade_count
    FROM today
    GROUP BY as_of_date, issue_type, cusip
  ),
  ranked AS (
    SELECT
      as_of_date, issue_type, cusip, security_description, trade_count,
      ROW_NUMBER() OVER (PARTITION BY issue_type ORDER BY trade_count DESC) AS rn
    FROM grouped
  )
  SELECT as_of_date, issue_type, cusip, security_description, trade_count
  FROM ranked
  WHERE rn <= 10
) S
ON  T.as_of_date = S.as_of_date
AND T.issue_type = S.issue_type
AND T.cusip      = S.cusip
WHEN MATCHED THEN
  UPDATE SET
    security_description = S.security_description,
    trade_count          = S.trade_count
WHEN NOT MATCHED THEN
  INSERT (as_of_date, issue_type, cusip, security_description, trade_count)
  VALUES (S.as_of_date, S.issue_type, S.cusip, S.security_description, S.trade_count);

/* Optional cleanup: keep only the latest top-10 for today */
DELETE FROM `eng-reactor-287421.muni_market_statistics.recap_top_issues`
WHERE as_of_date = CURRENT_DATE('US/Eastern')
  AND CONCAT(issue_type, '|', cusip) NOT IN (
    WITH today AS (
      SELECT *
      FROM `eng-reactor-287421.auxiliary_views_v2.msrb_final`
      WHERE trade_date = CURRENT_DATE('US/Eastern')
    ),
    grouped AS (
      SELECT
        IF(DATE_DIFF(trade_date, dated_date, YEAR) = 0, 'new', 'seasoned') AS issue_type,
        cusip,
        COUNT(*) AS trade_count
      FROM today
      GROUP BY issue_type, cusip
    ),
    ranked AS (
      SELECT
        issue_type, cusip,
        ROW_NUMBER() OVER (PARTITION BY issue_type ORDER BY trade_count DESC) AS rn
      FROM grouped
    )
    SELECT CONCAT(issue_type, '|', cusip)
    FROM ranked
    WHERE rn <= 10
  );

ELSE

END IF;