-- Create Date: 2025-04-01
-- Last Edit Date: 2025-04-01
-- Description: This must run after `insert all logs into archived usage data log table (identifies only logs that not yet present).sql`.
-- Schedule: Every Sunday at 9:00 UTC
CREATE OR REPLACE TABLE
  `api_calls_tracker.usage_data_internal` AS
SELECT
  *
FROM
  `api_calls_tracker.usage_data_internal`
WHERE
  DATE(time) > DATE_SUB(CURRENT_DATE(), INTERVAL 31 day)