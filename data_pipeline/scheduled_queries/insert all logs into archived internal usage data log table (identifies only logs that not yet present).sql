-- Create Date: 2025-04-01
-- Last Edit Date: 2025-04-01
-- Description: This must run before `remove usage logs from usage_data table that are older than 31 days to reduce query costs.sql`.
-- Schedule: Every Sunday at 8:00 UTC
INSERT INTO
  `eng-reactor-287421.api_calls_tracker.usage_data_internal_archive`
SELECT
  *
FROM
  `eng-reactor-287421.api_calls_tracker.usage_data_internal` AS usage_data
WHERE
  NOT EXISTS (  --not exists to avoid duplicates
  SELECT
    1  --arbitrary convention for not exists subquery
  FROM
    `eng-reactor-287421.api_calls_tracker.usage_data_internal_archive` AS archive
  WHERE
    archive.user = usage_data.user
    AND archive.time = usage_data.time
    AND archive.quantity = usage_data.quantity
    AND archive.direction = usage_data.direction
    AND archive.cusip = usage_data.cusip   -- 5 conditions to check for matches; this query should be idempotent
  )
