-- Create Date: 2025-01-02
-- Last Edit Date: 2025-01-02
-- Description: This must run before `remove usage logs from usage_data table that are older than 7 days to reduce query costs.sql`.
-- Schedule: Every Sunday at 8:00 UTC
INSERT INTO
  `eng-reactor-287421.api_calls_tracker.usage_data_archive`
SELECT
  *
FROM
  `eng-reactor-287421.api_calls_tracker.usage_data` AS usage_data
WHERE
  NOT EXISTS (  --not exists to avoid duplicates
  SELECT
    1  --arbitrary convention for not exists subquery
  FROM
    `eng-reactor-287421.api_calls_tracker.usage_data_archive` AS archive
  WHERE
    archive.user = usage_data.user
    AND archive.time = usage_data.time
    AND archive.quantity = usage_data.quantity
    AND archive.direction = usage_data.direction
    AND archive.cusip = usage_data.cusip   -- 5 conditions to check for matches; this query should be idempotent
  )

-- Quick update re: bigquery usage.  You might have noticed that the BQ costs were way up yesterday.  One cause was that we queried the usage logs frequently, and as that table continues to expand the query gets quite expensive.
-- To remedy this, we have implemented an archiving mechanism that, once a week, removes logs more than a week old and puts them into an archive table.  We use scheduled queries that to do this, the first copies the old logs to the archive table, the second removes the old logs from the usage_data table.   If you want to look at logs older than a week old, head to the archive table (and be aware that it is expensive to query).   The usage data is as before `api_calls_tracker.usage_data`  and the archive table. is `api_calls_tracker.usage_data_archive`.
-- https://console.cloud.google.com/bigquery/scheduled-queries/locations/us/configs/67713ac4-0000-2575-ba33-ac3eb155e2dc/runs?project=eng-reactor-287421
-- https://console.cloud.google.com/bigquery/scheduled-queries/locations/us/configs/6777aa7d-0000-2579-a1af-d4f547eeb0c4/runs?project=eng-reactor-287421