-- Create Date: 2025-01-02
-- Last Edit Date: 2025-01-02
-- Description: 
-- Schedule: Every day at 9:05 UTC
CREATE OR REPLACE TABLE
  `eng-reactor-287421.auxiliary_views.materialized_trade_history` AS
SELECT
  * EXCEPT (material_event_history, default_event_history)
FROM
  `eng-reactor-287421.primary_views.trade_history_with_reference_data`
WHERE
  trade_date > DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
ORDER BY
  trade_datetime desc
