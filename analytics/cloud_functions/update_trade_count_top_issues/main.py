# 2025-09-15
# Last Edited by Developer 
# 2025-09-25 

import logging
from datetime import datetime, date
import functions_framework

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday

from pytz import timezone
from google.cloud import bigquery, storage

from auxiliary_functions import access_secret_version  # uses GCP Secret Manager

# ---------- Config ----------
EASTERN      = timezone("America/New_York")
PROJECT_ID   = "eng-reactor-287421"

# ---------- Email helper (failures only) ----------
FAILURE_EMAIL_RECIPIENTS = ["ficc-eng@ficc.ai"]

def send_email(subject: str, body: str, recipients=FAILURE_EMAIL_RECIPIENTS) -> None:
    import smtplib
    from email.mime.text import MIMEText

    sender_email = access_secret_version("notifications_username", project_id=PROJECT_ID)
    password     = access_secret_version("notifications_password", project_id=PROJECT_ID)

    smtp_server = "smtp.gmail.com"
    port = 587
    server = smtplib.SMTP(smtp_server, port)
    server.starttls()
    server.login(sender_email, password)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipients)

    try:
        server.sendmail(sender_email, recipients, msg.as_string())
    except Exception as e:
        print(e)
    finally:
        server.quit()

# ---------- Holidays / Business Day ----------
class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    """US holiday calendar that includes Good Friday."""
    rules = USFederalHolidayCalendar.rules + [GoodFriday]

BUSINESS_DAY = CustomBusinessDay(calendar=USHolidayCalendarWithGoodFriday())

def is_business_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    cal = USHolidayCalendarWithGoodFriday()
    # Generate a reasonable range around 'd' to avoid annual-boundary bugs
    start = date(d.year - 1, 12, 15)
    end   = date(d.year + 1, 1, 15)
    hols = set(pd.to_datetime(cal.holidays(start=start, end=end)).date)
    return d not in hols

# ---------- SQL Jobs ----------
def upload_top_issues(bq_client: bigquery.Client):
    """
    Updates top issues (by trade count) for 'today' in NY time.
    """
    query = """
    MERGE `eng-reactor-287421.analytics_data_source.minute_top_issues` T
    USING (
      WITH today AS (
        SELECT
          cusip,
          security_description,
          trade_date,
          dated_date
        FROM `eng-reactor-287421.auxiliary_views_v2.msrb_final`
        WHERE trade_date = CURRENT_DATE('America/New_York')
      ),
      grouped AS (
        SELECT
          CURRENT_DATE('America/New_York') AS as_of_date,
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
      INSERT (as_of_date, issue_type, cusip, security_description, trade_count, created_at)
      VALUES (S.as_of_date, S.issue_type, S.cusip, S.security_description, S.trade_count, CURRENT_TIMESTAMP());

    DELETE FROM `eng-reactor-287421.analytics_data_source.minute_top_issues`
    WHERE as_of_date = CURRENT_DATE('America/New_York')
      AND CONCAT(issue_type, '|', cusip) NOT IN (
        WITH today AS (
          SELECT
            cusip,
            trade_date,
            dated_date
          FROM `eng-reactor-287421.auxiliary_views_v2.msrb_final`
          WHERE trade_date = CURRENT_DATE('America/New_York')
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
    """
    job = bq_client.query(query)
    job.result()
    return {"job_id": job.job_id}

def upload_trade_count(bq_client: bigquery.Client):
    """Upserts daily trade/volume summary for today (NY time)."""
    query = """
    MERGE `eng-reactor-287421.analytics_data_source.minute_trade_count` AS T
    USING (
      WITH
        today AS (
          SELECT
            trade_type,
            par_traded,
            trade_date
          FROM `eng-reactor-287421.auxiliary_views_v2.msrb_final`
          WHERE trade_date = CURRENT_DATE('America/New_York')
        ),
        ytd AS (
          SELECT
            trade_date
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
        SAFE_DIVIDE(SUM(IF(trade_type = 'D', par_traded, 0)), NULLIF(COUNTIF(trade_type = 'D'), 0)) AS avg_par_D,

        /* per-type P */
        COUNTIF(trade_type = 'P') AS trades_P,
        COALESCE(SUM(IF(trade_type = 'P', par_traded, 0)), 0) AS volume_P,
        SAFE_DIVIDE(SUM(IF(trade_type = 'P', par_traded, 0)), NULLIF(COUNTIF(trade_type = 'P'), 0)) AS avg_par_P,

        /* per-type S */
        COUNTIF(trade_type = 'S') AS trades_S,
        COALESCE(SUM(IF(trade_type = 'S', par_traded, 0)), 0) AS volume_S,
        SAFE_DIVIDE(SUM(IF(trade_type = 'S', par_traded, 0)), NULLIF(COUNTIF(trade_type = 'S'), 0)) AS avg_par_S,

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
        as_of_date, total_trades, total_volume, avg_par, total_trades_this_year,
        trades_D, volume_D, avg_par_D,
        trades_P, volume_P, avg_par_P,
        trades_S, volume_S, avg_par_S,
        big_trades_over_5mm_dealer, big_trades_over_5mm_buy, big_trades_over_5mm_sell,
        created_at
      )
      VALUES (
        S.as_of_date, S.total_trades, S.total_volume, S.avg_par, S.total_trades_this_year,
        S.trades_D, S.volume_D, S.avg_par_D,
        S.trades_P, S.volume_P, S.avg_par_P,
        S.trades_S, S.volume_S, S.avg_par_S,
        S.big_trades_over_5mm_dealer, S.big_trades_over_5mm_buy, S.big_trades_over_5mm_sell,
        CURRENT_TIMESTAMP()
      );
    """
    job = bq_client.query(query)
    job.result()
    return {"job_id": job.job_id}

# ---------- Entry Point ----------
@functions_framework.http
def main(request=None):
    """
    HTTP Cloud Function entry.
    Runs only on US business days in America/New_York.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("muni-daily-refresh")

    now_et = datetime.now(EASTERN)
    today_et = now_et.date()
    errors = []
    error_occurred = False

    if not is_business_day(today_et):
        msg = f"Non-business day ({today_et}); skipping."
        logger.info(msg)
        return {"status": "skipped", "reason": msg}, 200

    bq_client = None
    try:
        bq_client = bigquery.Client(project=PROJECT_ID)
        storage.Client(project=PROJECT_ID)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.exception(f"Failed to initialize clients: {str(e)}\n{tb}")
        error_occurred = True
        errors.append(f"Failed to initialize clients: {str(e)}\n{tb}")

    # Only run jobs if BigQuery client was successfully created
    if bq_client is not None:
        try:
            trade_job = upload_trade_count(bq_client)
            logger.info(f"Trade count job completed: {trade_job['job_id']}")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Error updating trade_counts: {str(e)}\n{tb}")
            error_occurred = True
            errors.append(f"Error updating trade_counts: {str(e)}\n{tb}")

        try:
            top_job = upload_top_issues(bq_client)
            logger.info(f"Top issues job completed: {top_job['job_id']}")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Error updating muni_top_issues: {str(e)}\n{tb}")
            error_occurred = True
            errors.append(f"Error updating muni_top_issues: {str(e)}\n{tb}")

    # If any errors occurred, send an alert email
    if error_occurred:
        subject = f"[ALERT] Analytics Data Upload ({today_et}): one or more jobs failed"
        body = f"When (ET): {datetime.now(EASTERN).isoformat()}\n\n"
        for e in errors:
            body += f"{e}\n"
        send_email(subject, body)

    return str(today_et), (500 if error_occurred else 200)