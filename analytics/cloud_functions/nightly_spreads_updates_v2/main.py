# Nightly Updates

"""
Cloud Function to update bond spread data nightly.
This function should be scheduled to run every night at a specific time.
Last Edited Date: 2025-10-21
"""

import os
import functions_framework
from google.cloud import bigquery, storage
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
from auxiliary_functions import access_secret_version
from pytz import timezone
import logging
import traceback
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FAILURE_EMAIL_RECIPIENTS = ["ficc-eng@ficc.ai"]
SUCCESS_EMAIL_RECIPIENTS = FAILURE_EMAIL_RECIPIENTS

# Import the functions from spreads.py
from spreadsV2 import (
    get_last_n_business_days,
    get_existing_data,
    process_missing_days_historical_file,
    process_missing_days_batch_pricing,
    USHolidayCalendarWithGoodFriday,
)

# Constants
PROJECT_ID = "eng-reactor-287421"
EASTERN = timezone("US/Eastern")


def send_email(subject, message, recipients=FAILURE_EMAIL_RECIPIENTS):
    import smtplib  # lazy loading for lower latency
    from email.mime.text import MIMEText  # lazy loading for lower latency

    sender_email = access_secret_version("notifications_username")
    password = access_secret_version("notifications_password")
    smtp_server = "smtp.gmail.com"
    port = 587

    server = smtplib.SMTP(smtp_server, port)
    server.starttls()
    server.login(sender_email, password)

    message = MIMEText(message)
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = ", ".join(recipients)

    try:
        server.sendmail(sender_email, recipients, message.as_string())
    except Exception as e:
        print(e)
    server.quit()


@functions_framework.http
def update_spreads(request):
    """
    HTTP Cloud Function to update bond spread data for:
      maturities: ['4.5-5.5', '9.5-10.5']
      quantities: [1000, 100]

    Behavior
    --------
    - qty = 1000: backfill the last N business days from InvestorTools GCS files
                  (uses `process_missing_days_historical_file`).
    - qty = 100 : ensure the most recent business day is present by
                  calling batch pricing once (uses `process_missing_days_batch_pricing`).
    Responses
    ---------
    - 200 OK with a JSON summary if all maturity/quantity combos succeed.
    - 500 ERROR with a JSON summary if any combo fails; the summary includes per-combo
      errors so you can see what went wrong while still allowing partial success.

    """
    maturities = ["4.5-5.5", "9.5-10.5"]
    quantities = [1000, 100]

    # Initialize clients
    try:
        bq_client = bigquery.Client(project=PROJECT_ID)
        storage_client = storage.Client(project=PROJECT_ID)
    except Exception as e:
        logger.exception("Failed to initialize GCP clients")
        return {"error": f"Client init failed: {e}"}, 500

    # This will run at 6PM EST
    run_time_et = datetime.now(EASTERN)
    logger.info(f"Running spreads update at {run_time_et.isoformat()}")

    # Inputs / config
    try:
        business_days = get_last_n_business_days(10)
    except Exception as e:
        logger.exception("Failed to compute business days")
        return {"error": f"Business day calculation failed: {e}"}, 500

    if not business_days:
        return {"error": "No business days returned by helper."}, 500
    most_recent_bd = business_days[0]

    results = []
    errors = []
    error_occurred = False

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [
            ex.submit(
                update_single_spread,
                m,
                q,
                bq_client=bq_client,
                storage_client=storage_client,
                business_days=business_days,
                most_recent_bd=most_recent_bd,
            )
            for m, q in product(maturities, quantities)
        ]
        for f in as_completed(futures):
            try:
                result, error = f.result()
                if result is not None:
                    results.append(result)
                if error is not None:
                    errors.append(error)
                    error_occurred = True
            except Exception:
                # defensive catch if a worker threw outside its try/except
                logger.exception("Unhandled exception surfaced from worker")
                error_occurred = True
                errors.append({"error": "unhandled worker exception"})

    # single email if anything failed
    if errors:
        subject = "[ALERT] Nightly Spreads Update Failures"
        pretty = "\n".join(
            f"- maturity={e.get('maturity')}, qty={e.get('quantity')}: {e.get('error')}"
            for e in errors
        )
        message = (
            f"Run time ET: {run_time_et.isoformat()}\n\n"
            f"Failures ({len(errors)}):\n{pretty}\n\n"
            f"Full results:\n{json.dumps(results, indent=2, default=str)}"
        )
        send_email(subject, message)
    summary = {
        "status": "error" if error_occurred else "success",
        "run_time_et": run_time_et.isoformat(),
        "business_days_checked": [d.strftime("%Y-%m-%d") for d in business_days],
        "most_recent_business_day": most_recent_bd.strftime("%Y-%m-%d"),
        "total_processed": sum(r.get("processed_days", 0) for r in results if isinstance(r, dict)),
        "detail": results,
    }

    return summary, 500 if error_occurred else 200


def update_single_spread(
    maturity, quantity, *, bq_client, storage_client, business_days, most_recent_bd
):
    try:
        if quantity == 1000:
            source = "gcp_file"
            logger.info(
                f"[Maturity:{maturity} | Quantity:{quantity}] Checking backfill from GCS files"
            )

            existing_df, existing_dates = get_existing_data(
                bq_client, business_days, maturity, quantity
            )
            missing_dates = [d for d in business_days if d not in existing_dates]

            if not missing_dates:
                logger.info(f"[Maturity:{maturity} | Quantity:{quantity}] Nothing missing")
                result = {
                    "maturity": maturity,
                    "quantity": quantity,
                    "checked_days": len(business_days),
                    "missing_days": 0,
                    "processed_days": 0,
                }
                return result, None

            processed = process_missing_days_historical_file(
                bq_client, storage_client, missing_dates, maturity
            )
            logger.info(f"[Maturity:{maturity} | Quantity:{quantity}] processed={processed}")

            result = {
                "maturity": maturity,
                "quantity": quantity,
                "checked_days": len(business_days),
                "missing_days": len(missing_dates),
                "processed_days": processed,
            }
            return result, None

        else:
            source = "ficc_batch_pricing"
            logger.info(
                f"[Maturity:{maturity} | Quantity:{quantity}] Ensuring most recent day {most_recent_bd} exists"
            )

            existing_df, existing_dates = get_existing_data(
                bq_client, [most_recent_bd], maturity, quantity
            )

            if most_recent_bd in existing_dates:
                logger.info(
                    f"[ Maturity:{maturity} | Quantity:{quantity}] Already populated for {most_recent_bd}"
                )
                result = {
                    "maturity": maturity,
                    "quantity": quantity,
                    "checked_days": 1,
                    "missing_days": 0,
                    "processed_days": 0,
                }
                return result, None  # <-- early return instead of 'continue'

            processed = process_missing_days_batch_pricing(bq_client, quantity, source, maturity)
            logger.info(f"[{maturity} | {quantity}] Processed via batch pricing: {processed}")

            result = {
                "maturity": maturity,
                "quantity": quantity,
                "checked_days": 1,
                "missing_days": 1 if processed > 0 else 0,
                "processed_days": processed,
            }
            return result, None

    except Exception as e:
        logger.exception(f"Error for maturity={maturity}, quantity={quantity}")
        error = {
            "maturity": maturity,
            "quantity": quantity,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        return None, error


# For running locally or via Cloud Scheduler
if __name__ == "__main__":
    # Simulate an HTTP request for local testing
    class MockRequest:
        pass

    result = update_spreads(MockRequest())
    print(result)
