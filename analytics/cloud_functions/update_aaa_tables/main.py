"""
Main entry point for update_aaa_tables Cloud Function.
"""

from __future__ import annotations

from datetime import datetime
import json

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
from zoneinfo import ZoneInfo

from update_aaa import update_aaa_yield_tables


class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    """US Federal holiday calendar extended with Good Friday."""

    rules = USFederalHolidayCalendar.rules + [GoodFriday]


BUSINESS_DAY = CustomBusinessDay(calendar=USHolidayCalendarWithGoodFriday())


def _requested_dates_et() -> list[str]:
    """
    Returns [previous_business_day_ET, today_ET] as ISO 'YYYY-MM-DD' strings,
    using US Federal holidays + Good Friday.
    """
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    prev_bd = (pd.Timestamp(today_et) - BUSINESS_DAY).date()
    return [prev_bd.isoformat(), today_et.isoformat()]


def main(request):
    """
    Cloud Functions (Gen 2) HTTP entry point.

    GET/POST -> runs the writer for [prev_bd, today] (ET) and returns a JSON summary.
    """
    try:
        dates = _requested_dates_et()
        print(f"[entry] dates={dates}")

        r5 = update_aaa_yield_tables(5, dates)
        r10 = update_aaa_yield_tables(10, dates)

        body = {
            "dates": dates,
            "written_rows": {"5y": r5, "10y": r10},
            "status": "ok",
        }
        return (json.dumps(body), 200, {"Content-Type": "application/json"})

    except Exception as e:
        body = {"status": "error", "message": f"{type(e).__name__}: {e}"}
        # Log full error to stderr for Cloud Logging
        print(f"[ERROR] {type(e).__name__}: {e}")
        return (json.dumps(body), 500, {"Content-Type": "application/json"})
