"""
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
from google.cloud import bigquery

# Import your model function from your modules package
# Ensure you unzip modules.zip at ./modules and keep this import path intact.
from modules.ficc.utils.nelson_siegel_model import yield_curve_level


# --------- Config ---------

# Source timeline: AAA coefficients, minute-granular (DATETIME)
AAA_TIMELINE = "`eng-reactor-287421.yield_curves_aaa.nelson_siegel_coef_minute`"

# Targets (schema: yield FLOAT64, datetime DATETIME)
TARGET_TABLES: Dict[int, str] = {
    5: "eng-reactor-287421.analytics_data_source.five_year_aaa",
    10: "eng-reactor-287421.analytics_data_source.ten_year_aaa",
}


# --------- Lazy BQ client (avoid import-time failures in serverless) ---------

_BQ: Optional[bigquery.Client] = None


def _bq() -> bigquery.Client:
    global _BQ
    if _BQ is None:
        _BQ = bigquery.Client()
    return _BQ


# --------- Core writer with minute-trunc de-dup & robust logging ---------


def update_aaa_yield_tables(maturity: int, dates: List[str]) -> int:
    """
    For the given maturity (5 or 10) and list of 'YYYY-MM-DD' strings, find all
    minutes in the AAA timeline for those dates that are NOT already present
    in the target table (by MINUTE), compute yields via yield_curve_level, and
    append only the new rows.

    Returns the number of rows written.
    """
    if not dates:
        print(f"[maturity={maturity}] No dates provided.")
        return 0
    if maturity not in TARGET_TABLES:
        raise ValueError("maturity must be 5 or 10")

    target_table = TARGET_TABLES[maturity]

    # Build anti-join **by minute** to avoid second/microsecond mismatches.
    # Also normalize the timeline to MINUTE to guarantee :00 seconds.
    query = f"""
        WITH timeline AS (
          SELECT DATETIME_TRUNC(`date`, MINUTE) AS dt
          FROM {AAA_TIMELINE}
          WHERE DATE(`date`) IN UNNEST(@ds)
          -- Optional: restrict to RTH only
          -- AND TIME(`date`) BETWEEN TIME '09:30:00' AND TIME '16:00:00'
        )
        SELECT t.dt
        FROM timeline t
        LEFT JOIN `{target_table}` y
          ON DATETIME_TRUNC(y.`datetime`, MINUTE) = t.dt
        WHERE y.`datetime` IS NULL
        ORDER BY t.dt ASC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("ds", "DATE", dates)]
    )

    df = _bq().query(query, job_config=job_config).to_dataframe()
    print(f"[maturity={maturity}] anti-join candidates: {len(df)} for dates={dates}")

    if df.empty:
        return 0

    # Ensure pandas datetime (naive), normalized to :00 seconds
    df["dt"] = pd.to_datetime(df["dt"]).dt.floor("T")

    # Compute predictions, surfacing model errors but continuing
    preds: list[float] = []
    dts: list[datetime] = []
    for ts in df["dt"]:
        dt_py = ts.to_pydatetime().replace(second=0, microsecond=0)  # ensure :00
        try:
            y = float(yield_curve_level(maturity, dt_py))
            preds.append(y)
            dts.append(dt_py)
        except Exception as e:
            print(f"[maturity={maturity}] MODEL ERROR at {dt_py}: {type(e).__name__}: {e}")
            # skip this minute; continue with others

    rows_written = len(preds)
    if rows_written == 0:
        print(f"[maturity={maturity}] 0 predictions; skipping load.")
        return 0

    out = pd.DataFrame(
        {
            "yield": preds,  # FLOAT64
            "datetime": dts,  # DATETIME (naive, :00 seconds)
        }
    )

    load_cfg = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=[
            bigquery.SchemaField("yield", "FLOAT"),
            bigquery.SchemaField("datetime", "DATETIME"),
        ],
    )
    job = _bq().load_table_from_dataframe(out, target_table, job_config=load_cfg)
    result = job.result()  # wait
    if job.errors:
        print(f"[maturity={maturity}] LOAD ERRORS: {job.errors}")
        raise RuntimeError(job.errors)

    print(f"[maturity={maturity}] Wrote {rows_written} new rows to {target_table}.")
    return rows_written
