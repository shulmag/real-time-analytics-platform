#!/usr/bin/env python3
"""
Simple script: read a single-column CSV of CUSIPs, call the batch pricing API,
and use divide-and-conquer (binary search) to find which CUSIPs cause a batch to fail.

- CSV format: one CUSIP per line (no headers needed; headers are ignored if present).
- All requests use Quantity=100 and Trade Type="S".
- Outputs failing CUSIPs to stdout and writes them to cusips_failed_<timestamp>.csv
"""

import csv
from datetime import datetime
from typing import List, Tuple
import requests

# ----------------------------
# Hard-coded settings (edit these)
# ----------------------------
CSV_PATH  = "/Users/hadassahlurbur/repos/ficc/api_call_templates/all bad one good - Sheet1.csv"     # e.g., "/home/user/cusips.csv"
USERNAME  = "eng@ficc.ai"
PASSWORD  = "1137Work!"
API_URL   = "https://api.ficc.ai/api/batchpricing"

FIXED_QUANTITY   = "100"  # sent for every CUSIP
FIXED_TRADE_TYPE = "S"    # sent for every CUSIP
TIMEOUT_SECONDS  = 60


def read_cusips(path: str) -> List[str]:
    """Read first column from CSV; ignore blank lines and strip quotes/spaces."""
    cusips: List[str] = []
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            c = row[0].strip().strip("'").strip('"')
            if c and c.lower() != "cusip":  # tolerate a header
                cusips.append(c)
    if not cusips:
        raise SystemExit("No CUSIPs found in CSV.")
    return cusips


def call_batch(cusips: List[str]) -> Tuple[bool, dict]:
    """
    Post a batch to the API. Return (ok, payload_or_error).
    ok=False for non-2xx, JSON parse error, explicit {"error": ...}, or length mismatch.
    """
    try:
        resp = requests.post(
            API_URL,
            data={
                "username": USERNAME,
                "password": PASSWORD,
                "cusipList": cusips,
                "quantityList": [FIXED_QUANTITY] * len(cusips),
                "tradeTypeList": [FIXED_TRADE_TYPE] * len(cusips),
            },
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        return False, {"error": f"request_exception: {e}"}

    if not resp.ok:
        return False, {"error": f"http_status_{resp.status_code}", "text": resp.text}

    try:
        payload = resp.json()
    except Exception as e:
        return False, {"error": f"json_parse_error: {e}", "text": resp.text}

    # Treat explicit API error as failure
    if isinstance(payload, dict) and "error" in payload:
        return False, payload

    # If server returns list-like rows, verify row count matches request size
    if isinstance(payload, list) and len(payload) != len(cusips):
        return False, {"error": "row_count_mismatch", "expected": len(cusips), "got": len(payload)}
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        if len(payload["data"]) != len(cusips):
            return False, {"error": "row_count_mismatch", "expected": len(cusips), "got": len(payload["data"])}

    return True, payload


def find_failing_cusips(cusips: List[str]) -> List[str]:
    """
    Binary-search to isolate failing CUSIPs:
      - If whole set passes: no failures.
      - If whole set fails and size==1: that CUSIP fails.
      - Else split in half and recurse on failing halves.
    """
    ok, _ = call_batch(cusips)
    if ok:
        return []
    if len(cusips) == 1:
        return [cusips[0]]

    mid = len(cusips) // 2
    left, right = cusips[:mid], cusips[mid:]

    failing: List[str] = []
    ok_left, _ = call_batch(left)
    if not ok_left:
        failing += find_failing_cusips(left)

    ok_right, _ = call_batch(right)
    if not ok_right:
        failing += find_failing_cusips(right)

    # Deduplicate (preserve order)
    seen = set()
    out: List[str] = []
    for c in failing:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def write_csv(cusips: List[str], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cusip"])
        for c in cusips:
            w.writerow([c])


def main():
    cusips = read_cusips(CSV_PATH)

    # Quick exit if everything passes
    ok_all, _ = call_batch(cusips)
    if ok_all:
        print("All CUSIPs passed batch pricing.")
        print("[]")
        return

    failing = find_failing_cusips(cusips)

    print(f"Found {len(failing)} failing CUSIP(s):")
    for c in failing:
        print(c)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = f"cusips_failed_{ts}.csv"
    write_csv(failing, out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
