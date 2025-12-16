#!/usr/bin/env python3
"""
Simple script: use a list of CUSIPs, call the batch pricing API,
and use divide-and-conquer (binary search) to find which CUSIPs cause a batch to fail.

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

# Just add your CUSIPs to this list!
CUSIP_LIST = ['121385T81', '121385T81', '121385U30', '121385U30', '121385U48', '121385U48',
 '121385U55', '121385U55', '121385U63', '121385U63', '121385U71', '121385U71',
 '121385U89', '121385U89', '121385U97', '121385U97', '121385UK2', '121385UK2',
 '121385UN6', '121385UN6', '121385US5', '121385US5', '121385V21', '121385V21',
 '121385V39', '121385V39', '121385V47', '121385V47', '121385V54', '121385V54',
 '121385V62', '121385V62', '121385V70', '121385V70', '121385V88', '121385V88',
 '121385V96', '121385V96', '121385VG0', '121385VG0', '121385VH8', '121385VH8',
 '121385VJ4', '121385VJ4', '121385VK1', '121385VK1', '121385VL9', '121385VL9',
 '121385VM7', '121385VM7', '121385VN5', '121385VN5', '121385VP0', '121385VP0',
 '121385VQ8', '121385VQ8', '121385W20', '121385W20', '121385W38', '121385W38',
 '121385W46', '121385W46', '121385W61', '121385W61', '121385W79', '121385W79',
 '121385W87', '121385W87', '121385W95', '121385W95', '121385X29', '121385X29',
 '121385X37', '121385X37', '121385X45', '121385X45', '121385X52', '121385X52',
 '121385X60', '121385X60', '121385X78', '121385X78', '121385X86', '121385X86',
 '121385X94', '121385X94', '121385XK9', '121385XK9', '121385XL7', '121385XL7',
 '121385XM5', '121385XM5', '121385XN3', '121385XN3', '121385XP8', '121385XP8',
 '121385XQ6', '121385XQ6', '121385XR4', '121385XR4', '121385XS2', '121385XS2',
 '121385XT0', '121385XT0', '121385XU7', '121385XU7', '121385XV5', '121385XV5',
 '121385Y28', '121385Y28', '121385Y36', '121385Y36', '121385Y44', '121385Y44',
 '121385Y51', '121385Y51', '121385Y69', '121385Y69', '121385Y77', '121385Y77',
 '121385Y85', '121385Y85', '121385YG7', '121385YG7', '121385YH5', '121385YH5',
 '121385YJ1', '121385YJ1', '121385YK8', '121385YK8', '121385YL6', '121385YL6',
 '121385YM4', '121385YM4', '121385YN2', '121385YN2', '121385YP7', '121385YP7',
 '121385YQ5', '121385YQ5', '121385YR3', '121385YR3', '121385YS1', '121385YS1',
 '121385Z27', '121385Z27', '121385Z35', '121385Z35', '121385Z43', '121385Z43',
 '121385Z50', '121385Z50', '121385Z68', '121385Z68', '121385Z76', '121385Z76',
 '121385Z84', '121385Z84', '121385Z92', '121385Z92', '121385ZB7', '121385ZB7',
 '121385ZC5', '121385ZC5', '121385ZD3', '121385ZD3', '121385ZE1', '121385ZE1',
 '121385ZF8', '121385ZF8', '121385ZG6', '121385ZG6', '121385ZH4', '121385ZH4',
 '121385ZJ0', '121385ZJ0', '121385ZK7', '121385ZK7', '121385ZL5', '121385ZL5',
 '121385ZM3', '121385ZM3', '121385ZN1', '121385ZN1', '121385ZX9', '121385ZX9',
 '121385ZY7', '121385ZY7', '121385ZZ4', '121385ZZ4', '12138ABH9', '12138ABH9',
 '12138ABJ5', '12138ABJ5', '12138RAB6', '12138RAB6', '12138RAC4', '12138RAC4',
 '12138RAD2', '12138RAD2', '12138TAV8', '12138TAV8', '12138TAW6', '12138TAW6',
 '12138TAX4', '12138TAX4', '12138TAY2', '12138TAY2', '12138TAZ9', '12138TAZ9',
 '12138TBA3', '12138TBA3', '12138TBB1', '12138TBB1', '12138TBC9', '12138TBC9',
 '12138TBD7', '12138TBD7', '12138TBE5', '12138TBE5', '12138TBF2', '12138TBF2',
 '121392CV4', '121392CV4', '121392DG6', '121392DG6', '121392DH4', '121392DH4',
 '121392DJ0', '121392DJ0', '121392DL5', '121392DL5', '121392DM3', '121392DM3',
 '121392DN1', '121392DN1', '121392DP6', '121392DP6', '121392DQ4', '121392DQ4',
 '121392DR2', '121392DR2', '12139LAA0', '12139LAA0', '12139LAB8', '12139LAB8',
 '12139LAC6', '12139LAC6', '1214032E8', '1214032E8', '1214032F5', '1214032F5',
 '1214032G3', '1214032G3', '1214032H1', '1214032H1', '1214032J7', '1214032J7',
 '1214032K4', '1214032K4', '1214032L2', '1214032L2', '1214032M0', '1214032M0',
 '1214032N8', '1214032N8', '1214032P3', '1214032P3', '1214032Q1', '1214032Q1',
 '1214032R9', '1214032R9', '1214032S7', '1214032S7', '1214032T5', '1214032T5',
 '1214032U2', '1214032U2', '1214032V0', '1214032V0', '1214033E7', '1214033E7',
 '1214033F4', '1214033F4', '1214033G2', '1214033G2', '1214033H0', '1214033H0',
 '1214033J6', '1214033J6', '1214033K3', '1214033K3', '1214033L1', '1214033L1',
 '1214033M9', '1214033M9', '1214033N7', '1214033N7', '1214033P2', '1214033P2',
 '1214033Q0', '1214033Q0', '1214033R8', '1214033R8', '1214033S6', '1214033S6',
 '1214033T4', '1214033T4', '1214033U1', '1214033U1', '1214033V9', '1214033V9',
 '1214033W7', '1214033W7', '1214034B2', '1214034B2', '1214034C0', '1214034C0',
 '1214034D8', '1214034D8', '1214034E6', '1214034E6', '1214034F3', '1214034F3',
 '1214034G1', '1214034G1', '1214034H9', '1214034H9', '1214034J5', '1214034J5',
 '1214034M8', '1214034M8', '1214034N6', '1214034N6', '121403G37', '121403G37',
 '121403G45', '121403G45', '121403G52', '121403G52', '121403K99', '121403K99',
 '121403L23', '121403L23', '121403L31', '121403L31', '121403P94', '121403P94',
 '121403Q28', '121403Q28', '121403Q36', '121403Q36', '121403Q44', '121403Q44',
 '121403Q51', '121403Q51', '121403Q69', '121403Q69', '121403Q77', '121403Q77',
 '121403Q85', '121403Q85', '121403Q93', '121403Q93', '121403R27', '121403R27',
 '121403R35', '121403R35', '121403R43', '121403R43', '121403R50', '121403R50',
 '121403R68', '121403R68', '121403R76', '121403R76', '121403R84', '121403R84',
 '121403T90', '121403T90', '121403U23', '121403U23', '121403U31', '121403U31',
 '121403U49', '121403U49', '121403U56', '121403U56', '121403U72', '121403U72',
 '121403U80', '121403U80', '121403U98', '121403U98', '121403UZ0', '121403UZ0',
 '121403V22', '121403V22', '121403V30', '121403V30', '121403V48', '121403V48',
 '121403V55', '121403V55', '121403V63', '121403V63', '121403V89', '121403V89',
 '121403VA4', '121403VA4', '121403VB2', '121403VB2', '121403VC0', '121403VC0',
 '121403VD8', '121403VD8', '121403W88', '121403W88', '121403W96', '121403W96',
 '121403X20', '121403X20', '121403X38', '121403X38', '121403X46', '121403X46',
 '121403X53', '121403X53', '121403X61', '121403X61', '121403X79', '121403X79',
 '121403X87', '121403X87', '121403X95', '121403X95', '121403XY0', '121403XY0',
 '121403Y29', '121403Y29', '121403Y37', '121403Y37', '121403Y45', '121403Y45',
 '121403Y52', '121403Y52', '121403Y60', '121403Y60', '121403Y78', '121403Y78',
 '121403Y86', '121403Y86', '121403Y94', '121403Y94', '121403Z69', '121403Z69',
 '121410PW8', '121410PW8', '121410PX6', '121410PX6', '121410PY4', '121410PY4',
 '121410PZ1', '121410PZ1', '121410QB3', '121410QB3', '121410QD9', '121410QD9',
 '121457EC5', '121457EC5', '121457ED3', '121457ED3', '121457EE1', '121457EE1',
 '121457EF8', '121457EF8', '121457EG6', '121457EG6', '121457EH4', '121457EH4',
 '121457EJ0', '121457EJ0', '121457EK7', '121457EK7', '121457EL5', '121457EL5',
 '121457EM3', '121457EM3', '121457EN1', '121457EN1', '121457EQ4', '121457EQ4',
 '121457FZ3', '121457FZ3', '121457GA7', '121457GA7', '121457GB5', '121457GB5',
 '121457GC3', '121457GC3']

USERNAME  = "eng@ficc.ai"
PASSWORD  = "1137Work!"
API_URL   = "https://api.ficc.ai/api/batchpricing"

FIXED_QUANTITY   = "100"  # sent for every CUSIP
FIXED_TRADE_TYPE = "S"    # sent for every CUSIP
TIMEOUT_SECONDS  = 60


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
    # Clean the CUSIP list (remove empty strings and strip whitespace)
    cusips = [c.strip() for c in CUSIP_LIST if c.strip()]
    
    if not cusips:
        print("No CUSIPs found in CUSIP_LIST.")
        return

    print(f"Testing {len(cusips)} CUSIPs...")

    # Quick exit if everything passes
    ok_all, _ = call_batch(cusips)
    if ok_all:
        print("All CUSIPs passed batch pricing.")
        print("[]")
        return

    failing = find_failing_cusips(cusips)

    print(f"\nFound {len(failing)} failing CUSIP(s):")
    for c in failing:
        print(c)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = f"cusips_failed_{ts}.csv"
    write_csv(failing, out_path)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()