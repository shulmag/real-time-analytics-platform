"""
This file is for local testing of the update_aaa_yiled_tables function without going through flask.
"""

from update_aaa import update_aaa_yield_tables

print("5y:", update_aaa_yield_tables(5, ["2025-10-08", "2025-10-09"]))
print("10y:", update_aaa_yield_tables(10, ["2025-10-08", "2025-10-09"]))
