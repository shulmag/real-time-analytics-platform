'''
Main entry point for ficc analytics API.
This module handles API routing, authentication, and core business logic for yield curve data.
READ-ONLY - for external data, such as Redis and BigQuery.
'''

# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser
from flask import jsonify, make_response, request
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
from pytz import timezone
from google.cloud import bigquery
from FiccYieldCurve import predict_ytw
from time_series_functions import intraday_open_close, get_overnight_change
from auxiliary_functions import date_format, KEY_MATURITIES
import redis_helper
import firebase
import logging
import firebase_admin
from firebase_admin import auth as admin_auth
from analytics_tracking import log_analytics_usage
from datetime import datetime
from auxiliary_functions import get_last_n_business_days
from auxiliary_functions import get_existing_data as get_existing_data_v2

# Calendar and timezone constants

class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    """Custom US Federal Holiday calendar that includes Good Friday"""
    rules = USFederalHolidayCalendar.rules + [GoodFriday]

BUSINESS_DAY = CustomBusinessDay(calendar=USHolidayCalendarWithGoodFriday())
EASTERN = timezone('America/New_York')

# Determines Smooth Window of AAA Benchmark
DEFAULT_SMOOTH_MINUTES = 5

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize BigQuery client
try:
    BQ_CLIENT = bigquery.Client()
except Exception:
    logger.warning("Failed to initialize BigQuery client, analytics tracking will be disabled")
    BQ_CLIENT = None

try:
    if not firebase_admin._apps:
        firebase_admin.initialize_app()   # uses default service account in Cloud Run/Functions
        logger.info("[auth] firebase_admin initialized")
except Exception as e:
    logger.warning(f"[auth] firebase_admin init failed: {e}")



def last_business_day(date_eastern):
    """
    Return the previous business day (skips weekends & US holidays).
    
    Args:
        date_eastern: A datetime object in Eastern timezone
        
    Returns:
        The previous business day as a date object
    """
    prev_day = (date_eastern - BUSINESS_DAY).date()
    return prev_day

def parse_date(date_str):
    """
    Parse date string to datetime object with Eastern timezone.
    
    Args:
        date_str: A date string in a format accepted by dateutil.parser
        
    Returns:
        A datetime object in Eastern timezone or None if parsing fails
    """
    if not date_str:
        logger.info("No date string provided")
        return None
    
    try:
        date = parser.parse(date_str)
        if date.tzinfo is None:
            date = EASTERN.localize(date)
        return date
    except Exception as e:
        logger.error(f"Failed to parse date '{date_str}': {str(e)}")
        return None

def cors_headers():
    """
    Return CORS headers for API responses.
    
    Returns:
        Dictionary of CORS headers for cross-origin requests
    """
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Access-Control-Max-Age': '3600'
    }


def main(request):
    """
    Main entry point for the ficc analytics API.
    
    Supported endpoints:
    - /api/yield-curves: Get yield curve data
    - /api/market-metrics: Get market strength metrics based on MSRB trade data
    - /api/muni-market-stats
    - /api/muni-market-stats-10d
    - /api/muni-top-issues  
    
    Query parameters:
    - type: 'daily' or 'realtime' (default: 'realtime')
    - start_date: Start date in ISO format (default: previous business day)
    - end_date: End date in ISO format (default: current date)
    - maturities: Comma-separated list of maturities (e.g., '1,2,5,10')
    - refresh: 'true' to indicate a refresh operation (used for analytics tracking)
    
    Authentication:
    - Requires Firebase authentication token in the Authorization header
    - Format: 'Bearer <token>'
    
    Returns:
        Flask response with JSON data or error message
    """
    
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        logger.debug("Handling CORS preflight request")
        return make_response('', 204, cors_headers())
    
    # Start timing for analytics tracking
    start_time = datetime.now()
    
    # Initialize user_email to None (will be set if token verification succeeds)
    user_email = None
    
    # Extract and verify Firebase token if present
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        try:
            # Verify the token and get user info
            decoded_token = firebase.verify_firebase_token(token)
            if decoded_token:
                user_email = decoded_token.get('email')
                logger.info(f"Request authenticated for user: {user_email}")
        except Exception as e:
            logger.warning(f"Token verification failed: {str(e)}")
    
    # Get request path and parameters
    request_json = request.get_json(silent=True)
    request_args = request.args
    
    # Check if this is a refresh action (client sends a refresh parameter)
    action = "refresh" if request_args.get('refresh', '').lower() == 'true' else "normal load"
    
    # Get the API path from URL or post data
    path = request.path.strip('/').lower() if request.path else None
    if path and path.startswith('api/'):
        path = path[4:]  # Remove 'api/' prefix
    
    if request_json and 'path' in request_json:
        path = request_json['path'].strip('/').lower()
    
    if not path:
        return make_response(jsonify({
            'status': 'error',
            'message': 'No API endpoint specified'
        }), 400, cors_headers())
    
    # Extract common parameters
    if request_json:
        curve_type = request_json.get('type', 'realtime')
        start_date_str = request_json.get('start_date')
        end_date_str = request_json.get('end_date')
        maturities_str = request_json.get('maturities')
    else:
        curve_type = request_args.get('type', 'realtime')
        start_date_str = request_args.get('start_date')
        end_date_str = request_args.get('end_date')
        maturities_str = request_args.get('maturities')
    
    # Parse dates
    current_datetime = datetime.now(tz=EASTERN)
    
    if not end_date_str:
        logger.debug("No end_date provided, using current datetime")
        end_date = current_datetime
    else:
        end_date = parse_date(end_date_str)
        if not end_date:
            logger.warning(f"Failed to parse end_date '{end_date_str}', using current datetime")
            end_date = current_datetime
    
    start_date = last_business_day(end_date)
    maturities = KEY_MATURITIES
    
    # Route to the appropriate handler based on the path
    try:
        response = None
        error_occurred = False
        
        if path == 'yield-curves':
            response = handle_yield_curves(curve_type, start_date, end_date, maturities)
        elif path == 'spreads':
        # Optional params (query string or JSON)
            days_str = (request_json or {}).get('days') if request_json else request_args.get('days')
            mparam   = (request_json or {}).get('maturities') if request_json else request_args.get('maturities')
            qparam   = (request_json or {}).get('quantities') if request_json else request_args.get('quantities')

            days = int(days_str) if days_str else 15
            maturities = [s.strip() for s in mparam.split(',')] if mparam else ['4.5-5.5', '9.5-10.5']
            quantities = [int(x) for x in qparam.split(',')] if qparam else [1000, 100]

            response = handle_spreads(days=days, maturities=maturities, quantities=quantities)

        elif path == 'market-metrics':
            response = handle_market_metrics(end_date)
        elif path == 'muni-market-stats':
            response = handle_muni_market_stats(end_date)
        elif path == 'muni-market-stats-10d':
            response = handle_muni_market_stats_10d()
        elif path == 'muni-top-issues':
            response = handle_muni_top_issues()
        elif path == 'aaa-benchmark':
            response = handle_aaa_benchmark()
        else:
            response = make_response(jsonify({
                'status': 'error',
                'message': f'Unknown API endpoint: {path}'
            }), 404, cors_headers())
            error_occurred = True
        
        # Log analytics usage
        component = f"Analytics {path}"
        log_analytics_usage(
            BQ_CLIENT, 
            user_email, 
            component, 
            action, 
            start_time, 
            error_occurred,
            request
        )
        
        return response
    except Exception as e:
        import traceback
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log error to analytics
        component = f"Analytics {path}"
        log_analytics_usage(
            BQ_CLIENT, 
            user_email, 
            component, 
            action, 
            start_time, 
            True,  # Error occurred
            request
        )
        
        return make_response(jsonify({
            'status': 'error',
            'message': f'Error processing request: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500, cors_headers())

def handle_yield_curves(curve_type, start_date, end_date, maturities):
    """
    Handle requests for yield curve data using Redis and FiccYieldCurve.
    
    Args:
        curve_type: Type of curve data ('daily' or 'realtime')
        start_date: Start date for data range
        end_date: End date for data range
        maturities: List of maturities to include
        
    Returns:
        Flask response with JSON data or error message
    """
    logger.info(f"Processing yield curves request: type={curve_type}")
    
    # Set end date to today's date
    today_date = datetime.now(EASTERN)
    # Set start date to the last business day
    prev_business_day = last_business_day(today_date)
    
    # --- START: NEW LOGIC ---
    # Check if today is a business day using the calendar defined at the top of the file
    is_today_business_day = not pd.bdate_range(
        start=today_date.date(), 
        end=today_date.date(), 
        freq=BUSINESS_DAY
    ).empty

    # If today is not a business day, set the query's end date to the previous business day
    # to effectively exclude today's data from the results.
    if is_today_business_day:
        query_end_date = today_date
        logger.info(f"Today is a business day. Querying data up to {today_date.strftime('%Y-%m-%d')}.")
    else:
        query_end_date = prev_business_day
        logger.info(f"Today is not a business day. Querying data up to {prev_business_day.strftime('%Y-%m-%d')}.")
    # --- END: NEW LOGIC ---

    start_date_str = prev_business_day.strftime('%Y-%m-%d')
    # Use the adjusted end date for the response payload
    end_date_str = query_end_date.strftime('%Y-%m-%d')
    
    # Try to get yield curve data from Redis
    try:
        # Get Nelson-Siegel parameters from Redis, using the adjusted end date
        redis_data = redis_helper.get_yield_curves(
            maturities=maturities, 
            prev_business_day=prev_business_day, 
            today_date=query_end_date  # <-- USE THE ADJUSTED DATE HERE
        )
        if not redis_data:
            logger.error("No yield curve data found in Redis")
            return make_response(jsonify({
                'status': 'error',
                'message': 'No yield curve data found in Redis',
                'details': 'Make sure the Redis bastion host is running and SSH tunnel is established'
            }), 500, cors_headers())
        
        logger.info(f"Retrieved {len(redis_data)} timestamps from Redis")
        
        # Use numpy array for efficient computation with all maturities at once
        maturities_array = np.array(maturities)
        
        # Calculate yield values for each timestamp using vectorized operations
        results = []
        for data_point in redis_data:
            timestamp = data_point['timestamp']
            params = data_point['params']
            
            # Vectorized calculation for all maturities at once
            yield_values = predict_ytw(
                maturity=maturities_array,
                const=params['const'],
                exponential=params['exponential'],
                laguerre=params['laguerre'],
                exponential_mean=params['exponential_mean'],
                exponential_std=params['exponential_std'],
                laguerre_mean=params['laguerre_mean'],
                laguerre_std=params['laguerre_std'],
                shape_parameter=params['shape_parameter']
            )
            
            # Create a data point for this timestamp with all maturity values
            yield_point = {
                'timestamp': timestamp,
                'values': {str(maturities[i]): float(yield_values[i]) for i in range(len(maturities))}
            }
            
            results.append(yield_point)
        
        # Return the result
        return make_response(jsonify({
            'status': 'success',
            'type': curve_type,
            'source': 'redis',
            'start_date': start_date_str,
            'end_date': end_date_str,
            'maturities': [int(m) for m in maturities],
            'data': results
        }), 200, cors_headers())
    except Exception as e:
        import traceback
        logger.error(f"Error calculating yield curves: {str(e)}")
        logger.error(traceback.format_exc())
        # If there's an error, return it
        return make_response(jsonify({
            'status': 'error',
            'message': f'Error calculating yield curves: {str(e)}',
            'traceback': traceback.format_exc(),
            'details': 'Error occurred while processing Redis data with FiccYieldCurve'
        }), 500, cors_headers())

def handle_market_metrics(date_eastern):
    """
    Handle requests for market strength metrics based on MSRB trade data.
    
    Args:
        date_eastern: Date for which to fetch market metrics
        
    Returns:
        Flask response with JSON data or error message
    """
    logger.info("Processing market metrics request")
    today = date_eastern.date()
    prev_day = last_business_day(date_eastern)
    
    query = f"""
        WITH trades AS (
          SELECT *
          FROM `eng-reactor-287421.auxiliary_views_v2.msrb_final`
          WHERE trade_date IN ('{today}', '{prev_day}')
        ),
        daily AS (
          SELECT
              trade_date,
              /*  ≥5MM  */
              COUNTIF(trade_type='S' AND is_trade_with_a_par_amount_over_5MM)           AS inst_buy_cnt,
              COUNTIF(trade_type='P' AND is_trade_with_a_par_amount_over_5MM)           AS inst_sell_cnt,
              COALESCE(SUM(CASE WHEN trade_type='S' AND is_trade_with_a_par_amount_over_5MM
                       THEN par_traded END), 0)                                         AS inst_buy_vol,
              COALESCE(SUM(CASE WHEN trade_type='P' AND is_trade_with_a_par_amount_over_5MM
                       THEN par_traded END), 0)                                         AS inst_sell_vol,
              /*  < 1MM  */
              COUNTIF(trade_type='S' AND par_traded < 1000000)                        AS retail_buy_cnt,
              COUNTIF(trade_type='P' AND par_traded < 1000000)                        AS retail_sell_cnt,
              COALESCE(SUM(CASE WHEN trade_type='S' AND par_traded < 1000000
                       THEN par_traded END), 0)                                         AS retail_buy_vol,
              COALESCE(SUM(CASE WHEN trade_type='P' AND par_traded < 1000000
                       THEN par_traded END), 0)                                         AS retail_sell_vol
          FROM trades
          GROUP BY trade_date
        )
        SELECT * FROM daily
    """
    
    try:
        raw = BQ_CLIENT.query(query).result().to_dataframe()
        # collapse any double rows → at most one per date
        raw = raw.groupby("trade_date", as_index=False).first()
        raw["trade_date"] = raw["trade_date"].astype(str)   # always safe
        df = (
            raw.set_index("trade_date")
            .reindex([str(prev_day), str(today)], fill_value=0)
        )
        
        logger.debug("Data frame prepared")
        
        # ---------------- helper -----------------
        def block(day_row, vol_divisor=1):
            """Return counts **and** volumes ($millions)."""
            try:
                buys_cnt  = int(day_row.get("buy_cnt", 0) or 0)
                sells_cnt = int(day_row.get("sell_cnt", 0) or 0)
                buys_vol  = float(day_row.get("buy_vol", 0) or 0)  / vol_divisor
                sells_vol = float(day_row.get("sell_vol", 0) or 0) / vol_divisor
                
                return {
                    "buys": buys_cnt,
                    "sells": sells_cnt,
                    "buyVol":  round(buys_vol, 1),
                    "sellVol": round(sells_vol, 1),
                    "buyVsSellRatio": round(buys_cnt / max(sells_cnt, 1), 2),
                    "netFlow": round(buys_vol - sells_vol, 1),
                }
            except (ValueError, TypeError, ZeroDivisionError) as e:
                logger.warning(f"Error processing block: {e}")
                return {
                    "buys": 0,
                    "sells": 0,
                    "buyVol": 0,
                    "sellVol": 0,
                    "buyVsSellRatio": 0,
                    "netFlow": 0,
                }
        
        # ---- build today / yesterday blocks ----
        inst_today   = block(df.loc[str(today)].rename(lambda c: c.replace("inst_", "")))
        inst_yday    = block(df.loc[str(prev_day)].rename(lambda c: c.replace("inst_", "")))
        retail_today = block(df.loc[str(today)].rename(lambda c: c.replace("retail_", "")), 1_000_000)
        retail_yday  = block(df.loc[str(prev_day)].rename(lambda c: c.replace("retail_", "")), 1_000_000)
        
        # Add yesterday data directly to the response object
        # First, ensure we have valid dictionaries
        if not isinstance(inst_today, dict):
            inst_today = {}
        if not isinstance(inst_yday, dict):
            inst_yday = {}
        if not isinstance(retail_today, dict):
            retail_today = {}
        if not isinstance(retail_yday, dict):
            retail_yday = {}
        
        # Manually add yesterday's data to today's objects
        # CRITICAL: Use consistent naming for all fields
        inst_today["yesterdayBuys"] = inst_yday.get("buys", 0) 
        inst_today["yesterdaySells"] = inst_yday.get("sells", 0)
        inst_today["yesterdayBuyVol"] = inst_yday.get("buyVol", 0)
        inst_today["yesterdaySellVol"] = inst_yday.get("sellVol", 0)
        inst_today["buyVsSellRatioChange"] = round(inst_today.get("buyVsSellRatio", 0) - inst_yday.get("buyVsSellRatio", 0), 2)
        inst_today["netFlowChange"] = round(inst_today.get("netFlow", 0) - inst_yday.get("netFlow", 0), 1)
        
        retail_today["yesterdayBuys"] = retail_yday.get("buys", 0)
        retail_today["yesterdaySells"] = retail_yday.get("sells", 0)
        retail_today["yesterdayBuyVol"] = retail_yday.get("buyVol", 0)
        retail_today["yesterdaySellVol"] = retail_yday.get("sellVol", 0)
        retail_today["buyVsSellRatioChange"] = round(retail_today.get("buyVsSellRatio", 0) - retail_yday.get("buyVsSellRatio", 0), 2)
        retail_today["netFlowChange"] = round(retail_today.get("netFlow", 0) - retail_yday.get("netFlow", 0), 1)
        
        logger.info("Market metrics calculated successfully")
        
        return make_response(
            jsonify({
                "status": "success",
                "date": str(today),
                "data": {
                    "marketStrength": inst_today,
                    "retailStrength": retail_today
                }
            }),
            200,
            cors_headers()
        )
        
    except Exception as e:
        logger.error(f"Error in handle_market_metrics: {str(e)}")
        return make_response(
            jsonify({
                "status": "error",
                "message": "Error processing market metrics",
                "date": str(today),
                "data": {
                    "marketStrength": {
                        "buys": 0, "sells": 0, "buyVol": 0, "sellVol": 0,
                        "buyVsSellRatio": 0, "netFlow": 0,
                        "yesterdayBuys": 0, "yesterdaySells": 0,
                        "yesterdayBuyVol": 0, "yesterdaySellVol": 0,
                        "buyVsSellRatioChange": 0, "netFlowChange": 0
                    },
                    "retailStrength": {
                        "buys": 0, "sells": 0, "buyVol": 0, "sellVol": 0,
                        "buyVsSellRatio": 0, "netFlow": 0,
                        "yesterdayBuys": 0, "yesterdaySells": 0,
                        "yesterdayBuyVol": 0, "yesterdaySellVol": 0,
                        "buyVsSellRatioChange": 0, "netFlowChange": 0
                    }
                }
            }),
            200,
            cors_headers()
        )
    
def handle_spreads(days=15, maturities=None, quantities=None):
    """
    Handle requests for investment grade bond spread data.
    Fetches historical spread data from BigQuery for the requested
    maturity–quantity pairs over the given business-day window.
    """
    logger.info("Processing spreads request")

    try:
        # Defaults match prior behavior
        maturities = maturities or ['4.5-5.5', '9.5-10.5']
        quantities = quantities or [1000, 100]

        # Build the date window (this used to be fixed at 15)
        business_days = get_last_n_business_days(days)

        frames = []
        for m in maturities:
            for q in quantities:
                # v2 helper accepts (bq_client, business_days, maturity, quantity)
                df, _ = get_existing_data_v2(BQ_CLIENT, business_days, m, q)
                if df is not None and not df.empty:
                    df = df.copy()
                    df['MaturityRange'] = m
                    df['Quantity'] = q
                    frames.append(df)

        if not frames:
            logger.warning("No spread data found in BigQuery for requested filters")
            return make_response(jsonify({
                'status': 'error',
                'message': 'No spread data available'
            }), 404, cors_headers())

        # Combine all pairs and sort by latest first (same as before)
        spreads_df = (
            pd.concat(frames, ignore_index=True)
              .sort_values('date', ascending=False)
        )

        # Shape JSON; keep existing fields and add maturityRange/quantity
        data = []
        for _, row in spreads_df.iterrows():
            item = {
                'date': row['date'],
                'avgSpreadDollar': row['AvgSpreadDollar'],
                'avgSpreadPercent': row['AvgSpreadPercent'],
                'numCusips': int(row['NumCUSIPs']),
                'maturityRange': row['MaturityRange'],
                'quantity': int(row['Quantity']),
            }
            # Preserve sourceFile if present in your schema
            if 'SourceFile' in row:
                item['sourceFile'] = row['SourceFile']
            elif 'source' in row:
                item['sourceFile'] = row['source']
            data.append(item)

        return make_response(jsonify({
            'status': 'success',
            'data': data,
            'count': len(data)
        }), 200, cors_headers())

    except Exception as e:
        logger.error(f"Error in handle_spreads: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return make_response(jsonify({
            'status': 'error',
            'message': f'Error fetching spread data: {str(e)}'
        }), 500, cors_headers())

def handle_muni_market_stats(date_eastern):
    """
    Handle requests for Muni Market Stats (three cards) based on recap tables.
    Returns JSON with today's totals, YTD count, most active issues, and customer flow.
    """
    logger.info("Processing muni market stats request")
    if BQ_CLIENT is None:
        return make_response(jsonify({'status': 'error','message': 'BigQuery client not initialized'}), 500, cors_headers())

    query = """
        DECLARE tz STRING DEFAULT 'America/New_York';
        DECLARE today DATE DEFAULT CURRENT_DATE(tz);

        WITH daily AS (
          SELECT
            total_trades            AS total_trades_today,
            total_volume            AS total_volume_today,
            avg_par                 AS avg_par_today,
            trades_P                AS customer_bought_trades_today,
            trades_S                AS customer_sold_trades_today,
            total_trades_this_year
          FROM `eng-reactor-287421.analytics_data_source.minute_trade_count`
          WHERE as_of_date = today
          LIMIT 1
        ),
        top AS (
          SELECT issue_type, cusip, security_description, trade_count
          FROM `eng-reactor-287421.analytics_data_source.minute_top_issues`
          WHERE as_of_date = today
        ),
        seasoned AS (
          SELECT
            cusip  AS seasoned_cusip,
            security_description AS seasoned_desc,
            trade_count AS seasoned_trades
          FROM top WHERE issue_type = 'seasoned' LIMIT 1
        ),
        newissued AS (
          SELECT
            cusip  AS new_cusip,
            security_description AS new_desc,
            trade_count AS new_trades
          FROM top WHERE issue_type = 'new' LIMIT 1
        )
        SELECT
          d.*,
          s.seasoned_cusip, s.seasoned_desc, s.seasoned_trades,
          n.new_cusip, n.new_desc, n.new_trades
        FROM daily d
        LEFT JOIN seasoned s ON TRUE
        LEFT JOIN newissued n ON TRUE
    """

    try:
        start = pd.Timestamp.now(tz=EASTERN)
        rows = BQ_CLIENT.query(query).result()
        data = [dict(row) for row in rows]
        payload = data[0] if data else {}
        return make_response(jsonify(payload), 200, cors_headers())
    except Exception as e:
        logger.error(f"Error in handle_muni_market_stats: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return make_response(jsonify({'status': 'error','message': f'Error fetching muni market stats: {str(e)}'}), 500, cors_headers())
    
def handle_muni_market_stats_10d():
    """
    Return last 10 trading days of recap_daily as stacked counts by D / S / P.
    Shape:
    [
      { "date": "2025-08-04", "D": 410, "S": 350, "P": 520, "total": 1280 },
      ...
    ]
    """
    logger.info("Processing muni market stats 10d request")
    if BQ_CLIENT is None:
        return make_response(jsonify({"status":"error","message":"BigQuery client not initialized"}), 500, cors_headers())

    query = """
      DECLARE tz STRING DEFAULT 'America/New_York';
      DECLARE today DATE DEFAULT CURRENT_DATE(tz);

      WITH last10 AS (
        SELECT as_of_date
        FROM (
          SELECT DISTINCT as_of_date
          FROM `eng-reactor-287421.analytics_data_source.minute_trade_count`
          WHERE as_of_date <= today
        )
        ORDER BY as_of_date DESC
        LIMIT 10
      )
      SELECT
        as_of_date,
        trades_D,
        trades_S,
        trades_P
      FROM `eng-reactor-287421.analytics_data_source.minute_trade_count`
      WHERE as_of_date IN (SELECT as_of_date FROM last10)
      ORDER BY as_of_date
    """
    try:
        rows = BQ_CLIENT.query(query).result()
        out = []
        for r in rows:
            d = dict(r)
            out.append({
                "date": d["as_of_date"].isoformat(),
                "D": int(d["trades_D"] or 0),
                "S": int(d["trades_S"] or 0),
                "P": int(d["trades_P"] or 0),
                "total": int((d["trades_D"] or 0) + (d["trades_S"] or 0) + (d["trades_P"] or 0)),
            })
        return make_response(jsonify(out), 200, cors_headers())
    except Exception as e:
        logger.error(f"Error in handle_muni_market_stats_10d: {e}")
        import traceback; logger.error(traceback.format_exc())
        return make_response(jsonify({"status":"error","message":str(e)}), 500, cors_headers())

def handle_muni_top_issues():
    """
    Return the top 10 issues for the current and previous business day.
    Optional query param: issue_type in {'seasoned','new'}.
    If absent, returns overall top 10 across types.
    """
    logger.info("Processing muni top issues request")
    if BQ_CLIENT is None:
        return make_response(jsonify({"status": "error", "message": "BigQuery client not initialized"}), 500, cors_headers())

    issue_type = request.args.get("issue_type")
    if issue_type:
        issue_type = issue_type.strip().lower()
        if issue_type not in {"seasoned", "new"}:
            return make_response(jsonify({"status": "error", "message": "issue_type must be 'seasoned' or 'new'"}), 400, cors_headers())

    # issue_type_filter is a variable that contains the filter for the results of the query, when it is NULL, all issues are passed.
    query = f"""
    DECLARE tz STRING DEFAULT 'America/New_York';
    DECLARE today DATE DEFAULT CURRENT_DATE(tz);
    DECLARE issue_type_filter STRING DEFAULT {('"' + issue_type + '"') if issue_type else 'NULL'};

    WITH days AS (
      SELECT
        (SELECT MAX(as_of_date) FROM `eng-reactor-287421.analytics_data_source.minute_top_issues` WHERE as_of_date <= today) AS curr_bd
    ),
    prev_day AS (
      SELECT
        (SELECT MAX(as_of_date)
         FROM `eng-reactor-287421.analytics_data_source.minute_top_issues`, days
         WHERE as_of_date < days.curr_bd) AS prev_bd
    ),
    curr AS (
      SELECT
        as_of_date,
        issue_type,
        cusip,
        security_description,
        trade_count,
        ROW_NUMBER() OVER (ORDER BY trade_count DESC, cusip) AS rnk
      FROM `eng-reactor-287421.analytics_data_source.minute_top_issues`, days
      WHERE as_of_date = days.curr_bd
        AND (issue_type_filter IS NULL OR LOWER(issue_type) = LOWER(issue_type_filter))
    ),
    prev AS (
      SELECT
        as_of_date,
        issue_type,
        cusip,
        security_description,
        trade_count,
        ROW_NUMBER() OVER (ORDER BY trade_count DESC, cusip) AS rnk
      FROM `eng-reactor-287421.analytics_data_source.minute_top_issues`, prev_day
      WHERE as_of_date = prev_day.prev_bd
        AND (issue_type_filter IS NULL OR LOWER(issue_type) = LOWER(issue_type_filter))
    )
    SELECT 'current' AS bucket, as_of_date, issue_type, cusip, security_description, trade_count, rnk
    FROM curr WHERE rnk <= 10
    UNION ALL
    SELECT 'previous' AS bucket, as_of_date, issue_type, cusip, security_description, trade_count, rnk
    FROM prev WHERE rnk <= 10
    ORDER BY bucket, rnk;
    """

    try:
        rows = BQ_CLIENT.query(query).result()
        current, previous = [], []
        curr_date, prev_date = None, None

        for row in rows:
            d = dict(row)
            item = {
                "rank": int(d["rnk"]),
                "issue_type": d["issue_type"],
                "cusip": d["cusip"],
                "security_description": d["security_description"],
                "trade_count": int(d["trade_count"]),
            }
            if d["bucket"] == "current":
                curr_date = d["as_of_date"].isoformat() if d["as_of_date"] else curr_date
                current.append(item)
            else:
                prev_date = d["as_of_date"].isoformat() if d["as_of_date"] else prev_date
                previous.append(item)

        payload = {
            "current_as_of_date": curr_date,
            "previous_as_of_date": prev_date,
            "issue_type": issue_type if issue_type else None,
            "current": current,
            "previous": previous,
        }
        return make_response(jsonify(payload), 200, cors_headers())

    except Exception as e:
        logger.error(f"Error in handle_muni_top_issues: {e}")
        import traceback; logger.error(traceback.format_exc())
        return make_response(jsonify({"status": "error", "message": str(e)}), 500, cors_headers())

# code for api for AAA Benchmark

def _query_intraday_series(table_name, days):
    """
    Pull intraday minute-level mean_ytw from BigQuery for the given table and list of DATEs.
    Returns a DataFrame with columns: dt (tz-aware), mean_ytw, date (YYYY-MM-DD), time (HH:MM).
    """
    if BQ_CLIENT is None:
        raise RuntimeError("BigQuery client not initialized")

    date_list_sql = ", ".join([f"DATE '{d}'" for d in days])
    query = f"""
        SELECT
          datetime AS dt,
          mean_ytw
        FROM `eng-reactor-287421.aaa_benchmark.{table_name}`
        WHERE DATE(datetime) IN ({date_list_sql})
          AND TIME(datetime) BETWEEN "09:30:00" AND "16:00:00"
        ORDER BY dt
    """
    df = BQ_CLIENT.query(query).result().to_dataframe()
    if df.empty:
        return pd.DataFrame(columns=["dt", "date", "time", "mean_ytw"])

    # Normalize time columns
    df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(EASTERN)
    df = df.sort_values("dt")
    df["date"] = df["dt"].dt.date.astype(str)
    df["time"] = df["dt"].dt.strftime("%H:%M")
    return df

def _rolling_ma(df, window_minutes: int = DEFAULT_SMOOTH_MINUTES):
    """
    Rolling mean over `window_minutes` per trading day.
    Emits values only after a full window (first N-1 minutes are NaN and dropped downstream).
    """
    if df.empty:
        return pd.DataFrame(columns=["dt", "date", "time", "mean_ytw"])
    window_minutes = max(1, int(window_minutes))
    df = df.copy().set_index("dt")
    out = []
    for _, g in df.groupby(pd.Grouper(freq="D")):
        if g.empty:
            continue
        smoothed = g["mean_ytw"].rolling(f"{window_minutes}T", min_periods=1).mean()
        gg = g.copy()
        gg["mean_ytw"] = smoothed
        out.append(gg)
    sm = pd.concat(out).reset_index().rename(columns={"index": "dt"})
    sm["date"] = sm["dt"].dt.date.astype(str)
    sm["time"] = sm["dt"].dt.strftime("%H:%M")
    return sm[["dt", "date", "time", "mean_ytw"]]

def _merge_5y_10y(dates, window_minutes: int = DEFAULT_SMOOTH_MINUTES):
    """
    Returns: { 'YYYY-MM-DD': [ {time, '5': val, '10': val}, ... ], ... }
    Only dates that actually have rows will appear.
    """
    df_5y  = _query_intraday_series("five_year", dates)
    df_10y = _query_intraday_series("ten_year", dates)

    df_5y  = _rolling_ma(df_5y, window_minutes)
    df_10y = _rolling_ma(df_10y, window_minutes)

    merged = pd.merge(
        df_5y[["date", "time", "mean_ytw"]],
        df_10y[["date", "time", "mean_ytw"]],
        on=["date", "time"],
        how="outer",
        suffixes=("_5", "_10")
    ).sort_values(["date", "time"])

    payload = {}
    for d, g in merged.groupby("date"):
        rows = []
        for _, r in g.iterrows():
            rows.append({
                "time": r["time"],
                "5": None if pd.isna(r["mean_ytw_5"]) else float(r["mean_ytw_5"]),
                "10": None if pd.isna(r["mean_ytw_10"]) else float(r["mean_ytw_10"]),
            })
        # keep only non-empty dates (at least one non-NaN)
        if any((row["5"] is not None) or (row["10"] is not None) for row in rows):
            payload[d] = rows
    return payload

def handle_aaa_benchmark():
    """
    Serve N-minute smoothed intraday mean_ytw for 5y & 10y.
    Always queries [prev business day, today]. If today has no rows, it is omitted.
    Query param: smooth (minutes, default 5, clamped 1..120)
    Shape:
    {
      "status": "success",
      "dates": ["YYYY-MM-DD", ...],   // only dates with data
      "smoothing_minutes": N,
      "data": { "YYYY-MM-DD": [ { "time":"09:35","5":..., "10":... }, ... ] }
    }
    """
    now_et  = datetime.now(EASTERN)
    prev_bd = last_business_day(now_et)   # uses your calendar helper :contentReference[oaicite:0]{index=0}

    # Always query prev_bd and today; let BigQuery return 0 rows for today if it’s a non-trading day.
    requested_dates = [str(prev_bd), now_et.date().isoformat()]

    # smoothing param (optional)
    wm_str = (request.args.get("smooth") or "").strip()
    try:
        wm = int(wm_str) if wm_str else DEFAULT_SMOOTH_MINUTES
    except ValueError:
        wm = DEFAULT_SMOOTH_MINUTES
    wm = max(1, min(120, wm))

    try:
        payload = _merge_5y_10y(requested_dates, wm)

        # Only include dates that actually have data (mirrors YieldCharts behavior)
        present_dates = sorted(payload.keys())

        return make_response(jsonify({
            "status": "success",
            "dates": present_dates,
            "smoothing_minutes": wm,
            "data": payload
        }), 200, cors_headers())
    except Exception as e:
        logger.error(f"[aaa-benchmark] {e}")
        import traceback; logger.error(traceback.format_exc())
        return make_response(jsonify({"status":"error","message":str(e)}), 500, cors_headers())

