#Nightly Updates

"""
Cloud Function to update bond spread data nightly.
This function should be scheduled to run every night at a specific time.
"""

import os
import functions_framework
from google.cloud import bigquery, storage
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
from pytz import timezone
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the functions from spreads.py
from spreads import (
    get_last_n_business_days, 
    get_existing_data, 
    process_missing_days,
    USHolidayCalendarWithGoodFriday
)

# Constants
PROJECT_ID = 'eng-reactor-287421'
EASTERN = timezone('US/Eastern')

@functions_framework.http
def update_spreads(request):
    """
    HTTP Cloud Function to update bond spread data.
    
    Args:
        request (flask.Request): The request object.
        
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`.
    """
    # Initialize clients
    try:
        bq_client = bigquery.Client(project=PROJECT_ID)
        storage_client = storage.Client(project=PROJECT_ID)
    except Exception as e:
        logger.error(f"Failed to initialize clients: {str(e)}")
        return {'error': f'Failed to initialize clients: {str(e)}'}, 500
    
    # Get the current date in Eastern timezone
    current_time = datetime.now(EASTERN)
    logger.info(f"Running spreads update at {current_time}")
    
    try:
        # Get last 10 business days to ensure we have recent data
        business_days = get_last_n_business_days(10)
        logger.info(f"Checking data for business days: {[d.strftime('%Y-%m-%d') for d in business_days]}")
        
        # Get existing data and check which dates we have
        existing_df, existing_dates = get_existing_data(bq_client, business_days)
        
        # Find missing dates
        missing_dates = [d for d in business_days if d not in existing_dates]
        
        if not missing_dates:
            logger.info("All recent business days already have data")
            return {
                'status': 'success',
                'message': 'All recent business days already have data',
                'checked_days': len(business_days),
                'missing_days': 0
            }, 200
        
        logger.info(f"Found {len(missing_dates)} missing dates: {[d.strftime('%Y-%m-%d') for d in missing_dates]}")
        
        # Process missing days
        processed_count = process_missing_days(bq_client, storage_client, business_days, existing_dates)
        
        logger.info(f"Successfully processed {processed_count} days")
        
        return {
            'status': 'success',
            'message': f'Successfully processed {processed_count} days',
            'checked_days': len(business_days),
            'missing_days': len(missing_dates),
            'processed_days': processed_count
        }, 200
        
    except Exception as e:
        logger.error(f"Error updating spreads: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'error': f'Error updating spreads: {str(e)}',
            'traceback': traceback.format_exc()
        }, 500


# For running locally or via Cloud Scheduler
if __name__ == "__main__":
    # Simulate an HTTP request for local testing
    class MockRequest:
        pass
    
    result = update_spreads(MockRequest())
    print(result)