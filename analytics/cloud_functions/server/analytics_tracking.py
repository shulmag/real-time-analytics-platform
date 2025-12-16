'''
Analytics usage tracking module for ficc analytics.
Tracks user sessions, activity data, IP addresses, and response times in BigQuery.
'''
from datetime import datetime
import uuid
import traceback

def get_client_ip(request):
    """
    Extract the client's real IP address from request headers, accounting for proxies.
    Checks multiple header possibilities in order of reliability.
    
    Args:
        request: Flask request object
        
    Returns:
        str: Client IP address or None if not found
    """
    # Check standard headers that might contain the real IP behind proxies
    if request is None:
        return None
        
    # Check X-Forwarded-For header first (most common for proxies)
    if request.headers.get('X-Forwarded-For'):
        # X-Forwarded-For format: client, proxy1, proxy2, ...
        # Get the leftmost IP which should be the original client
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    
    # Check other common headers for proxied requests
    if request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    
    if request.headers.get('CF-Connecting-IP'):  # Cloudflare
        return request.headers.get('CF-Connecting-IP')
        
    if request.headers.get('True-Client-IP'):  # Akamai and others
        return request.headers.get('True-Client-IP')
    
    # Fall back to remote_addr if no proxy headers found
    if hasattr(request, 'remote_addr'):
        return request.remote_addr
        
    return None

def log_analytics_usage(bq_client, user_email, component, action=None, start_time=None, error=False, request=None):
    """
    Log ficc analytics usage to BigQuery for analytics tracking.
    
    Args:
        bq_client: BigQuery client instance (if None, logging is skipped)
        user_email: User's email address (uses 'anonymous' for unauthenticated requests)
        component: Component being used (dashboard, yield-curves, etc.)
        action: Optional user action (e.g., 'refresh', 'normal load')
        start_time: Optional start time for timing (to calculate response times)
        error: Boolean indicating whether an error occurred
        request: Flask request object (for IP and user agent extraction)
        
    Returns:
        None - logs data to BigQuery but doesn't return any values
    """
    try:
        # Early return if BigQuery client is not available
        if bq_client is None:
            return
            
        # Use anonymous for null email addresses
        if user_email is None:
            user_email = "anonymous"
            
        # Calculate response time if start_time provided
        response_time_ms = None
        if start_time:
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
        # Generate session ID if not already stored
        session_id = str(uuid.uuid4())
        
        # Extract client IP address (handles proxies)
        ip_address = get_client_ip(request)
        
        # Extract client type from user agent if available
        client_type = "web"  # Default to web
        if request and hasattr(request, 'headers') and 'User-Agent' in request.headers:
            user_agent = request.headers.get('User-Agent', '').lower()
            if 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent or 'ipad' in user_agent:
                client_type = "mobile"
        
        # Prepare row data
        row = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            "user_email": user_email,
            "component": component,
            "action": action,
            "session_id": session_id,
            "ip_address": ip_address,
            "client_type": client_type,
            "response_time_ms": response_time_ms,
            "error_status": error
        }
        
        # Send to BigQuery
        table_id = "eng-reactor-287421.api_calls_tracker.ficc_analytics_usage"
        
        try:
            # First try using the direct SQL insert method for better reliability
            try:
                # Build an INSERT query
                columns = ', '.join(row.keys())
                placeholders = ', '.join(['@' + key for key in row.keys()])
                query = f"INSERT INTO `{table_id}` ({columns}) VALUES ({placeholders})"
                
                # Create job config with query parameters
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter(key, 'STRING' if isinstance(value, str) else 
                                                            'BOOL' if isinstance(value, bool) else 
                                                            'FLOAT64' if isinstance(value, float) else 
                                                            'INT64', value)
                        for key, value in row.items()
                    ]
                )
                
                # Run the query
                query_job = bq_client.query(query, job_config=job_config)
                query_job.result()  # Wait for query to finish
                return
            except Exception:
                # Silently fall back to insert_rows_json method
                pass
            
            # Attempt standard insertion
            errors = bq_client.insert_rows_json(table_id, [row])
            
            if errors:
                # Only log on error
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to log analytics to BigQuery: {errors}")
        except Exception:
            # Silently fail to avoid disrupting the main application flow
            pass
    
    except Exception:
        # Don't let logging errors affect the main functionality
        pass