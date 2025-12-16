# FICC Analytics Cloud Function

This cloud function serves as the backend API for the FICC Analytics dashboard application.

## Deployment

To deploy to Google Cloud Functions:

```bash
gcloud functions deploy analytics-server-v2 \
  --runtime python312 \
  --region us-central1 \
  --source . \
  --entry-point main \
  --trigger-http \
  --memory 1024MB \
  --timeout 540s \
  --vpc-connector yield-curve-connector
```

## Redis Implementation

The application now uses Redis to store and retrieve yield curve data for improved performance. The Redis implementation has the following components:

1. **redis_helper.py**: A module that provides functions for:
   - Connecting to Redis through a bastion host
   - Retrieving yield curve data from Redis
   - Calculating yield values using the Nelson-Siegel model
   - Generating multiple data points for charting
   - Calculating CoD (Change of Day) values

2. **Connection Details**:
   - Redis is accessed through a bastion host at IP 10.227.69.60:6379
   - Local connection requires SSH tunneling through the bastion host

### Setting Up Redis Connection

To connect to Redis:

1. Start the bastion host:
```bash
gcloud compute instances start redis-bastion --project=eng-reactor-287421 --zone=us-central1-c
```

2. Create an SSH tunnel for port forwarding:
```bash
gcloud compute ssh redis-bastion --project=eng-reactor-287421 --zone=us-central1-c --tunnel-through-iap -- -L 6379:10.227.69.60:6379
```

3. The application will then be able to connect to Redis via localhost:6379

## Authentication Flow

The backend uses Firebase Authentication to secure API endpoints. The flow is as follows:

1. Frontend obtains a Firebase ID token upon user login
2. All API requests include this token in the Authorization header
3. Backend verifies the token using Firebase Admin SDK
4. User information (email) is extracted from the verified token
5. Frontend supports both email/password and passwordless email link authentication

### Email-Only Authentication with Magic Links

The system now implements passwordless authentication using Firebase Email Link (magic links):

1. New users can register and log in with just their email (no password required)
2. Existing users can continue to use email/password authentication
3. The feature can be toggled via a configuration flag `EMAIL_ONLY_AUTH_ENABLED`
4. Authentication tokens are cached on the frontend to prevent Firebase quota issues

### IP Address Logging

The system properly detects client IP addresses even behind proxies by checking headers in the following order:
1. X-Forwarded-For
2. X-Real-IP
3. CF-Connecting-IP (Cloudflare)
4. True-Client-IP (Akamai)
5. remote_addr (direct connection)

IP addresses are logged in BigQuery for analytics purposes.

## API Endpoints

### /api/verify-auth

Verifies the Firebase token and returns user information.

**Headers**:
- `Authorization`: Bearer token format (`Bearer your-firebase-token`)

**Response**:
```json
{
  "status": "success",
  "authenticated": true,
  "user": {
    "email": "user@example.com",
    "email_verified": true,
    "uid": "user123"
  }
}
```

### /api/yield-curves

Provides yield curve data for different maturities. Now uses Redis to retrieve data.

**Headers**:
- `Authorization`: Bearer token format (`Bearer your-firebase-token`)

**Parameters**:
- `type`: 'daily' or 'realtime' (default: 'realtime')
- `start_date`: Start date in ISO format (default: depends on type)
- `end_date`: End date in ISO format (default: current date)
- `maturities`: Comma-separated list of maturities (e.g., '1,2,5,10')

**Response**:
```json
{
  "status": "success",
  "type": "realtime",
  "source": "redis",
  "start_date": "2025-05-01",
  "end_date": "2025-05-02",
  "maturities": [1, 2, 5, 10, 15, 20, 30],
  "data": [
    {
      "timestamp": "2025-05-01:09:30",
      "values": {
        "5": 334.56,
        "10": 421.78,
        "15": 443.21,
        "20": 447.65
      }
    },
    ...
  ]
}
```

**Notes**:
- For 'realtime' type, if no start_date is provided, the API will use the previous business day as the start date
- Yield values are returned in basis points (e.g., 334.56 = 3.3456%)
- The frontend divides these values by 100 to display as percentages
- Data is now sourced from Redis, with the `source` field in the response indicating this
- Since Redis only has one timestamp of data, multiple data points are simulated for charting
- Timestamps use the format "YYYY-MM-DD:HH:MM" to match Redis key format

### /api/cod-values

Provides Change of Day (CoD) values comparing today with yesterday. Now uses Redis for data.

**Headers**:
- `Authorization`: Bearer token format (`Bearer your-firebase-token`)

**Parameters**:
- `type`: 'daily' or 'realtime' (default: 'realtime')
- `maturities`: Comma-separated list of maturities (e.g., '5,10,15,20')

**Response**:
```json
{
  "status": "success",
  "type": "realtime",
  "source": "redis",
  "date": "2025-05-02",
  "data": {
    "5": {
      "yesterday": 3.3456,
      "today": 3.3952,
      "change": 0.0496
    },
    "10": {
      "yesterday": 4.2178,
      "today": 4.2578,
      "change": 0.0400
    },
    ...
  }
}
```

**Notes**:
- Yield values are returned as percentages (e.g., 3.3456 = 3.3456%)
- The `change` value is the difference between today and yesterday in percentage points
- Since Redis only has one timestamp of data, yesterday's values are simulated
- Data is sourced from Redis, with the `source` field in the response indicating this

### /api/market-metrics

Provides market strength metrics based on MSRB trade data.

**Headers**:
- `Authorization`: Bearer token format (`Bearer your-firebase-token`)

**Parameters**:
- `date`: Date in YYYY-MM-DD format (default: current date)

**Response**:
```json
{
  "status": "success",
  "date": "2025-05-02",
  "data": {
    "marketStrength": {
      "buys": 123,
      "sells": 87,
      ...
    },
    "retailStrength": {
      "buys": 456,
      "sells": 321,
      ...
    }
  }
}
```

## Date Handling

The application implements several date handling features:

1. **Business Day Detection**: The system detects weekends and holidays to determine business days
2. **Market Hours**: Different display logic is used before/after 9:35 AM ET
3. **Date Formatting**: Consistent date formatting is used for all displayed dates:
   - Full format: "Friday, May 2, 2025" (used in market closed messages)
   - Short format: "5/2" (used in chart legends)

## Development

### Running Locally

To run the cloud function locally:

```bash
cd /Users/gil/git/ficc/analytics/cloud_function
python run_server.py
```

Or manually:

```bash
cd /Users/gil/git/ficc/analytics/cloud_function
functions-framework --target=main --port=8000
```

### Testing Authentication

To test the authentication flow:

1. Obtain a Firebase ID token from the frontend. This can be done by:
   - Logging in to the frontend app
   - Opening browser devtools and using the following in the console:
     ```javascript
     // In browser console while logged in
     await firebase.auth().currentUser.getIdToken()
     ```
   - Copy the returned token

2. Run the test script:
   ```bash
   python test_auth.py your-firebase-token
   ```

### Testing with Mock Dates

For testing with mock dates:

1. Create a `test_local.py` file with a mock date:
```python
from datetime import datetime
from pytz import timezone
# CRITICAL: It's important to use May 2, 2025 (Friday) as the test date
# The frontend expects data for May 2, 2025 and May 1, 2025 (Thursday) in test mode
MOCK_DATE = datetime(2025, 5, 2, 10, 30, 0, tzinfo=timezone('US/Eastern'))
```

2. Run the local server:
```bash
python run_server.py
```

> **Note**: When testing on weekends or holidays, the backend will automatically
> return data for the last two business days, thanks to the `last_business_day` function.
> The frontend will display these dates correctly without any hardcoding.



> **Note**: The function no longer uses `--allow-unauthenticated` since we've 
> implemented Firebase authentication.