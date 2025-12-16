# ficc Analytics

This repository contains the FICC Analytics platform, which includes a cloud function backend and a React frontend.

## Deployment Requirements

Before deploying the application, ensure the following configurations are updated:

### Frontend Configuration

1. **API Endpoint Configuration**
   - In `frontend/src/config.js`, ensure the API_URL is set correctly for the target environment:
   ```javascript
   // For production:
   export const API_URL = 'https://us-central1-eng-reactor-287421.cloudfunctions.net/analytics-server';
   ```

2. **Email Authentication Configuration**
   - Make sure `EMAIL_ONLY_AUTH_ENABLED` is set to `true` to enable passwordless email link authentication
   - The `AUTH_REDIRECT_URL` should be set to `window.location.origin` to work correctly in any environment
   - The `AUTH_PERSISTENCE_DAYS` determines how long the authentication state persists (default: 30 days)

3. **Firebase Configuration**
   - Set up your Firebase project for email authentication:
     - In Firebase Console > Authentication > Sign-in methods, ensure both Email/Password and Email link authentication are enabled
     - In Project settings > Public-facing name, set the app name to "ficc.ai Analytics" to improve email templates
     - Make sure your domains are in the authorized domains list in Firebase Authentication settings

### Backend Configuration

1. **Redis Connection**
   - In `cloud_function/redis_helper.py`, update the Redis host configuration:
   ```python
   # For production:
   REDIS_HOST = '10.227.69.60'  # Use direct IP in production
   ```

2. **Google Cloud Credentials**
   - Uncomment the credentials setup in `cloud_function/main.py` if needed:
   ```python
   import os
   # Set credentials path
   creds_path = '/path/to/credentials.json'
   if os.path.exists(creds_path):
       os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
   ```

3. **BigQuery Analytics Logging**
   - Ensure the BigQuery client is properly initialized
   - Check that the table ID in `analytics_tracking.py` is correct:
   ```python
   table_id = "eng-reactor-287421.api_calls_tracker.ficc_analytics_usage"
   ```

## Deployment Steps

### Frontend Deployment

1. Build the React app for production:
   ```bash
   cd frontend
   npm run build
   ```

2. Deploy to App Engine:
   ```bash
   gcloud app deploy analytics.yaml
   ```

### Backend Deployment

1. Deploy as a Cloud Function:
   ```bash
  cd cloud_function
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

## Redis Setup for Local Development

For local development, you'll need to set up an SSH tunnel to the Redis instance:

1. Start the Redis bastion host:
   ```bash
   gcloud compute instances start redis-bastion \
     --project=eng-reactor-287421 \
     --zone=us-central1-c
   ```

2. Create an SSH tunnel for port forwarding:
   ```bash
   gcloud compute ssh redis-bastion \
     --project=eng-reactor-287421 \
     --zone=us-central1-c \
     --tunnel-through-iap \
     -- -L 6379:10.227.69.60:6379
   ```

3. Update the Redis host in `cloud_function/redis_helper.py` for local development:
   ```python
   # For local development:
   REDIS_HOST = '127.0.0.1'  # Connect through local port forwarding
   ```

## Feature Summary

### Email-Only Authentication

The application now supports passwordless email link (magic link) authentication:

1. New users can register and sign in using only their email address
2. Existing users can continue to use email/password authentication
3. The feature can be toggled via the `EMAIL_ONLY_AUTH_ENABLED` flag in `config.js`
4. Authentication tokens are cached to prevent Firebase quota issues

### Analytics Tracking

The system tracks analytics data in BigQuery:

1. User sessions and activities are logged
2. IP address detection works behind proxies with proper header handling
3. Authentication events are tracked
4. Error conditions are monitored

## Testing the Application

Before deploying to production, test the following:

1. Email link authentication for new users
2. Password authentication for existing users
3. Authentication persistence (30-day login retention)
4. IP address logging in analytics
5. Redis connection for yield curve data
6. BigQuery analytics logging

## Troubleshooting

### Email Authentication Issues

- If Firebase emails show "none" as the app name, update the project's public-facing name in Firebase Console
- If Dynamic Links don't work, ensure you've set up a Dynamic Links domain in Firebase Console
- For email link issues, check the Firebase Authentication logs in Firebase Console

### Redis Connection Issues

- Ensure the Redis bastion host is running
- Check that the SSH tunnel is active for local development
- Verify the REDIS_HOST configuration is correct for the environment

### Analytics Logging Issues

- Check that Google Cloud credentials are correctly set up
- Verify that the BigQuery project and table exist
- Check for permission issues in the Cloud IAM settings