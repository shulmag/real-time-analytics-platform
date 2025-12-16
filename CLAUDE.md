# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a monorepo containing Ficc.ai's infrastructure for municipal bond pricing, market data processing, and analytics. The repository includes multiple independent services, cloud functions, web applications, and data pipelines that work together to provide real-time bond pricing and market analysis.

**GCP Project**: `eng-reactor-287421`

## Repository Structure

### Major Components

- **cloud_functions/** - 50+ Cloud Functions for data processing, model training, and scheduled tasks
- **analytics/** - Analytics platform with React frontend and Cloud Function backend
- **blockchain/** - Stellar blockchain service for storing municipal bond prices
- **ficc_student_portal/** - Student portal application (frontend + server)
- **muni_market_monitor/** - Real-time muni bond market monitoring with price comparison
- **data_pipeline/** - BigQuery scheduled queries and data transformations
- **cloud_scheduler_jobs/** - Cloud Scheduler job configurations in JSON format
- **app_engine/** - Website and demo applications deployed on App Engine
- **ficc-chrome-extension/** - Chrome extension for ficc.ai
- **FixClientLite_SourceCode/** - C# FIX protocol client for ingesting trade messages
- **yield_curve/** - Jupyter notebooks for yield curve modeling
- **VertexAI/** - Model experimentation and backtesting tools

## Key Commands

### Cloud Functions Deployment

Deploy a Cloud Function from its directory:
```bash
cd cloud_functions/<function-name>
gcloud functions deploy <function-name> \
  --runtime python312 \
  --region us-central1 \
  --source . \
  --entry-point main \
  --trigger-http \
  --memory 1024MB \
  --timeout 540s
```

For functions requiring VPC access (Redis, private services):
```bash
gcloud functions deploy <function-name> \
  --runtime python312 \
  --region us-central1 \
  --source . \
  --entry-point main \
  --trigger-http \
  --memory 1024MB \
  --timeout 540s \
  --vpc-connector yield-curve-connector
```

### Cloud Run Deployments

Deploy services to Cloud Run:
```bash
# Blockchain service
cd blockchain/server
gcloud run deploy blockchain \
  --source . \
  --allow-unauthenticated \
  --region us-central1 \
  --cpu 1 \
  --memory 512Mi \
  --timeout 300 \
  --execution-environment gen2

# Muni Market Monitor backend
cd muni_market_monitor/server
gcloud run deploy monitor \
  --source . \
  --allow-unauthenticated \
  --region us-central1 \
  --cpu 1 \
  --memory 1Gi \
  --timeout 300 \
  --execution-environment gen2
```

### Frontend Development & Deployment

Analytics platform:
```bash
# Development
cd analytics/frontend
npm install
npm run dev  # Runs on http://localhost:5173

# Production build & deploy
npm run build
gcloud app deploy analytics.yaml
```

Muni Market Monitor:
```bash
# Development
cd muni_market_monitor/frontend
npm install
npm run dev  # Runs on http://localhost:5174

# Production build & deploy
npm run build
gcloud app deploy market_monitor.yaml
gcloud app deploy dispatch.yaml
```

Student Portal:
```bash
cd ficc_student_portal/frontend
npm install
npm run dev
```

### Redis Setup for Local Development

Many services require Redis access. For local development, set up SSH tunnel:
```bash
# Start Redis bastion host
gcloud compute instances start redis-bastion \
  --project=eng-reactor-287421 \
  --zone=us-central1-c

# Create SSH tunnel
gcloud compute ssh redis-bastion \
  --project=eng-reactor-287421 \
  --zone=us-central1-c \
  --tunnel-through-iap \
  -- -L 6379:10.227.69.60:6379
```

Update Redis host in code to `127.0.0.1` for local development, and `10.227.69.60` for production.

### Cloud Scheduler Jobs Management

Export all Cloud Scheduler jobs:
```bash
# Export jobs from each region
gcloud scheduler jobs list --location=us-east4 --format=json > jobs_east4.json
gcloud scheduler jobs list --location=us-central1 --format=json > jobs_central1.json
gcloud scheduler jobs list --location=us-west1 --format=json > jobs_west1.json
```

Export a single job:
```bash
gcloud scheduler jobs describe <job_name> --location=<location> --format=json > <job_name>.json
git add -f <job_name>.json  # JSON files are gitignored by default
```

## Architecture

### Cloud Functions

The `cloud_functions/` directory contains numerous serverless functions organized by purpose:

**Data Processing & Updates:**
- `train-minute-yield-curve` - Fits real-time yield curve using S&P index data and ETF models
- `price-entire-universe-for-investortools` - Prices all bonds for InvestorTools integration
- `fast-trade-history-redis-update-v2` - Updates Redis with trade history data
- `reference-data-redis-update-v3` - Updates Redis with reference data
- `update-sp-all-indices-and-maturities` - Updates S&P index data
- `update-daily-etf-prices` - Updates ETF pricing data

**Model Training:**
- `train-daily-yield-curve` - Daily yield curve model training
- `train-daily-etf-model` - ETF model training
- `deploy-vertex-training-pipeline` - Deploys Vertex AI training pipelines

**Data Ingestion:**
- `get-msrb-trade-messages` - Fetches MSRB trade data
- `load-ice-file-to-bq` - Loads ICE data files to BigQuery
- `copy-msrb-replay-to-gcs` - Archives MSRB data to Google Cloud Storage

**Monitoring & Alerts:**
- `check-demo-status-v2` - Monitors demo environment health
- `send-vm-status` - Sends VM status reports
- `email-daily-gcp-expenses` - Daily GCP cost notifications
- `end-of-day-usage-summary` - Daily usage reports

**Blockchain:**
- `muni-price-to-blockchain` - Publishes bond prices to Stellar blockchain

### Analytics Platform

Located in `analytics/`:
- **Frontend**: React app with Firebase authentication, deployed to App Engine
- **Backend**: Cloud Function (`analytics-server-v2`) providing RESTful API
- **Features**: Real-time bond analytics, email-only authentication, usage tracking in BigQuery
- **Redis Integration**: Caches yield curve data for fast access

### Blockchain Service

Located in `blockchain/`:
- **Server**: Flask application for managing muni bond prices on Stellar blockchain
- **Frontend**: React interface for blockchain data visualization
- **Smart Contracts**: Soroban contracts in `soroban-muni-contracts/`
- **Deployment**: Cloud Run service with public access

### Data Pipeline

Located in `data_pipeline/`:
- **Scheduled Queries**: BigQuery scheduled queries for data transformations
- **View Definitions**: Jupyter notebooks defining materialized views

### Student Portal

Located in `ficc_student_portal/`:
- **Frontend**: React application
- **Backend**: Python server (Dockerfile-based deployment)
- Educational platform for bond pricing concepts

### Muni Market Monitor

Located in `muni_market_monitor/`:
- **Purpose**: Real-time municipal bond market monitoring with price deltas
- **Frontend**: React + Vite with Bootstrap UI
- **Backend**: FastAPI server with Google Cloud Storage integration
- **Features**: Compare real-time prices to yesterday's 4 PM compliance prices
- **Data Storage**: Compliance prices cached in GCS bucket `market-monitor`

## Important Infrastructure Details

### Redis Configuration
- **Production Host**: `10.227.69.60:6379`
- **Local Development**: Use SSH tunnel through `redis-bastion` host
- **Purpose**: Caches yield curves, trade history, and reference data for low-latency access

### BigQuery Datasets
- **Main Dataset**: Project contains multiple datasets for trade data, reference data, and analytics
- **Analytics Tracking**: `api_calls_tracker.ficc_analytics_usage`
- **Yield Curve Data**: `spBondIndex` and `spBondIndexMaturities` datasets

### Firebase
- **Authentication**: Email-only (magic link) and email/password authentication
- **Token Caching**: 30-day authentication persistence
- **Configuration**: Each frontend has Firebase config in `src/config.js`

### VPC Connector
- **Name**: `yield-curve-connector`
- **Purpose**: Allows Cloud Functions to access Redis and other private resources
- **Region**: `us-central1`

### App Engine
- Multiple YAML files for different services (analytics, market monitor, website)
- Use `dispatch.yaml` for routing between services

## Development Workflow

### Working with Cloud Functions
1. Navigate to the function directory: `cd cloud_functions/<function-name>`
2. Install dependencies locally if needed: `pip install -r requirements.txt`
3. Test locally by setting `TESTING = True` in code and providing credentials
4. Deploy using gcloud command with appropriate flags
5. Monitor logs: `gcloud functions logs read <function-name> --limit 50`

### Frontend Configuration Updates
Before deploying frontends, verify:
1. **API URLs** are set correctly in `config.js` files
2. **Firebase config** matches target environment
3. **Authentication settings** are enabled in Firebase Console
4. **CORS** is configured on backend endpoints if needed

### Testing Backend APIs
Most backends support local development with `uvicorn` or Flask dev server:
```bash
# FastAPI
uvicorn main:app --reload --port 8000

# Flask
python main.py
```

### Credentials Management
- **Service Account**: Use `creds.json` in root directory (gitignored)
- **Local Development**: Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable
- **Production**: Cloud Functions and Cloud Run use default service account

## Common Patterns

### Environment Detection
Many services use a `TESTING` flag to switch between local and production modes:
```python
TESTING = False  # Set to True for local development

if TESTING:
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/creds.json'
```

### BigQuery Client Initialization
```python
from google.cloud import bigquery
client = bigquery.Client(project='eng-reactor-287421')
```

### Redis Client Pattern
```python
import redis
REDIS_HOST = '10.227.69.60'  # Production
# REDIS_HOST = '127.0.0.1'  # Local development
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0)
```

## Related Repositories

- **ficc_python**: Internal Python package for ML models and data processing (see its CLAUDE.md)
- **mbs**: Mortgage-backed securities tools
- **ficc-warm-fuzzies**: Internal tools and utilities
