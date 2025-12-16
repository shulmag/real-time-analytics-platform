# 🏛 Muni Market Monitor

Real-time municipal bond market monitoring system with price comparison functionality. Track MUB holdings, compare real-time prices to yesterday's compliance prices, and analyze price movements.

## 🌟 Features

- Real-time municipal bond price monitoring
- Historical price comparison (real-time vs. yesterday's 4 PM compliance prices)
- Price delta visualization with color-coded indicators
- Firebase authentication for secure access
- Responsive table interface with sorting and pagination
- Automated compliance price caching in Google Cloud Storage

## 🏗 Architecture

### Frontend (React + Vite)
- Bootstrap UI components for responsive design
- Real-time data updates
- Firebase authentication integration
- React Table for advanced table features
- Automatic token management

### Backend (FastAPI)
- RESTful API endpoints
- Google Cloud Storage integration for caching
- FICC API integration for price data
- Pandas for data processing
- Timezone-aware datetime handling

## 🚀 Quick Start

### Prerequisites
- Node.js (v18.17.0 or higher)
- Python 3.10+
- Google Cloud SDK
- Firebase project credentials

### Frontend Development
```bash
# Install dependencies
cd frontend
npm install

# Start development server
npm run dev  # Running at http://localhost:5174
```

### Backend Development
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn main:app --reload --port 8000
```

## 🔧 Configuration

### Frontend Environment Variables - TBD
Create `.env.local`:
```env
VITE_FIREBASE_CONFIG_API_KEY=your_api_key
VITE_FIREBASE_CONFIG_AUTH_DOMAIN=your_domain
VITE_FIREBASE_CONFIG_PROJECT_ID=your_project_id
VITE_FIREBASE_CONFIG_STORAGE_BUCKET=your_bucket
VITE_FIREBASE_CONFIG_MESSAGING_SENDER_ID=your_sender_id
VITE_FIREBASE_CONFIG_APP_ID=your_app_id
```

### Backend Environment Variables
Create `.env`:
```env
GCS_BUCKET_NAME=market-monitor
GCS_FILE_NAME=compliance_prices.pkl
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
```

## 📦 Deployment

### Frontend Deployment
```bash
# Build the frontend
cd frontend
npm run build

# Deploy to App Engine
gcloud app deploy market_monitor.yaml
gcloud app deploy dispatch.yaml
```

### Backend Deployment
```bash
# Deploy to Cloud Run
gcloud run deploy monitor \
  --source . \
  --allow-unauthenticated \
  --region us-central1 \
  --cpu 1 \
  --memory 1Gi \
  --timeout 300 \
  --execution-environment gen2
```

## 📡 API Endpoints

### GET /prices
Fetches current municipal bond prices with historical comparison.

Parameters:
- `access_token`: Firebase authentication token (required)
- `limit`: Number of holdings to retrieve (default: 30)

Response:
```json
[
  {
    "cusip": "string",
    "quantity": "integer",
    "trade_type": "string",
    "price_realtime": "float",
    "price_yesterday": "float",
    "price_delta": "float",
    "ytw_realtime": "float",
    "ytw_yesterday": "float",
    "ytw_delta": "float",
    "coupon": "float",
    "security_description": "string"
  }
]
```

## 📂 Project Structure
```
muni_market_monitor/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Login.jsx
│   │   │   ├── NavBar.jsx
│   │   │   └── PriceTable.jsx
│   │   ├── services/
│   │   │   ├── api.js
│   │   │   └── auth.js
│   │   └── App.jsx
│   ├── market_monitor.yaml
│   └── dispatch.yaml
└── server/
    ├── main.py
    ├── Dockerfile
    └── requirements.txt
```



## 🤝 Support

For support or inquiries, please contact gil@ficc.ai.