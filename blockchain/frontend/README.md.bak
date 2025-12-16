# Municipal Bond Oracle

A decentralized oracle for municipal bond pricing data using Stellar's Soroban smart contracts.

## Overview

The Municipal Bond Oracle is a production-ready system for storing and retrieving municipal bond price data on the Stellar blockchain. It integrates real-time municipal bond prices from FICC.ai and records them immutably on-chain through Soroban smart contracts.

![Screenshot of the Oracle UI](screenshot_url_here.png)

## Architecture

The system consists of three main components:

### 1. Smart Contract (Rust)

- Written in Rust for the Soroban platform
- Stores price, yield, trade amount, and trade type for each bond
- Maintains historical price data by CUSIP
- Includes basic admin validation
- Located in `/contracts/` directory

### 2. Backend Service (Python)

- Flask server handling API endpoints
- Soroban contract interactions via stellar-sdk
- FICC.ai API integration for real-time pricing
- Google Cloud services integration (Secret Manager, Cloud Storage)
- Located in the root directory

### 3. Frontend (React)

- Real-time price ticker
- Blockchain price display
- Transaction interface with Freighter wallet integration
- Auto-refreshing data
- Located in `/frontend/` directory

## API Endpoints

### Read-only Operations

- `GET /health` - Check system health
- `GET /get_price/<cusip>` - Get complete price history for a CUSIP
- `GET /latest_prices?limit=N` - Get latest prices for all tracked CUSIPs
- `GET /ticker` - Get top holdings with current blockchain prices

### Write Operations

- `POST /prepare_price_update` - Prepare a blockchain transaction with FICC pricing
- `POST /submit_transaction` - Submit a signed transaction to the blockchain

## Development Setup

### Prerequisites

- Rust (latest stable)
- Python 3.10+
- Node.js 16+
- Soroban CLI
- Freighter wallet (for transaction signing)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/muni-bond-oracle.git
   cd muni-bond-oracle
   ```

2. Set up the smart contract:
   ```bash
   cd contracts
   cargo build --release --target wasm32-unknown-unknown
   ```

3. Set up the backend:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the frontend:
   ```bash
   cd frontend
   npm install
   npm run build
   ```

### Configuration

1. Create a `.env` file with the following variables:
   ```
   SOROBAN_RPC_URL=https://soroban-testnet.stellar.org
   CONTRACT_ID=your_contract_id
   ADMIN_SECRET=your_admin_secret_key
   PUBLIC_KEY=your_public_key
   ```

2. Configure Google Cloud credentials (if using GCP services)

## Deployment

### Smart Contract


The flow should now be:

User confirms fees in modal
Freighter signs the transaction
Signed transaction is sent to this endpoint
Transaction is submitted to network
User sees result

Frontend (React) → Freighter → Backend (Flask) → Soroban Network
   |                  |             |                 |
   |-- Show fees      |             |                 |
   |-- Ask Freighter to sign        |                 |
   |                  |             |                 |
   |                  |-- Signs     |                 |
   |                  |             |                 |
   |-- Send signed XDR ------------>|                 |
   |                                |-- Submit ------>|
   |                                |                 |
   |<---- Return result ------------|<---- Result ----|



