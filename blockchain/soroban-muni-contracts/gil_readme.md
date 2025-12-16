# Soroban Muni Contract - create by Gil on 2/20/25

This repository contains a Soroban contract for managing municipal bond price data. The contract allows storing and retrieving price history, as well as fetching the latest price for a given CUSIP. Below are the steps to set up, build, and deploy the contract successfully.

---

## Prerequisites

- **Rust**: Ensure you have Rust installed (latest stable version recommended). Use [rustup](https://rustup.rs/) for installation.
- **Soroban CLI**: Install the Soroban CLI:
  ```bash
  cargo install --force --locked soroban-cli
  ```
- **Dependencies**: Ensure `cargo` can fetch and compile dependencies.

---

## Steps to Build and Deploy

### 1. Upgrade Dependencies

Update the `Cargo.toml` dependencies to ensure compatibility with the latest Soroban SDK:
```toml
[dependencies]
soroban-sdk = "22"

[dev-dependencies]
soroban-sdk = { version = "22", features = ["testutils"] }
```
Run the following command to update:
```bash
cargo update
```

### 2. Configure the Network

Ensure the Soroban network configuration is set up correctly:
```bash
soroban config network rm testnet
soroban config network add --global testnet \
  --rpc-url https://horizon-testnet.stellar.org:443 \
  --network-passphrase "Test SDF Network ; September 2015"
```
Verify the configuration:
```bash
soroban config network list
```

### 3. Build the Contract

Build the contract targeting WebAssembly:
```bash
cargo build --release --target wasm32-unknown-unknown
```

The compiled `.wasm` file will be located at:
```
target/wasm32-unknown-unknown/release/soroban_muni_contract.wasm
```

### 4. Deploy the Contract

Deploy the contract to the testnet:
```bash
soroban contract deploy \
  --network testnet \
  --source admin \
  --wasm target/wasm32-unknown-unknown/release/soroban_muni_contract.wasm
```
```
soroban contract deploy \
  --wasm target/wasm32-unknown-unknown/release/soroban_muni_contract.wasm \
  --source SCZOEGFFYBXJ6UCZIJ6UP5JMU26CGHKVYCLIKFKFCOBXHYYDZZ3FO6ZN \
  --network testnet

soroban contract invoke \
  --id CAPN7ERFCUORY2QW5D7JOXXQ2WJMR4H725ZTSEAV3WLAAQ5SP2P6D7M2 
  --source SCZOEGFFYBXJ6UCZIJ6UP5JMU26CGHKVYCLIKFKFCOBXHYYDZZ3FO6ZN \
  --network testnet \
  -- initialize \
  --admin GD32A5CCG7QBE6K3VG4LNPE5TVYIZBX73F4XHM6CN3PZRXDKWAKRXM7V
```

### 5. Fund the Account

Fund the `admin` account using the Stellar Friendbot:
```bash
curl "https://friendbot.stellar.org?addr=$(soroban config identity address admin)"
```

### 6. Invoke Contract Methods

Test the deployed contract by invoking methods. For example, to fetch the latest price:
```bash
soroban contract invoke \
  --network testnet \
  --source admin \
  --id <contract-id> \
  --method get_latest_price \
  --args '[{"cusip": "123456"}]'
```

---

## Notes

- **Error Handling**: If you encounter JSON-RPC errors or network issues, ensure your RPC endpoint is correct and reachable.
- **Testing**: Update and run the tests in `test.rs` to validate contract logic before deployment.

---

## Project Structure

The repository follows this structure:
```
.
├── contracts
│   └── soroban_muni_contract
│       ├── src
│       │   ├── lib.rs
│       │   └── test.rs
│       └── Cargo.toml
├── Cargo.toml
└── README.md
```


