build:
	cargo build --target wasm32-unknown-unknown --release

deploy: build
	soroban contract deploy --network testnet --source admin --wasm target/wasm32-unknown-unknown/release/soroban_muni_contract.wasm