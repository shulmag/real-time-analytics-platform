# Soroban Project

## Project Structure

This repository uses the recommended structure for a Soroban project:
```text
.
├── contracts
│   └── hello_world
│       ├── src
│       │   ├── lib.rs
│       │   └── test.rs
│       └── Cargo.toml
├── Cargo.toml
└── README.md
```

- New Soroban contracts can be put in `contracts`, each in their own directory. There is already a `hello_world` contract in there to get you started.
- If you initialized this project with any other example contracts via `--with-example`, those contracts will be in the `contracts` directory as well.
- Contracts should have their own `Cargo.toml` files that rely on the top-level `Cargo.toml` workspace for their dependencies.
- Frontend libraries can be added to the top-level directory as well. If you initialized this project with a frontend template via `--frontend-template` you will have those files already included.


First deploy (though looks like you already have a contract ID):

soroban contract deploy \
  --wasm target/wasm32-unknown-unknown/release/soroban_muni_contract.wasm \
  --source SCZOEGFFYBXJ6UCZIJ6UP5JMU26CGHKVYCLIKFKFCOBXHYYDZZ3FO6ZN \
  --network testnet

Then initialize:

soroban contract invoke \
  --id CBAT7RBWXVATFFXJAGT443E3CDTRCJ7BUDEEQMKCTV3LXGCMK3PZZNQZ \
  --source SCZOEGFFYBXJ6UCZIJ6UP5JMU26CGHKVYCLIKFKFCOBXHYYDZZ3FO6ZN \
  --network testnet \
  -- initialize \
  --admin GD32A5CCG7QBE6K3VG4LNPE5TVYIZBX73F4XHM6CN3PZRXDKWAKRXM7V

  soroban contract invoke \
  --id CBAT7RBWXVATFFXJAGT443E3CDTRCJ7BUDEEQMKCTV3LXGCMK3PZZNQZ \
  --source SCZOEGFFYBXJ6UCZIJ6UP5JMU26CGHKVYCLIKFKFCOBXHYYDZZ3FO6ZN \
  --network testnet \
  -- get_latest_price \
  --cusip "TEST"

  soroban contract invoke \
  --id CBAT7RBWXVATFFXJAGT443E3CDTRCJ7BUDEEQMKCTV3LXGCMK3PZZNQZ \
  --source SCZOEGFFYBXJ6UCZIJ6UP5JMU26CGHKVYCLIKFKFCOBXHYYDZZ3FO6ZN \
  --network testnet \
  -- add_price \
  --caller GD32A5CCG7QBE6K3VG4LNPE5TVYIZBX73F4XHM6CN3PZRXDKWAKRXM7V \
  --cusip "TEST" \
  --price 100000 \
  --yield_value 3500 \
  --trade_amount 1000 \
  --trade_type "D"