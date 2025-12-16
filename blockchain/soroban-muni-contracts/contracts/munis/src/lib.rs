// Gil Shulman on 2025-02-20
// Tell Rust this is a contract that doesn't use the standard library
// This is required for Soroban smart contracts
#![no_std]

// Import the specific parts of the Soroban SDK we need
// These give us the basic building blocks for our contract
use soroban_sdk::{contract, contractimpl, contracterror, contracttype, Address, Env, String, Vec};
use core::cmp;

// Define the possible errors our contract can return
// Each error has a unique number that identifies it on the blockchain
#[contracterror]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum MuniError {
    Unauthorized = 1,    // When someone who isn't admin tries to add prices
    CusipNotFound = 2,   // When we try to look up a CUSIP that isn't in storage
    NoPriceHistory = 3,  // When we try to get prices for a CUSIP with no history
    InvalidInput = 4,    // When the input parameters are invalid
}

// Define what information we store for each price entry
// This is like a receipt for each trade that happens
#[contracttype]
#[derive(Clone)]
pub struct PriceData {
    price: i128,         // The actual price, stored as millicents (e.g., 100.123 -> 100123)
    yield_value: i128,   // The yield percentage, stored as millibps (e.g., 3.456% -> 3456)
    trade_amount: i128,  // How many bonds were traded
    trade_type: String,  // What kind of trade (e.g., "BUY" or "SELL")
    timestamp: u64,      // When this price was recorded (Unix timestamp)
}

// Define how we group all prices for a single municipal bond
// Think of this as a folder containing all trades for one bond
#[contracttype]
#[derive(Clone)]
pub struct MuniPrice {
    cusip: String,                 // The unique identifier for this bond
    price_history: Vec<PriceData>, // A list of all historical prices for this bond
}

// Define the structure for returning just the latest price for each bond
// This is used when we want to show current prices without full history
#[contracttype]
#[derive(Clone)]
pub struct LatestPriceEntry {
    cusip: String,           // The bond's identifier
    latest_price: PriceData, // Only the most recent price information
}

// Declare this as a Soroban contract
#[contract]
pub struct MuniPriceContract;

// Implement all the contract's functionality
#[contractimpl]
impl MuniPriceContract {
    const DAYS_30_IN_LEDGERS: u32 = 30 * 24 * 60 * 60 / 5;  // ~518,400 ledgers

    /// Initialize the contract with an admin address
    /// Admin will only be needed for special administrative tasks
    pub fn initialize(env: Env, admin: Address) {
        env.storage().instance().set(&[0], &admin);
    }

    /// Add a single price entry for a CUSIP
    /// This function is public - anyone can add prices
    pub fn add_price(
        env: Env,
        caller: Address,  // Track who added the price, but don't restrict access
        cusip: String,
        price: i128,
        yield_value: i128,
        trade_amount: i128,
        trade_type: String,
    ) -> Result<(), MuniError> {
        // Create new price data entry
        let new_price = PriceData {
            price,
            yield_value,
            trade_amount,
            trade_type,
            timestamp: env.ledger().timestamp(),
        };

        // Get existing price record or create new one
        let mut muni: MuniPrice = env
            .storage()
            .persistent()
            .get(&cusip)
            .unwrap_or_else(|| MuniPrice {
                cusip: cusip.clone(),
                price_history: Vec::new(&env),
            });

        // Add new price to history
        muni.price_history.push_back(new_price);

        // Store updated record with TTL
        env.storage().persistent().set(&cusip, &muni);
        env.storage().persistent().extend_ttl(&cusip, Self::DAYS_30_IN_LEDGERS, Self::DAYS_30_IN_LEDGERS);

        // Update CUSIP list if needed
        let mut cusip_list: Vec<String> = env
            .storage()
            .persistent()
            .get(&[1])
            .unwrap_or_else(|| Vec::new(&env));

        if !cusip_list.contains(&cusip) {
            cusip_list.push_back(cusip);
            env.storage().persistent().set(&[1], &cusip_list);
            env.storage().persistent().extend_ttl(&[1], Self::DAYS_30_IN_LEDGERS, Self::DAYS_30_IN_LEDGERS);
        }

        Ok(())
    }

    /// Administrative function - only callable by admin
    /// This can be used for future administrative tasks
    pub fn admin_function(
        env: Env,
        caller: Address,
    ) -> Result<(), MuniError> {
        // Verify caller is admin
        let stored_admin: Address = env.storage().instance().get(&[0]).unwrap();
        if stored_admin != caller {
            return Err(MuniError::Unauthorized);
        }
        
        // Place for future admin operations
        Ok(())
    }

    // Get all historical prices for a specific bond
    pub fn get_cusip_prices(env: Env, cusip: String) -> Vec<PriceData> {
        env.storage()
            .persistent()
            .get(&cusip)  // Try to get the price history
            .map(|m: MuniPrice| m.price_history)  // If found, return just the price list
            .unwrap_or_else(|| Vec::new(&env))    // If not found, return empty list
    }

    // Get a list of all bonds we're tracking
    pub fn get_all_cusips(env: Env) -> Vec<String> {
        env.storage()
            .persistent()
            .get(&[1])  // Get the CUSIP list from key [1]
            .unwrap_or_else(|| Vec::new(&env))  // Return empty list if none found
    }

    // Get the most recent price for each bond
    // Can limit how many results to return
    pub fn get_all_latest_prices(env: Env, limit: Option<u32>) -> Vec<LatestPriceEntry> {
        // First get list of all bonds
        let cusip_list = Self::get_all_cusips(env.clone());
        let mut latest_prices = Vec::new(&env);  // Create empty result list
        
        // Figure out how many results to return
        let limit_value = limit.unwrap_or(cusip_list.len() as u32);

        // Look through bonds up to the limit
        for i in 0..cmp::min(limit_value as u32, cusip_list.len() as u32) {
            // For each CUSIP...
            if let Some(cusip) = cusip_list.get(i) {
                // Try to get its price history
                if let Some(muni) = env.storage().persistent().get::<_, MuniPrice>(&cusip) {
                    // If we found history, get the most recent price
                    if let Some(latest_price) = muni.price_history.last() {
                        // Add it to our results
                        latest_prices.push_back(LatestPriceEntry {
                            cusip: cusip.clone(),
                            latest_price: latest_price.clone(),
                        });
                    }
                }
            }
        }

        latest_prices  // Return all the latest prices we found
    }
}

// Test module - only included when running tests
#[cfg(test)]
mod test {
    use super::*;
    use soroban_sdk::testutils::Address as _;
    
    #[test]
    fn test_public_price_addition() {
        let env = Env::default();
        let admin = Address::generate(&env);
        let user = Address::generate(&env);  // Non-admin user
        let contract_id = env.register_contract(None, MuniPriceContract);
        let client = MuniPriceContractClient::new(&env, &contract_id);

        client.initialize(&admin);
        let cusip = String::from_str(&env, "123456789");

        // Test that non-admin can add prices
        client.add_price(
            &user,  // Using non-admin user
            &cusip,
            &100_000i128,
            &500i128,
            &1_000i128,
            &String::from_str(&env, "BUY"),
        );

        let prices = client.get_cusip_prices(&cusip);
        assert_eq!(prices.len(), 1);
    }

    #[test]
    fn test_admin_function() {
        let env = Env::default();
        let admin = Address::generate(&env);
        let user = Address::generate(&env);
        let contract_id = env.register_contract(None, MuniPriceContract);
        let client = MuniPriceContractClient::new(&env, &contract_id);

        client.initialize(&admin);

        // Test admin access succeeds
        assert!(client.try_admin_function(&admin).is_ok());

        // Test non-admin access fails
        assert!(client.try_admin_function(&user).is_err());
    }
}
