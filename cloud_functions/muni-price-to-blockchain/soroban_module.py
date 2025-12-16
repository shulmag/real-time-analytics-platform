"""
Created by: Gil
Helper functions that manage Soroban operations: build → simulate → prepare → sign → send.
Updated for Stellar Mainnet.
"""

BASE_FEE = 10000  # 1000

from typing import Dict, Any
from stellar_sdk import (
    SorobanServer, 
    Keypair, 
    Network,
    scval,
    Server,
    TransactionBuilder,
    Address,
    xdr as stellar_xdr
)

# Constants
MAINNET_URL = "https://horizon.stellar.org"
SOROBAN_RPC_URL = "https://soroban-rpc.mainnet.stellar.gateway.fm"

def init_soroban_server(rpc_url: str) -> dict:
    """Initialize SorobanServer and return it in a context dict"""
    soroban_server = SorobanServer(rpc_url)
    print(f"[SorobanService] Initialized SorobanServer with URL: {rpc_url}")
    return {
        "soroban_server": soroban_server
    }

def check_connection(context: dict) -> bool:
    """Verify connectivity by fetching the latest ledger."""
    try:
        latest_ledger = context["soroban_server"].get_latest_ledger()
        print(f"[SorobanService] Latest ledger: {latest_ledger}")
        return True
    except Exception as e:
        print(f"[SorobanService] Connection check failed: {e}")
        return False

def build_invoke_transaction(context: dict, source_account, contract_id, function_name, params, base_fee=100000):
    """Build an unprepared transaction to invoke a Soroban contract function."""
    try:
        tb = context["soroban_server"].build_transaction_builder(
            source_account=source_account,
            base_fee=base_fee
        )
        tb.set_timeout(300)
        tb.append_invoke_contract_function_op(
            contract_id=contract_id,
            function_name=function_name,
            parameters=params
        )
        transaction = tb.build()
        print(f"[SorobanService] Built transaction: {transaction}")
        return transaction
    except Exception as e:
        print(f"[SorobanService] Error building transaction: {e}")
        raise

def simulate(context: dict, transaction):
    """Simulate the transaction with Soroban to retrieve footprints."""
    try:
        xdr = transaction.to_xdr()
        response = context["soroban_server"].simulate_transaction(xdr)
        print(f"[SorobanService] Simulation result: {response}")
        return response
    except Exception as e:
        print(f"[SorobanService] Simulation failed: {e}")
        raise

def prepare(context: dict, transaction, simulation):
    """Merge simulation results into the transaction."""
    try:
        prepared_transaction = context["soroban_server"].prepare_transaction(transaction, simulation)
        print(f"[SorobanService] Prepared transaction: {prepared_transaction}")
        return prepared_transaction
    except Exception as e:
        print(f"[SorobanService] Preparation failed: {e}")
        raise

def sign(transaction, secret_key: str):
    """Sign the prepared transaction using the secret key."""
    try:
        keypair = Keypair.from_secret(secret_key)
        transaction.sign(keypair)
        print(f"[SorobanService] Signed transaction.")
        return transaction
    except Exception as e:
        print(f"[SorobanService] Signing failed: {e}")
        raise

def send(context: dict, transaction):
    """Submit the signed transaction to the Soroban network."""
    try:
        send_result = context["soroban_server"].send_transaction(transaction)
        print(f"[SorobanService] Transaction sent. Result: {send_result}")
        return send_result
    except Exception as e:
        print(f"[SorobanService] Sending failed: {e}")
        raise

def add_prices_batch(
    context: dict,
    contract_id: str,
    public_key: str,
    secret_key: str,
    cusips: list,
    prices: list,
    yields: list,
    trade_amounts: list,
    trade_types: list
) -> Dict[str, Any]:
    """
    Add multiple price records to the Soroban contract in a single transaction.
    """
    try:
        server = Server(MAINNET_URL)  # Updated to mainnet
        source_account = server.load_account(public_key)

        # Convert individual elements to proper scval types first
        cusips_vec = [scval.to_string(cusip) for cusip in cusips]
        prices_vec = [scval.to_int128(price) for price in prices]
        yields_vec = [scval.to_int128(y) for y in yields]
        amounts_vec = [scval.to_int128(amt) for amt in trade_amounts]
        types_vec = [scval.to_string(t) for t in trade_types]

        # Create the parameters vector
        params = [
            scval.to_address(Address(public_key)),  # Caller address
            scval.to_vec(cusips_vec),              # CUSIPs
            scval.to_vec(prices_vec),              # Prices (in millicents)
            scval.to_vec(yields_vec),              # Yields (in millibps)
            scval.to_vec(amounts_vec),             # Trade amounts
            scval.to_vec(types_vec)                # Trade types
        ]

        transaction = (
            TransactionBuilder(
                source_account=source_account,
                network_passphrase=Network.PUBLIC_NETWORK_PASSPHRASE,  # Updated to public network
                base_fee=BASE_FEE
            )
            .set_timeout(600)
            .append_invoke_contract_function_op(
                contract_id=contract_id,
                function_name="add_prices_batch",
                parameters=params
            )
            .build()
        )

        print(f"[add_prices_batch] Processing {len(cusips)} prices...")
        simulation_result = simulate(context, transaction)
        prepared_transaction = prepare(context, transaction, simulation_result)
        signed_transaction = sign(prepared_transaction, secret_key)
        response = send(context, signed_transaction)

        if response.status == "SUCCESS":
            return {
                "status": "success",
                "hash": response.hash,
                "cusips": cusips
            }
        else:
            return {
                "status": "error",
                "message": f"Transaction failed with status: {response.status}",
                "hash": response.hash
            }

    except Exception as e:
        print(f"Error in batch price submission: {str(e)}")
        return {"status": "error", "message": str(e)}

def read_price_data(
    context: dict,
    contract_id: str,
    cusip: str,
    public_key: str,
    secret_key: str  # Added for completeness
) -> Dict[str, Any]:
    """
    READ ONLY: Calls get_latest_price but does NOT send.
    Updated for mainnet.
    """
    try:
        server = Server(MAINNET_URL)  # Updated to mainnet
        source_account = server.load_account(public_key)

        transaction = (
            TransactionBuilder(
                source_account=source_account,
                network_passphrase=Network.PUBLIC_NETWORK_PASSPHRASE,  # Updated to public network
                base_fee=BASE_FEE
            )
            .set_timeout(300)
            .append_invoke_contract_function_op(
                contract_id=contract_id,
                function_name="get_cusip_prices",  # Updated to match contract function name
                parameters=[scval.to_string(cusip)]
            )
            .build()
        )

        # SIMULATE ONLY:
        simulation_result = simulate(context, transaction)
        if simulation_result.error:
            return {"success": False, "error": simulation_result.error}

        # The return data is in simulation_result.results[0].xdr
        raw_xdr = simulation_result.results[0].xdr
        # Parse SCVal from raw_xdr as needed:
        sc_val = stellar_xdr.SCVal.from_xdr(raw_xdr)

        # Example: sc_val might have multiple fields. Just returning it raw for demonstration:
        return {
            "success": True,
            "data": str(sc_val)  # or parse in detail
        }
        
    except Exception as e:
        print(f"[SorobanService] Error reading price data: {e}")
        return {
            "success": False,
            "error": str(e)
        }


class FeeTracker:
    def __init__(self):
        self.transactions = []
        
    def add_transaction(self, transaction_data):
        """Add a transaction's fee data"""
        if hasattr(transaction_data, 'transaction'):
            base_fee = getattr(transaction_data.transaction, 'fee', 0)
            resource_fee = 0
            
            # Extract resource fee from soroban_data if it exists
            if hasattr(transaction_data.transaction, 'soroban_data'):
                soroban_data = transaction_data.transaction.soroban_data
                if hasattr(soroban_data, 'resource_fee'):
                    resource_fee = int(soroban_data.resource_fee)
            
            self.transactions.append({
                'base_fee': base_fee,
                'resource_fee': resource_fee,
                'total_fee': base_fee + resource_fee
            })
            
    def add_transaction_fees(self, fee_info):
        """Add transaction fee information"""
        self.transactions.append(fee_info)

    def get_statistics(self):
        """Calculate fee statistics"""
        if not self.transactions:
            return {
                'count': 0,
                'avg_base_fee': 0,
                'avg_resource_fee': 0,
                'avg_total_fee': 0,
                'total_xlm_cost': 0
            }
        
        count = len(self.transactions)
        avg_base = sum(t['base_fee'] for t in self.transactions) / count
        avg_resource = sum(t['resource_fee'] for t in self.transactions) / count
        avg_total = sum(t['total_fee'] for t in self.transactions) / count
        total_xlm = sum(t['total_fee'] for t in self.transactions) / 10000000  # Convert to XLM
        
        return {
            'count': count,
            'avg_base_fee': avg_base / 10000000,  # Convert to XLM
            'avg_resource_fee': avg_resource / 10000000,  # Convert to XLM
            'avg_total_fee': avg_total / 10000000,  # Convert to XLM
            'total_xlm_cost': total_xlm
        }

    def print_summary(self):
        """Print a human-readable summary of fees"""
        stats = self.get_statistics()
        summary = "\n=== Soroban Fee Summary ===\n"
        summary += f"Total Transactions: {stats['count']}\n"
        summary += f"Average Base Fee: {stats['avg_base_fee']:.7f} XLM\n"
        summary += f"Average Resource Fee: {stats['avg_resource_fee']:.7f} XLM\n"
        summary += f"Average Total Fee: {stats['avg_total_fee']:.7f} XLM\n"
        summary += f"Total XLM Cost: {stats['total_xlm_cost']:.7f} XLM\n"
        summary += "==========================\n"
        
        print(summary)
        return summary