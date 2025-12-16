from stellar_sdk import xdr as stellar_xdr
import re 

def parse_latest_prices_xdr(raw_xdr: str):
    """
    Parse the raw XDR string returned by get_all_latest_prices into Python dicts.
    """
    if not raw_xdr:
        print("No XDR data provided")
        return []
    
    try:
        print(f"Starting XDR parse of: {raw_xdr[:100]}...")
        
        # Convert string XDR -> SCVal object
        sc_val = stellar_xdr.SCVal.from_xdr(raw_xdr)
        # print(f"SCVal type: {sc_val.type}")
        
        results = []
        
        # For type 16 (SCV_VEC), use sc_val.vec
        if sc_val.type == stellar_xdr.SCValType.SCV_VEC:
            # print("Processing SCV_VEC...")
            for item in sc_val.vec.sc_vec:
                # print(f"Processing vector item...")
                try:
                    if item.type != stellar_xdr.SCValType.SCV_MAP:
                        continue

                    entry = {}
                    map_entries = item.map.sc_map
                    
                    # First level: find cusip and latest_price
                    for map_entry in map_entries:
                        key = map_entry.key
                        val = map_entry.val
                        
                        if key.type == stellar_xdr.SCValType.SCV_SYMBOL:
                            key_str = key.sym.sc_symbol.decode()
                            # print(f"Found key: {key_str}")
                            
                            if key_str == "cusip":
                                entry["cusip"] = val.str.sc_string.decode()
                            elif key_str == "latest_price":
                                # Process the price data map
                                price_entries = val.map.sc_map
                                for price_entry in price_entries:
                                    price_key = price_entry.key.sym.sc_symbol.decode()
                                    price_val = price_entry.val
                                    
                                    if price_key == "price":
                                        entry["price"] = float(price_val.i128.lo.uint64) / 1000.0
                                    elif price_key == "yield_value":
                                        entry["yield_value"] = float(price_val.i128.lo.uint64) / 1000.0
                                    elif price_key == "trade_amount":
                                        entry["trade_amount"] = int(price_val.i128.lo.uint64)
                                    elif price_key == "trade_type":
                                        entry["trade_type"] = price_val.str.sc_string.decode()
                                    elif price_key == "timestamp":
                                        entry["timestamp"] = int(price_val.u64.uint64)
                    
                    if entry and "cusip" in entry:
                        # print(f"Adding entry for CUSIP: {entry['cusip']}")
                        results.append(entry)
                        
                except Exception as e:
                    print(f"Error processing item: {e}")
                    continue

        print(f"Successfully parsed {len(results)} price entries")
        return results

    except Exception as e:
        print(f"Error in main XDR parsing: {str(e)}")
        print(f"Raw XDR (first 200 chars): {raw_xdr[:200]}")
        return []
    
# In xdr_parser.py

def parse_price_history_xdr(raw_xdr: str):
    """Parse price history from stringified XDR representation."""
    if not raw_xdr:
        print("No XDR data provided for history")
        return []
    
    try:
        print(f"Starting history parse of: {raw_xdr[:100]}...")
        results = []

        # Handle string format that starts with "<SCVal"
        if raw_xdr.startswith('<SCVal'):
            print("Processing stringified SCVal...")

            # Find all map entries (each is a historical price entry)
            map_sections = raw_xdr.split('<SCVal [type=17, map=')[1:]  # Skip first split which is header
            
            for section in map_sections:
                try:
                    entry = {}
                    
                    # Extract price
                    price_match = re.search(r'price.*?uint64=(\d+)', section)
                    if price_match:
                        entry['price'] = float(price_match.group(1)) / 1000.0

                    # Extract timestamp
                    timestamp_match = re.search(r'timestamp.*?uint64=(\d+)', section)
                    if timestamp_match:
                        entry['timestamp'] = int(timestamp_match.group(1))

                    # Extract trade amount
                    trade_amount_match = re.search(r'trade_amount.*?uint64=(\d+)', section)
                    if trade_amount_match:
                        entry['trade_amount'] = int(trade_amount_match.group(1))

                    # Extract trade type
                    trade_type_match = re.search(r'trade_type.*?sc_string=b\'([^\']+)\'', section)
                    if trade_type_match:
                        entry['trade_type'] = trade_type_match.group(1)

                    # Extract yield value
                    yield_match = re.search(r'yield_value.*?uint64=(\d+)', section)
                    if yield_match:
                        entry['yield_value'] = float(yield_match.group(1)) / 1000.0

                    if entry and 'price' in entry:  # Only add if we have at least a price
                        print(f"Adding history entry: price={entry['price']}, timestamp={entry.get('timestamp')}")
                        results.append(entry)

                except Exception as e:
                    print(f"Error processing price entry: {e}")
                    continue

        results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)  # Most recent first
        print(f"Successfully parsed {len(results)} history entries")
        return results

    except Exception as e:
        print(f"Error parsing price history: {str(e)}")
        return []