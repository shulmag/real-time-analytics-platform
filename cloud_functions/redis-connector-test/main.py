'''
Description: Pings the yield curve redis, reference data redis, trade history redis, and the similar 
             trade history redis to ensure that a given VPC connector is functioning correctly and allowing us to 
             access these redis instances.'''
import redis


def test_access_to_redis(redis_client, key, redis_client_name: str = None, num_calls: int = 1000) -> None:
    if redis_client_name is not None: print(f'Testing {redis_client_name} redis with key: {key} for {num_calls} calls')
    for call_idx in range(1, num_calls + 1):
        try:
            redis_client.get(key)
            if call_idx % 50 == 0: print(f'{call_idx} calls successful')
        except Exception as e:
            print(call_idx, e)


def main(args):
    yc_client = redis.Redis(host='10.227.69.60', port=6379, db=0)    # must use primary endpoint since there is no read endpoint since the redis instance capacity is only 1 GB
    ref_client = redis.Redis(host='10.108.4.36', port=6379, db=0)    # use read endpoint since use case is read-only allowing for lower latency and to not accidentally corrupt the redis by attempting to write to it
    trade_history_client = redis.Redis(host='10.75.46.229', port=6379, db=0)    # use read endpoint since use case is read-only allowing for lower latency and to not accidentally corrupt the redis by attempting to write to it
    similar_trade_history_client = redis.Redis(host='10.117.191.181', port=6379, db=0)    # use read endpoint since use case is read-only allowing for lower latency and to not accidentally corrupt the redis by attempting to write to it
    
    ARBITRARY_CUSIP = '64971XQM3'
    ARBITRARY_DATETIME = '2024-01-03 09:30:00'
    ARBITRARY_KEY_FOR_SIMILAR_TRADE_HISTORY_REDIS = similar_trade_history_client.randomkey()
    print(f'Using the following key to test access to the similar trade history redis: {ARBITRARY_KEY_FOR_SIMILAR_TRADE_HISTORY_REDIS}')
    
    test_access_to_redis(yc_client, ARBITRARY_DATETIME, 'yield-curve')
    test_access_to_redis(ref_client, ARBITRARY_CUSIP, 'reference-data-deque-v3')
    test_access_to_redis(trade_history_client, ARBITRARY_CUSIP, 'trade-history-v2')
    test_access_to_redis(similar_trade_history_client, ARBITRARY_KEY_FOR_SIMILAR_TRADE_HISTORY_REDIS, 'similar-trade-history-v2')

    return 'SUCCESS'
