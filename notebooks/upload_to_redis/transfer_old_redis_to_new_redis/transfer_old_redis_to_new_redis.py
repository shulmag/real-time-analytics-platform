'''
Description: Transfer data from one redis to another redis. This can be used when creating a new redis that 
             should have the same content as another redis.
             Use `python -u transfer_old_redis_to_new_redis.py >> output.txt` to print output into a file.
             Use `nohup python -u transfer_old_redis_to_new_redis.py >> output.txt 2>&1 &` to run the above procedure in the background (can close the connection to the VM).
'''
import os
import sys
import time
from datetime import timedelta
import redis


server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'app_engine', 'demo', 'server'))    # get the directory containing the 'app_engine/demo/server' package
sys.path.append(server_dir)    # add the directory to sys.path

from modules.issue_key_map import ISSUE_KEY_MAP


OLD_REDIS_CLIENT = redis.Redis(host=None, port=6379, db=0)    # use read endpoint since use case is read-only allowing for lower latency and to not accidentally corrupt the redis by attempting to write to it
NEW_REDIS_CLIENT = redis.Redis(host=None, port=6379, db=0)

BATCH_SIZE = 1000    # larger batch size means faster cumulative transfer since there are fewer round trips to the redis, but requires more memory


def update_issue_key_for_similar_trades_redis_key(similar_trades_redis_key_with_issue_key_as_integer: bytes):
    '''Assumes that `similar_trades_redis_key_with_issue_key_as_integer` is of the form: `<issue_key>_<years_to_maturity_date_by_5>_<coupon_by_1>`. 
    For more context, see `app_engine/demo/server/modules/similar_trade_history.py::get_similar_trade_history_group(...)`.'''
    similar_trades_redis_key_with_issue_key_as_integer = similar_trades_redis_key_with_issue_key_as_integer.decode('utf-8')    # convert bytes to string
    first_underscore_position = similar_trades_redis_key_with_issue_key_as_integer.find('_')
    assert first_underscore_position != -1, 'Underscore not found, but assuming that the key is of the form: `<issue_key>_<years_to_maturity_date_by_5>_<coupon_by_1>`'
    issue_key, rest_of_similar_trades_redis_key = similar_trades_redis_key_with_issue_key_as_integer[:first_underscore_position], similar_trades_redis_key_with_issue_key_as_integer[first_underscore_position:]
    issue_key = int(issue_key)
    if issue_key in ISSUE_KEY_MAP:
        issue_key = ISSUE_KEY_MAP[issue_key]
    else:    # use the old `issue_key` to fill in if the key is not found in the map; perhaps the `issue_key` is for an issue with CUSIPs no longer outstanding
        print(f'Issue key: {issue_key} not found in `ficc/app_engine/demo/server/modules/issue_key_map.py::ISSUE_KEY_MAP`')
    return f'{issue_key}{rest_of_similar_trades_redis_key}'


def main(old_redis_client=OLD_REDIS_CLIENT, new_redis_client=NEW_REDIS_CLIENT, key_func=lambda x: x):
    '''`key_func` is a one-argument function that takes a key from the old redis as input and returns 
    a modified key that will be the key in the new redis; it is initialized to the identity function.'''
    num_keys_in_old_redis = old_redis_client.dbsize()
    print(f'Number of keys in old redis to be transferred: {num_keys_in_old_redis}')
    
    new_redis_client.flushdb()    # clear the new redis to make sure that we are starting from scratch

    cursor = 0
    total_keys_transferred = 0

    start_time = time.time()
    while True:
        cursor, keys = old_redis_client.scan(cursor=cursor, match='*', count=BATCH_SIZE)
        num_keys = len(keys)
        
        if keys:
            values = old_redis_client.mget(keys)    # Use .mget to get all values for the batch of keys
            
            # Use a pipeline to batch set operations in the destination
            with new_redis_client.pipeline() as pipe:
                for key, value in zip(keys, values):
                    pipe.set(key_func(key), value)
                pipe.execute()

        total_keys_transferred += num_keys
        print(f'Transferred {num_keys} keys in this batch. {total_keys_transferred} keys transferred so far in {timedelta(seconds=time.time() - start_time)}')
        
        if cursor == 0:
            break

    print(f'Data transfer complete. Execution time: {timedelta(seconds=time.time() - start_time)}')


if __name__ == '__main__':
    main()
    # main(key_func=update_issue_key_for_similar_trades_redis_key)
