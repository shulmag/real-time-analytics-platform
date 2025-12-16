'''
Description: Used to rebuild the reference-data-redis-deque so that every CUSIP is initialized with a deque of one item, 
             where that item is the most recents snapshot of the reference data.
             Use `python -u rebuild_redis.py >> output.txt` to print output into a file.
             Use `nohup python -u rebuild_redis.py >> output.txt 2>&1 &` to run the above procedure in the background (can close the connection to the VM).'''
import os
import sys

import pandas as pd


reference_data_redis_update_v2_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))    # get the directory containing the 'cloud_functions/reference_data_redis_v2' package
sys.path.append(reference_data_redis_update_v2_dir)    # add the directory to sys.path


from main import REFERENCE_DATA_TABLE_NAME, REFERENCE_DATA_FEATURES, function_timer, get_data_from_pickle_file_if_query_matches, update_reference_data_redis


def get_all_reference_data_query() -> str:
    '''Return a query to select the latest reference data for a given cusip. The condition 
    `ref_valid_to_date > current_datetime("America/New_York")` will retrieve data that is 
    most current. The `ORDER BY cusip` clause allows us to read and debug more clearly since 
    it is in a replicable order.'''
    return f'''SELECT {", ".join(REFERENCE_DATA_FEATURES)} 
               FROM {REFERENCE_DATA_TABLE_NAME} 
               WHERE cusip IS NOT NULL AND DATETIME(ref_valid_to_date) > CURRENT_DATETIME("America/New_York")
               ORDER BY cusip'''


@function_timer
def get_all_reference_data() -> pd.DataFrame:
    all_reference_data_query = get_all_reference_data_query()
    print(f'Query to get all reference data:\n{all_reference_data_query}')
    return get_data_from_pickle_file_if_query_matches(all_reference_data_query, 'all_reference_data.pkl')    # stores the query and the data in a pickle file for easy retrieval during testing


def main():
    all_reference_data = get_all_reference_data()
    update_reference_data_redis(all_reference_data, new_entries_only=True, verbose=True)
    return 'SUCCESS'


if __name__ == '__main__':
    main()
