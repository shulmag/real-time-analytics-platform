'''
Description: Used to rebuild the reference-data-redis-deque so that every CUSIP is initialized with a deque of one item, 
where that item is the most recents snapshot of the reference data.'''
from collections import deque
import time
from datetime import timedelta
import pickle
# from tqdm import tqdm

import pandas as pd

from main import REFERENCE_DATA_REDIS_CLIENT, REFERENCE_DATA_FEATURES, function_timer, get_data_from_pickle_file_if_query_matches


def get_all_reference_data_query() -> str:
    '''Return a query to select the latest reference data for a given cusip. The condition 
    `ref_valid_to_date > current_datetime("America/New_York")` will retrieve data that is 
    most current. The `ORDER BY cusip` clause allows us to read and debug more clearly since 
    it is in a replicable order.'''
    return f'''SELECT {", ".join(REFERENCE_DATA_FEATURES)} 
               FROM eng-reactor-287421.reference_data_v1.reference_data_flat 
               WHERE cusip IS NOT NULL AND DATETIME(ref_valid_to_date) > CURRENT_DATETIME("America/New_York")
               ORDER BY cusip'''


@function_timer
def get_all_reference_data() -> pd.DataFrame:
    all_reference_data_query = get_all_reference_data_query()
    print(f'Query to get all reference data:\n{all_reference_data_query}')
    return get_data_from_pickle_file_if_query_matches(all_reference_data_query, 'all_reference_data.pkl')    # stores the query and the data in a pickle file for easy retrieval during testing


def main():
    all_reference_data = get_all_reference_data()
    cusips = all_reference_data['cusip'].tolist()
    print(f'Updating the reference data for {len(cusips)} cusips')    # : {sorted(cusips)}''')

    start_time = time.time()
    total_keys_transferred = 0
    BATCH_SIZE = 100    # arbitrary selection
    df_chunks = [all_reference_data[idx : idx + BATCH_SIZE] for idx in range(0, len(all_reference_data), BATCH_SIZE)]
    print(f'Took {timedelta(time.time() - start_time)} seconds to create `df_chunks`')
    start_time_loop = time.time()
    for chunk in df_chunks:    # tqdm(df_chunks, total=len(df_chunks)):
        start_time_chunk = time.time()
        
        with REFERENCE_DATA_REDIS_CLIENT.pipeline() as pipe:
            for row_idx, row in chunk.iterrows():
                cusip = row['cusip']
                pipe.set(cusip, pickle.dumps(deque([row])))
            pipe.execute()
        
        total_keys_transferred += len(chunk)
        end_time_chunk = time.time()
        print(f'Transferred {total_keys_transferred} so far. Most recent chunk took {timedelta(seconds=end_time_chunk - start_time_chunk)} seconds. Total time elapsed for loop: {timedelta(seconds=end_time_chunk - start_time_loop)} seconds')

    print(f'Update complete. Execution time: {timedelta(seconds=time.time() - start_time)}')
    return 'SUCCESS'


if __name__ == '__main__':
    main()
