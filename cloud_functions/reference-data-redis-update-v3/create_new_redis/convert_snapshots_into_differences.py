'''
Description: For each CUSIP in the reference data redis, take the deque of snapshots and convert it into a deque where the 
             first item is a snapshot and each of the subsequent items is a dictionary that contains the difference between 
             the previous snapshot and the current snapshot.

             Use `python -u convert_snapshots_into_differences.py >> output.txt` to print output into a file.
             Use `nohup python -u convert_snapshots_into_differences.py >> output.txt 2>&1 &` to run the above procedure in the background (can close the connection to the VM).
'''
import os
import sys
from collections import deque

from transfer_old_reference_data_to_new_redis import transfer_old_reference_data_to_new_redis


reference_data_redis_update_v2_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))    # get the directory containing the 'cloud_functions/reference_data_redis_v2' package
sys.path.append(reference_data_redis_update_v2_dir)    # add the directory to sys.path


from main import get_differences_between_snapshots


def convert_deque_of_snapshots_into_deque_of_differences(deque_of_snapshots: deque) -> deque:
    '''Convert the `deque_of_snapshots` into a deque where the first snapshot remains intact, 
    but each subsequent item in the deque is a dictionary of differences between the current 
    item and the previous item.'''
    current_snapshot = deque_of_snapshots[0]
    deque_of_differences = deque([current_snapshot])
    for next_snapshot_idx in range(1, len(deque_of_snapshots)):    # starting after the first snapshot since it is used to initialize the new deque
        next_snapshot = deque_of_snapshots[next_snapshot_idx]
        deque_of_differences.append(get_differences_between_snapshots(current_snapshot, next_snapshot))
        current_snapshot = next_snapshot
    return deque_of_differences


if __name__ == '__main__':
    transfer_old_reference_data_to_new_redis(function_to_apply_to_each_value=convert_deque_of_snapshots_into_deque_of_differences)
