'''
Filepath: /ficc/cloud_functions/create_redis_snapshots/create_redis_snapshots.py
Description: Export Redis snapshots concurrently and deletes old snapshots.
'''
from datetime import datetime, timedelta
import concurrent.futures    # for concurrency
from pytz import timezone

from google.cloud import redis_v1
from google.cloud import storage

from auxiliary_functions import run_five_times_before_raising_redis_connector_error


EASTERN = timezone('US/Eastern')
CURRENT_DATETIME = datetime.now(EASTERN)

YEAR_MONTH_DAY = '%Y%m%d'
YEAR_MONTH_DAY_HOUR_MINUTE = f'{YEAR_MONTH_DAY}-%H%M'
DAYS_TO_KEEP = 7    # number of days in the past to keep redis snapshots

PROJECT_ID = 'eng-reactor-287421'
REDIS_LOCATION = ['us-central1', 'us-east1']

BUCKET_NAME_TO_STORE_SNAPSHOTS = 'ficc_redis_snapshots'


def get_redis_instances(project_id: str = PROJECT_ID, locations: list = REDIS_LOCATION, cloud_redis_client=None) -> list:
    '''Get all redis instances names from `location`.'''
    if cloud_redis_client is None: cloud_redis_client = redis_v1.CloudRedisClient()    # set up Redis client
    instance_names = []
    # Loop through every location to get all instances
    for location in locations:
        parent = f'projects/{project_id}/locations/{location}'
        instances = cloud_redis_client.list_instances(request={'parent': parent}).instances
        instance_names.extend([instance.name for instance in instances])

    return instance_names


def export_redis_snapshot(instance_name: str, client, current_date: str, current_datetime: str):
    instance_id = instance_name.split('/')[-1].replace('-', '_')
    output_config = {'gcs_destination': {'uri': f'gs://{BUCKET_NAME_TO_STORE_SNAPSHOTS}/{instance_id}/{current_date}/{current_datetime}-redis-snapshot.rdb'}}
    operation = client.export_instance(name=instance_name, output_config=output_config)    # call the Redis API to start the export
    
    # Wait for operation to complete
    response = operation.result()    # avoid printing `response` since it is an extremely verbose description of the redis instance
    print(f'Export completed successfully for {instance_id}')
    return f'Success for {instance_id}'


def delete_old_snapshots(bucket_name: str, days_to_keep: int):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()    # List all objects in the bucket

    # Loop through blobs to check and delete old snapshots
    for blob in blobs:
        path_parts = blob.name.split('/')    # Extract the directory or file path
        if len(path_parts) > 1:    # Ensures it's not the root directory
            date_part = path_parts[1]    # Assuming second part is the date
            
            try:
                if len(date_part) == 8:    # Format: YYYYMMDD
                    snapshot_date = datetime.strptime(date_part, YEAR_MONTH_DAY)
                else:    # Skip if it does not match the expected format
                    print(f'Skipping date: {date_part}, since it does not match the expected format of {YEAR_MONTH_DAY}')
                    continue

                if CURRENT_DATETIME.replace(tzinfo=None) - snapshot_date > timedelta(days=days_to_keep):    # .replace(tzinfo=None) makes the `CURRENT_DATETIME` timezone naive, because the string from the snapshot is timestamp naive
                    print(f'Deleting old snapshot: {blob.name}')
                    blob.delete()
                    
            except ValueError as e:    # Skip any malformed date strings
                print(f'Skipping invalid date: {date_part} from {type(e)}: {e}')
                continue

@run_five_times_before_raising_redis_connector_error
def main(requests):
    # Get the current date and time for the folder and file names
    current_date_string = CURRENT_DATETIME.strftime(YEAR_MONTH_DAY)
    current_datetime_string = CURRENT_DATETIME.strftime(f'{YEAR_MONTH_DAY_HOUR_MINUTE}')

    client = redis_v1.CloudRedisClient()    # set up Redis client
    instances = get_redis_instances(cloud_redis_client=client)
    with concurrent.futures.ThreadPoolExecutor() as executor:    # use a thread pool to execute the exports concurrently
        futures = {executor.submit(export_redis_snapshot, instance_name, client, current_date_string, current_datetime_string): instance_name for instance_name in instances}    # submit export tasks concurrently
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    successful_exports, failed_exports = [], []
    for instance_name, result in zip(futures.values(), results):
        if 'Failed' in result:
            failed_exports.append(instance_name)
        else:
            successful_exports.append(instance_name)
    delete_old_snapshots(BUCKET_NAME_TO_STORE_SNAPSHOTS, DAYS_TO_KEEP)
    
    if failed_exports:
        return f'Successful exports: {successful_exports}. Failed exports: {failed_exports}.'
    else:
        return f'All exports, {successful_exports}, completed successfully (or after retries).'