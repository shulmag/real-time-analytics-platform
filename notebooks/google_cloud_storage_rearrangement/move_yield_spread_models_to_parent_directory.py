'''
Description: This script moves all archived yield spread models in the Google Cloud bucket `automated_training` 
             to a parent directory called `yield_spread_models`.
'''
import os
import sys

from google.cloud import storage

server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'app_engine', 'demo', 'server'))    # get the directory containing the 'app_engine/demo/server' package
sys.path.append(server_dir)    # add the directory to sys.path
print('NOTE: this file must be run from the `notebooks/google_cloud_storage_rearrangement/` directory')

from modules.get_creds import get_creds


get_creds()


def move_directories_with_prefix(bucket_name: str, source_prefix: str, destination_parent_folder: str) -> None:
    '''Moves all objects in directories with a given `source_prefix` to a `destination_parent_folder` in Google Cloud bucket `bucket_name`.'''
    # Initialize the GCS client
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=source_prefix)
    
    for blob in blobs:
        # Generate the new destination path
        source_path = blob.name
        destination_path = f'{destination_parent_folder}/{source_path}'
        
        # Copy the blob to the new location
        new_blob = bucket.copy_blob(blob, bucket, destination_path)
        print(f'Copied {source_path} to {new_blob.name}')
        
        # Delete the original blob
        blob.delete()
        print(f'Deleted original blob: {source_path}')


if __name__ == '__main__':
    bucket_name = 'automated_training'
    source_prefix = 'model-'
    destination_parent_folder = 'yield_spread_model'

    move_directories_with_prefix(bucket_name, source_prefix, destination_parent_folder)
