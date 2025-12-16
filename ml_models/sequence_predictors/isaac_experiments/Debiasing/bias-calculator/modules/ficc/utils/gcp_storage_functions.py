'''
 '''

import pickle5 as pickle

'''
This function is used to upload data to the cloud bucket
'''
def upload_data(storage_client, bucket_name, file_name):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_name)
    print(f"File {file_name} uploaded to {bucket_name}.")


'''
This function is used to download the data from the GCP storage bucket.
It is assumed that we will be downloading a pickle file
'''
def download_pickle_file(storage_client, bucket_name, file_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    pickle_in = blob.download_as_bytes()
    data = pickle.loads(pickle_in) 
    # print(f"File {file_name} downloaded from {bucket_name}.")
    return data