import functions_framework
from google.cloud import storage, bigquery, secretmanager
from pytz import timezone
import paramiko
from io import BytesIO

# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'

bq_client = bigquery.Client()
eastern = timezone('US/Eastern')
client = storage.Client()
bucket = client.get_bucket('sp_ref_data')

def access_secret_version(project_id, secret_id, version_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    return payload

def get_new_files(files_list, sftp, dir='Municipal'):
    sftp.chdir(dir)
    ls_prev = set(files_list)
    ls = set(sftp.listdir())
    add = ls - ls_prev
    return add

def update_sp_files_loading_processing_table(zip_file_name):
    query = (
        f"""
      INSERT INTO `eng-reactor-287421.reference_data.sp_loading_processing` 
      VALUES('{zip_file_name}', 0, CURRENT_DATE("US/Eastern"))
  """
    )
    query_job = bq_client.query(query)
    query_job.result()

def main(args):
    '''
    Moving to the new host from S&P
    '''
    
    HOST = 'sftp1.spglobal.com'
    SP_USERNAME = access_secret_version(
        'eng-reactor-287421', 'sp_username', 'latest'
    )
    SP_PWD = access_secret_version('eng-reactor-287421', 'sp_password', 'latest')

    blobs = client.list_blobs('sp_ref_data')
    files_list = [i.name for i in blobs]

    transport = paramiko.Transport((HOST, 22))
    transport.connect(username=SP_USERNAME, password=SP_PWD)
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        new_files = get_new_files(files_list, sftp)

        print(f"Uploading new files\n {new_files}")

        for filename in new_files:
            print(filename)  # Ensure filename is correct
            blob = bucket.blob(filename)
            file_obj = BytesIO()
            try:
                sftp.getfo(filename, file_obj)
                file_obj.seek(0)
                blob.upload_from_file(file_obj, timeout=3600)  # Set timeout to 60 minutes for Google Cloud Storage upload
                update_sp_files_loading_processing_table(filename)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    finally:
        sftp.close()
        transport.close()

    return "SUCCESS"

if __name__ == "__main__":
    main('f')
