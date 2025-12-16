import os
import paramiko
from google.cloud import storage, secretmanager

# Constants
PROJECT_ID = 'eng-reactor-287421'
BUCKET_NAME = 'cusip_global_services'
ISSUER_FOLDER = 'Issuer_R_Files/'
ISSUES_FOLDER = 'Issues_E_Files/'
FTP_HOST = 'sftp.cusip.com'
FTP_PORT = 22

def access_secret_version(secret_id, version_id='latest'):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def get_sftp_connection():
    username = access_secret_version('cusip_username')
    password = access_secret_version('cusip_password')
    
    transport = paramiko.Transport((FTP_HOST, FTP_PORT))
    transport.connect(username=username, password=password)
    return paramiko.SFTPClient.from_transport(transport)

def get_existing_files(storage_client, folder):
    blobs = storage_client.list_blobs(BUCKET_NAME, prefix=folder)
    return set(blob.name.split('/')[-1] for blob in blobs)

def upload_to_gcs(storage_client, local_path, gcs_path):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"File {local_path} uploaded to {gcs_path}")

def cusip_file_transfer(args):
    storage_client = storage.Client()
    sftp = get_sftp_connection()
    
    try:
        sftp.chdir('/Inbox')
        print(f"Current SFTP directory: {sftp.getcwd()}")
        
        # Get existing files in GCS
        existing_r_files = get_existing_files(storage_client, ISSUER_FOLDER)
        existing_e_files = get_existing_files(storage_client, ISSUES_FOLDER)
        
        print("Existing R files in GCS:", existing_r_files)
        print("Existing E files in GCS:", existing_e_files)
        
        # List and process all files in the current directory
        for file in sftp.listdir():
            if file.endswith('R.PIP'):
                if file not in existing_r_files:
                    local_path = f"/tmp/{file}"
                    sftp.get(file, local_path)
                    upload_to_gcs(storage_client, local_path, f"{ISSUER_FOLDER}{file}")
                    os.remove(local_path)
                    print(f"Processed new R file: {file}")
                else:
                    print(f"Skipping R file (already exists in GCS): {file}")
            elif file.endswith('E.PIP'):
                if file not in existing_e_files:
                    local_path = f"/tmp/{file}"
                    sftp.get(file, local_path)
                    upload_to_gcs(storage_client, local_path, f"{ISSUES_FOLDER}{file}")
                    os.remove(local_path)
                    print(f"Processed new E file: {file}")
                else:
                    print(f"Skipping E file (already exists in GCS): {file}")
            else:
                print(f"Skipping file (not relevant): {file}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        sftp.close()
    
    return "File transfer process completed."
