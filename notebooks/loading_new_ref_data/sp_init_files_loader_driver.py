import os
import requests
from google.cloud import storage, bigquery
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import google.auth
import google.oauth2.id_token

# Replace these constants with your actual values
GCP_PROJECT = 'eng-reactor-287421'
GCS_BUCKET_NAME = 'sp_ref_data'
GCS_DIRECTORY = '11082024_init_file/'
CLOUD_FUNCTION_URL = 'https://us-central1-eng-reactor-287421.cloudfunctions.net/init_sp_xml_to_sp_nested'
PROCESSED_FILES_TABLE = 'eng-reactor-287421.sp_reference_data.processed_files'

# Initialize clients
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'
storage_client = storage.Client(project=GCP_PROJECT)
bigquery_client = bigquery.Client(project=GCP_PROJECT)

def get_xml_files():
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=GCS_DIRECTORY)
    xml_files = [blob.name for blob in blobs if blob.name.endswith('.xml')]
    print(f"Sample XML files: {xml_files[:5]}")
    return xml_files

def get_processed_files():
    query = f"""
    SELECT file_name
    FROM `{PROCESSED_FILES_TABLE}`
    """
    query_job = bigquery_client.query(query)
    result = query_job.result()
    processed_files = set(row.file_name for row in result)
    print(f"Sample processed file names from BigQuery: {list(processed_files)[:5]}")
    return processed_files

def main():
    xml_files = get_xml_files()
    print(f"Total XML files found: {len(xml_files)}")
    if not xml_files:
        print("No XML files found in GCS.")
        return

    # Extract file names without directory paths
    xml_file_names = [os.path.basename(f) for f in xml_files]

    processed_files = get_processed_files()
    print(f"Total processed files in BigQuery: {len(processed_files)}")

    unprocessed_files = [f for f, fname in zip(xml_files, xml_file_names) if fname not in processed_files]
    total_files = len(unprocessed_files)
    print(f"Total unprocessed files: {total_files}")

    batch_size = 8  # Start with 1, increase as needed

    # Create an authorized request object
    auth_req = Request()

    # Load credentials from the service account file
    credentials = service_account.IDTokenCredentials.from_service_account_file(
        '/Users/gil/git/ficc/creds.json',
        target_audience=CLOUD_FUNCTION_URL
    )

    for i in range(0, total_files, batch_size):
        batch = unprocessed_files[i:i+batch_size]
        for file_name in batch:
            payload = {'file_name': file_name}
            try:
                # Refresh credentials to get an ID token
                credentials.refresh(auth_req)
                id_token = credentials.token

                headers = {
                    'Authorization': f'Bearer {id_token}'
                }

                response = requests.post(
                    CLOUD_FUNCTION_URL,
                    json=payload,
                    headers=headers,
                    timeout=540  # Cloud Function maximum timeout
                )
                if response.status_code == 200:
                    print(f"Processed {file_name} successfully.")
                else:
                    print(f"Failed to process {file_name}. Status code: {response.status_code}. Response: {response.text}")
            except Exception as e:
                print(f"Exception occurred while processing {file_name}: {e}")

if __name__ == "__main__":
    main()
