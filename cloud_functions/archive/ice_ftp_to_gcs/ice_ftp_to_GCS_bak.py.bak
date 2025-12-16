#import os
#credential_path = "/Users/Gil/git/ficc/data_loader/Cusip Global Service Importer-2fdcdfc4edba.json"
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= credential_path

from ftplib import FTP
from google.cloud import storage, bigquery, secretmanager
from pytz import timezone

# Construct a BigQuery client object.
bq_client = bigquery.Client()
# ET time zone: 
eastern = timezone('US/Eastern')
client = storage.Client()
bucket = client.get_bucket('ref_data_1')

def access_secret_version(project_id, secret_id, version_id):
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()
    # Build the resource name of the secret version.
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    # Access the secret version.
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    return(payload)

host = 'ftpprod.icedataservices.com'
ice_user_name = access_secret_version('eng-reactor-287421','ice_user_name','latest') 
ice_pwd = access_secret_version('eng-reactor-287421','ice_password','latest')

files_list = []
blobs = client.list_blobs('ref_data_1')
for blob in blobs: 
    files_list.append(blob.name)

ftp = FTP(host)
ftp.login(user=ice_user_name, passwd = ice_pwd)

def get_new_files(dir='FICCAIFTPH1'):
    ftp.cwd(dir)
    ls_prev = set(files_list)

    ls = set(ftp.nlst())
    add = ls-ls_prev
    #yield is a keyword in Python that is used to return from a function without destroying the states of its local variable
    return add

def update_ice_files_loading_processing_table(zip_file_name):
    query = """
        INSERT INTO `eng-reactor-287421.reference_data.ice_files_loading_processing` 
        VALUES('""" + zip_file_name + """',0, CURRENT_DATE("US/Eastern"))
    """
    query_job = bq_client.query(query)
    query_job.result()

def main(args):
    new_files = get_new_files()
    for filename in new_files:
        blob = bucket.blob(filename)
        with open('/tmp/' + filename, 'wb+') as f:
            ftp.retrbinary('RETR %s' % filename,f.write)
            f.seek(0)
            blob.upload_from_file(f)
            update_ice_files_loading_processing_table(filename)
    ftp.close()