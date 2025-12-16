import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/shayaan/ahmad_creds.json'

from ftplib import FTP
from google.cloud import storage, bigquery, secretmanager
from pytz import timezone

bq_client = bigquery.Client()
eastern = timezone('US/Eastern')
client = storage.Client()
bucket = client.get_bucket('ref_data_1')

def access_secret_version(project_id, secret_id, version_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    return(payload)

def get_new_files(files_list,ftp,dir='FICCAIFTPH1'):
    ftp.cwd(dir)
    ls_prev = set(files_list)
    ls = set(ftp.nlst())
    add = ls-ls_prev
    return add

def update_ice_files_loading_processing_table(zip_file_name):
    query = """
        INSERT INTO `eng-reactor-287421.reference_data.ice_files_loading_processing` 
        VALUES('""" + zip_file_name + """',0, CURRENT_DATE("US/Eastern"))
    """
    query_job = bq_client.query(query)
    query_job.result()

def main(args):
    HOST = 'ftpprod.icedataservices.com'
    ICE_USER_NAME = access_secret_version('eng-reactor-287421','ice_user_name','latest') 
    ICE_PWD = access_secret_version('eng-reactor-287421','ice_password','latest')


    blobs = client.list_blobs('ref_data_1')
    files_list = [i.name for i in blobs]
    
    ftp = FTP(HOST)
    ftp.login(user=ICE_USER_NAME, passwd = ICE_PWD)
    new_files = get_new_files(files_list, ftp)
    
    for filename in new_files:
        blob = bucket.blob(filename)
        with open('/tmp/' + filename, 'wb+') as f:
            ftp.retrbinary('RETR %s' % filename,f.write)
            f.seek(0)
            blob.upload_from_file(f)
            update_ice_files_loading_processing_table(filename)
    ftp.close()




