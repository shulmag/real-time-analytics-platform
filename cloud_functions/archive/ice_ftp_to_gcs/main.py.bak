'''Hits ICE API and gets XML file 3 times a day. This XML file contains the reference data that is then processed by 
the cloud function `ice_gcs_to_bq` and used for training / testing the model and for production. This is then stored 
in to a Google Cloud Storage bucket. TODO: combine this cloud function with `ice_gs_to_bq` and the uploading of the 
reference data to the reference data redis into one sequence, to prevent lag between these subprocedures.'''
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
    return payload


def get_new_files(files_list, ftp, dir='FICCAIFTPH1'):
    ftp.cwd(dir)
    ls_prev = set(files_list)
    ls = set(ftp.nlst())
    add = ls - ls_prev
    return add


def update_ice_files_loading_processing_table(zip_file_name):
    query = (
        """
      INSERT INTO `eng-reactor-287421.reference_data.ice_files_loading_processing` 
      VALUES('"""
        + zip_file_name
        + """',0, CURRENT_DATE("US/Eastern"))
  """
    )
    query_job = bq_client.query(query)
    query_job.result()


def main(args):
    '''
    Moving to the host to new host from ice
    '''
    # HOST = 'ftpprod.icedataservices.com'
    HOST = 'ftp.pna.icedataservices.com'
    ICE_USER_NAME = access_secret_version(
        'eng-reactor-287421', 'ice_user_name', 'latest'
    )
    ICE_PWD = access_secret_version('eng-reactor-287421', 'ice_password', 'latest')

    blobs = client.list_blobs('ref_data_1')
    files_list = [i.name for i in blobs]
    # files_list = ["gsm_update_muni_APFICC_GSMF10I.2217.1_1.20231004T1400-04.xml.gz"]

    ftp = FTP(HOST)
    ftp.login(user=ICE_USER_NAME, passwd=ICE_PWD)
    new_files = get_new_files(files_list, ftp)

    print(f"Uploading new files\n {new_files}")

    for filename in new_files:
        blob = bucket.blob(filename)
        with open('/tmp/' + filename, 'wb+') as f:
            ftp.retrbinary('RETR %s' % filename, f.write)
            f.seek(0)
            blob.upload_from_file(f)
            update_ice_files_loading_processing_table(filename)
    ftp.close()

    return "SUCCESS"
