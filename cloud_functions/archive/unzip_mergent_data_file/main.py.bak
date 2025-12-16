from zipfile import ZipFile
from zipfile import is_zipfile
from google.cloud import storage
from io import StringIO
import io


def main(event, context):
    file_name = event['name']

    # def zipextract(bucketname, zipfilename_with_path):

    storage_client = storage.Client()
    bucket = storage_client.get_bucket('mergent_data')

    blob = bucket.blob(file_name)
    zipbytes = io.BytesIO(blob.download_as_string())

    if is_zipfile(zipbytes):
        with ZipFile(zipbytes, 'r') as myzip:
            for contentfilename in myzip.namelist():
                contentfile = myzip.read(contentfilename)
                blob = bucket.blob(contentfilename)
                blob.upload_from_string(contentfile)
