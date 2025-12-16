from google.cloud import bigquery, storage
import os
import csv
import xmltodict
import json
import gzip
import re
from datetime import datetime

# need credentials from VM
credentials_path = "/home/gil/data_loader/ficc_dev-b15c6ab86e26.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# pull schema information as stored in ice_bq_map for functions that conform xmls to bigquery schema
# generate list of xmls and pass to json creater, then load to table

PROJECT = 'eng-reactor-287421'
dataset = 'reference_data'
data_table = 'ice_nested'
map_table = 'ice_bq_map'

bqclient = bigquery.Client(project=PROJECT,)

# get mapping schema from bigquery
map_query = """SELECT * FROM {}.{}""".format(dataset, map_table)

bq_map_result = bqclient.query(map_query).result()
bq_map_result = list(bq_map_result)
ice_schema, map_version, leaf_nodes, repeated_record_nodes, address_key_list, bad_address_list = bq_map_result[
    0]
leaf_nodes = [[entry['path'], entry['type'], entry['mode']]
              for entry in leaf_nodes]
address_key_list = [[entry['path'], entry['key']]
                    for entry in address_key_list]
repeated_leafs = [entry[0] for entry in leaf_nodes if entry[2] == 'REPEATED']
all_repeated_nodes = repeated_leafs + repeated_record_nodes


def repeated_enforcement(address_list, xml_dict):
# address_list is '/' separated string of nodes address
# xml_dict is ice instrument in python dict
# function checks all repeated nodes in address, if in the instrument, checks if instrument is wrapped in a list
# if not, entry is wrapped in a list

    split_address_list = [entry.split('/')[1:] for entry in address_list]

    for address in split_address_list:
        current_xml_level = xml_dict
        for current_index in range(len(address)-1):
            if address[current_index] not in current_xml_level:
                break
            current_xml_level = current_xml_level[address[current_index]]

        if address[-1] not in current_xml_level:
            continue
        else:
            if not isinstance(current_xml_level[address[-1]], list):
                current_xml_level[address[-1]
                                  ] = [current_xml_level[address[-1]]]


def record_key_flatten(address_key_list, xml_dict):
# address_key_list contains '/' separated string of nodes address and the key value to extract
# key value in child node of node in address_key_list is extracted and put as value for node    

    split_address_key_list = [
        [entry[0].split('/')[1:], entry[1]] for entry in address_key_list]

    for entry in split_address_key_list:
        address, key = entry
        current_xml_level = xml_dict
        for current_index in range(len(address)-1):
            if address[current_index] not in current_xml_level:
                break
            current_xml_level = current_xml_level[address[current_index]]

        if address[-1] not in current_xml_level:
            continue
        else:
            if isinstance(current_xml_level[address[-1]], dict):
                current_xml_level[address[-1]
                                  ] = current_xml_level[address[-1]][key]


def derepeat_bad_fields(bad_address_list, xml_dict):
# this is a one off on a poorly handled field in ICE data
# usually not repeated, when repeated, just take the first entry in list
    
    split_bad_address_list = [entry.split(
        '/')[1:] for entry in bad_address_list]

    for address in split_bad_address_list:
        current_xml_level = xml_dict
        for current_index in range(len(address)-1):
            if address[current_index] not in current_xml_level:
                break
            current_xml_level = current_xml_level[address[current_index]]

        if address[-1] not in current_xml_level:
            continue
        else:
            if isinstance(current_xml_level[address[-1]], list):
                current_xml_level[address[-1]
                                  ] = current_xml_level[address[-1]][0]


def get_timestamp_from_file_name(file_name):
    #regex to pull timestamp from file name
    date = re.search(
        '([0-9]{4}[0-9]{2}[0-9]{2}T[0-9]{4}-[0-9]{2})', file_name)[0]
    date = date.replace('-', '')
    return(datetime.strftime(datetime.strptime(date, '%Y%m%dT%H%M%S'), '%Y-%m-%dT%H:%M:%SZ'))


def xmls_list_generator(gz_file_name):
# with zip file name download to temp file, open and read through about 10mb of zip file size
# parse to list of xml in string format and pass each iteration through generator

    storage_client = storage.Client()
    bucket = storage_client.bucket('ref_data_1')
    # e.g. 'gsm_update_muni_APFICC_GSMF10I.35.1_1.20201221T0800-05.xml.gz')
    blob = bucket.get_blob(gz_file_name)
    file = blob.download_to_filename('/tmp/temp_gz_file.xml.gz')

    start = True
    last_partial_entry = ''
    keep_reading = True
    with gzip.open('/tmp/temp_gz_file.xml.gz', 'rt') as f:
        while keep_reading:
            # read_length equal to a zip file size of about 15mb
            read_length = int(1e8)
            file_str = f.read(read_length)

            file_str = last_partial_entry + file_str
            if len(file_str)<read_length:
                file_str = file_str.split('</payload>')[0]
            if start:
                file_str = file_str.split('<payload>')[1]
                start = False

            xml_list = file_str.split('</instrument>')
            #save last entry of the list for next loop, will be empty if string cleanly ends on an xml instrument
            last_partial_entry = xml_list.pop()
            
            if not file_str:
                keep_reading = False
                
           
            yield xml_list



def load_nested_table_uri_json(table_id, uri):
# load from created json file into main table

    bq_client = bigquery.Client()
    ice_nested_table = bqclient.get_table(table_id)

    job_config = bigquery.LoadJobConfig(
        schema=ice_nested_table.schema,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    )

    # Make an API request.
    load_job = bq_client.load_table_from_uri(uri, table_id, job_config=job_config)
    load_job.result()
    


def get_nested_ndjson_as_uri(file_name, xmls_generator, timestamp):
# with file name for save, xmls list generator and timestamp for recording time of upload to table

    json_file_name = '%snest.json' % (file_name)

    # save memory by writing one json string to file at a time rather than store to a variable
    with open('/tmp/%s' % (json_file_name), 'wb+') as f:
        
        # loop through capped size xml lists
        for xmls in xmls_generator:
            for xml_entry in xmls:
                xml_str = xml_entry+'</instrument>'
                instrument_dict = xmltodict.parse(
                    xml_str, attr_prefix='', cdata_key='text')

                # add timestamp to dict, constant for now
                instrument_dict['ice_file_date'] = timestamp

                # conform fields with bq schema
                derepeat_bad_fields(bad_address_list, instrument_dict)
                record_key_flatten(address_key_list, instrument_dict)
                repeated_enforcement(all_repeated_nodes, instrument_dict)

                ndjson = json.dumps(instrument_dict)+'\n'

                f.write(ndjson.encode())
        
        f.seek(0)
        storage_client = storage.Client()
        bucket = storage_client.bucket('ice_ndjsons')
        blob = bucket.blob(json_file_name)
        blob.upload_from_filename('/tmp/%s' % (json_file_name))
        f.close
    uri = 'gs://ice_ndjsons/' + json_file_name
    # remove temp json file
    os.remove('/tmp/%s' % (json_file_name))
    return(uri)

def get_to_process_files():
    
    PROJECT = 'eng-reactor-287421'
    dataset = 'reference_data'
    process_table = 'ice_files_loading_processing'

    bqclient = bigquery.Client(project=PROJECT,)

    # get mapping schema from bigquery
    get_files_query = """SELECT zip_file_name FROM {}.{} GROUP BY zip_file_name HAVING MAX(status) = 0""".format(dataset, process_table)

    get_files_result = bqclient.query(get_files_query).result()
    get_files_result = list(get_files_result)
    get_files_result = [entry[0] for entry in get_files_result]
    
    return get_files_result

def update_ice_files_loading_processing_table(zip_file_name, code):
    query = """
        INSERT INTO `eng-reactor-287421.reference_data.ice_files_loading_processing` 
        VALUES('""" + zip_file_name + """',{}, CURRENT_DATE("US/Eastern"))
    """.format(code)
    query_job = bqclient.query(query)
    query_job.result()
    

def main():

    print("starting main function")
    
    files_to_process = get_to_process_files()
    
    for file_name in files_to_process:
        try:
            print("starting to process "+file_name)
            xmls_generator = xmls_list_generator(file_name)
            uri = get_nested_ndjson_as_uri(
                file_name, xmls_generator, get_timestamp_from_file_name(file_name))
            load_nested_table_uri_json(
                'eng-reactor-287421.reference_data.ice_nested', uri)
            update_ice_files_loading_processing_table(file_name,2)
            print("uploaded "+file_name)
           
        except Exception as e:
            print(e)
            
            
            try:
                update_ice_files_loading_processing_table(file_name,1)
                print("error processing "+file_name)
            except Exception as e:
                print(e)
                print("total error, breaking")
                break
                
    print("completed all new unprocessed files")

            
            
# run main program            
main()
            

      