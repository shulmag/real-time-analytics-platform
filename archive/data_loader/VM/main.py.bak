from google.cloud import bigquery, storage
import os
import csv
import xmltodict
import json
import gzip
import re
from datetime import datetime

PROJECT = 'eng-reactor-287421'
dataset = 'reference_data'
data_table = 'ice_nested'
map_table = 'ice_bq_map'

bqclient = bigquery.Client(project=PROJECT,)

# get mapping schema from bigquery
map_query = """
SELECT * FROM {}.{}

           """.format(dataset, map_table)

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

# address_list is '/' seperated string of nodes address
# xml_dict is ice instrument in python dict
# function checks all repeated nodes in address, if in the instrument, checks if instrument is wrapped in a list
# if not, entry is wrapped in a list


def repeated_enforcement(address_list, xml_dict):
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


# address_key_list contains '/' seperated string of nodes address and the key value to extract
# key value in child node of node in address_key_list is extracted and put as value for node

def record_key_flatten(address_key_list, xml_dict):
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

# this is a one off on a poorly handled field in ICE data
# usually not repeated, when repeated, just take the first entry in list


def derepeat_bad_fields(bad_address_list, xml_dict):
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
    date = re.search(
        '([0-9]{4}[0-9]{2}[0-9]{2}T[0-9]{4}-[0-9]{2})', file_name)[0]
    date = date.replace('-', '')
    return(datetime.strftime(datetime.strptime(date, '%Y%m%dT%H%M%S'), '%Y-%m-%dT%H:%M:%SZ'))


def get_xmls_list(gz_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket('ref_data_1')
    # e.g. 'gsm_update_muni_APFICC_GSMF10I.35.1_1.20201221T0800-05.xml.gz')
    blob = bucket.get_blob(gz_file_name)
    file = blob.download_to_filename('/tmp/temp_gz_file.xml.gz')

    f = gzip.open('/tmp/temp_gz_file.xml.gz', 'rt')
    file_str = f.read()
    clean_file = file_str.split('<payload>')[1]
    return (clean_file.split('</instrument>'))


def load_nested_table_uri_json(table_id, uri):

    bq_client = bigquery.Client()
    ice_nested_table = bqclient.get_table(table_id)

    job_config = bigquery.LoadJobConfig(
        schema=ice_nested_table.schema,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    )

    # Make an API request.
    bq_client.load_table_from_uri(uri, table_id, job_config=job_config)


def get_nested_ndjson_as_uri(file_name, xmls, timestamp):
    ndjson = ''
    total = len(xmls)-1

    for n in range(total):
        xml_str = xmls[n]+'</instrument>'
        instrument_dict = xmltodict.parse(
            xml_str, attr_prefix='', cdata_key='text')

        # add timestamp to dict, constant for now
        instrument_dict['ice_file_date'] = timestamp

        # conform fields with bq schema
        derepeat_bad_fields(bad_address_list, instrument_dict)
        record_key_flatten(address_key_list, instrument_dict)
        repeated_enforcement(all_repeated_nodes, instrument_dict)

        ndjson += json.dumps(instrument_dict)+'\n'

    json_file_name = '%snest.json' % (file_name)
    with open('/tmp/%s' % (json_file_name), 'wb+') as f:
        f.write(ndjson.encode())
        f.seek(0)
        storage_client = storage.Client()
        bucket = storage_client.bucket('ice_ndjsons')
        blob = bucket.blob(json_file_name)
        blob.upload_from_filename('/tmp/%s' % (json_file_name))
        f.close
    uri = 'gs://ice_ndjsons/' + json_file_name
    return(uri)

def update_ice_files_loading_processing_table(zip_file_name):
    query = """
        INSERT INTO `eng-reactor-287421.reference_data.ice_files_loading_processing` 
        VALUES('""" + zip_file_name + """',1, CURRENT_DATE("US/Eastern"))
    """
    query_job = bq_client.query(query)
    query_job.result()

def get_new_ice_files_from_loading_table():
    query = """
        SELECT
        *
        FROM (
        SELECT
            COUNT(zip_file_name) AS zip_count,
            zip_file_name
        FROM
            `eng-reactor-287421.reference_data.ice_files_loading_processing`
        GROUP BY
            zip_file_name) files
        WHERE
        files.zip_count = 1
    """
    query_job = bqclient.query(query)
    query_job.result()
    destination = query_job.destination
    table = bqclient.get_table(destination)
    # Download rows:
    rows = bqclient.list_rows(table)
    return rows #???
#    for row in rows:
#        return(row["zip file name??"])
#        break

def main():
    new_files = get_new_ice_files_from_loading_table()
    for file_name in new_files:
        file_name = event['name']
        xmls = get_xmls_list(file_name)
        uri = get_nested_ndjson_as_uri(
            file_name, xmls, get_timestamp_from_file_name(file_name))
        load_nested_table_uri_json(
            'eng-reactor-287421.reference_data.ice_nested', uri)
        update_ice_files_loading_processing_table(file_name)

get_new_ice_files_from_loading_table()