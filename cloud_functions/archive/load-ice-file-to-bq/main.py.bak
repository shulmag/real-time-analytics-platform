from google.cloud import bigquery, storage
import xmltodict
import json
import gzip
import re
from datetime import datetime
from google.api_core.exceptions import BadRequest

# # need credentials from testing on VM. Not to be used during production
# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/shayaan/ahmad_creds.json'

# pull schema information as stored in ice_bq_map for functions that conform xmls to bigquery schema
# generate list of xmls and pass to json creater, then load to table
PROJECT = 'eng-reactor-287421'
DATASET = 'reference_data'
LOADING_PROCESSING_TABLE = 'ice_files_loading_processing'
ICE_NESTED_TABLE = 'ice_nested'
ICE_BQ_MAP_TABLE = 'ice_bq_map'
BQ_CLIENT = bigquery.Client(project=PROJECT)
STORAGE_CLIENT = storage.Client()

# get mapping schema from bigquery
bq_map_result = list(BQ_CLIENT.query(f'SELECT * FROM {DATASET}.{ICE_BQ_MAP_TABLE}').result())
ice_schema, map_version, leaf_nodes, repeated_record_nodes, address_key_list, bad_address_list = bq_map_result[0]
leaf_nodes = [[entry['path'], entry['type'], entry['mode']] for entry in leaf_nodes]

address_key_list = [[entry['path'], entry['key']] for entry in address_key_list]
repeated_leafs = [entry[0] for entry in leaf_nodes if entry[2] == 'REPEATED']
all_repeated_nodes = repeated_leafs + repeated_record_nodes


def repeated_enforcement(address_list, xml_dict):
    '''Takes a list of address nodes separated by '/', `address_list`, and an XML which is 
    represented as a Python dictionary, `xml_dict`, and goes through the `xml_dict` level by 
    level by checking if the address item is in the dictionary, and wrapping the instrument 
    in a list, if it is not already in a list itself. 
    NOTE: this function mutates the input `xml_dict`.'''
    split_address_list = [address.split('/')[1:] for address in address_list]    # TODO: why do we slice with at index 1 after splitting at `/`?
    for address in split_address_list:
        current_xml_level = xml_dict    # start at the very top level of the nested XML dictionary
        all_but_last_address_item = address[:-1]
        for depth_idx, address_item in enumerate(all_but_last_address_item):    # TODO: why do skip investigating the last item?
            if address_item not in current_xml_level:
                break
            current_xml_level = current_xml_level[address_item]    # go to the next level down in the `xml_dict`
        print(f'For address: {all_but_last_address_item}, arrived at level: {all_but_last_address_item[:depth_idx]}')

        last_address_item = address[-1]
        print(f'Last address item: {last_address_item}')
        if last_address_item not in current_xml_level:
            print('Last address item is not present in the current level')
            continue
        else:
            if not isinstance(current_xml_level[last_address_item], list):
                current_xml_level[last_address_item] = [current_xml_level[last_address_item]]
    return xml_dict


def record_key_flatten(address_key_list, xml_dict):
    '''Takes a list of pairs of address nodes separated by '/' and keys, `address_key_list`, and an XML 
    which is represented as a Python dictionary, `xml_dict`, and goes through the `xml_dict` level by 
    level by checking if the address item is in the dictionary, and setting the key to be the value of 
    the XML dictionary at that level. The key value is the child node of the node in `address_key_list`.'''
    split_address_key_list = [[address_key[0].split('/')[1:], address_key[1]] for address_key in address_key_list]    # TODO: why do we slice with at index 1 after splitting at `/`?
    for address, key in split_address_key_list:
        current_xml_level = xml_dict    # start at the very top level of the nested XML dictionary
        all_but_last_address_item = address[:-1]
        for depth_idx, address_item in enumerate(all_but_last_address_item):    # TODO: why do skip investigating the last item?
            if address_item not in current_xml_level:
                break
            current_xml_level = current_xml_level[address_item]    # go to the next level down in the `xml_dict`
        print(f'For address: {all_but_last_address_item}, arrived at level: {all_but_last_address_item[:depth_idx]}')

        last_address_item = address[-1]
        print(f'Last address item: {last_address_item}')
        if last_address_item not in current_xml_level:
            print('Last address item is not present in the current level')
            continue
        else:
            if isinstance(current_xml_level[last_address_item], dict):
                current_xml_level[last_address_item] = current_xml_level[last_address_item][key]
    return xml_dict


def derepeat_bad_fields(bad_address_list, xml_dict):
    '''Takes care of a poorly handled field in ICE data. When repeated, takes just the first entry in the list.'''
    split_bad_address_list = [entry.split('/')[1:] for entry in bad_address_list]
    for address in split_bad_address_list:
        current_xml_level = xml_dict    # start at the very top level of the nested XML dictionary
        all_but_last_address_item = address[:-1]
        for depth_idx, address_item in enumerate(all_but_last_address_item):    # TODO: why do skip investigating the last item?
            if address_item not in current_xml_level:
                break
            current_xml_level = current_xml_level[address_item]    # go to the next level down in the `xml_dict`
        print(f'For address: {all_but_last_address_item}, arrived at level: {all_but_last_address_item[:depth_idx]}')
        
        last_address_item = address[-1]
        print(f'Last address item: {last_address_item}')
        if last_address_item not in current_xml_level:
            print('Last address item is not present in the current level')
            continue
        else:
            if isinstance(current_xml_level[last_address_item], list):
                current_xml_level[last_address_item] = current_xml_level[last_address_item][0]    # take the first entry if there are repeats


def get_timestamp_from_file_name(file_name):
    # regex to pull timestamp from file name
    date = re.search('([0-9]{4}[0-9]{2}[0-9]{2}T[0-9]{4}-[0-9]{2})', file_name)[0]
    date = date.replace('-', '')
    return datetime.strftime(datetime.strptime(date, '%Y%m%dT%H%M%S'), '%Y-%m-%dT%H:%M:%SZ')


def xmls_list_generator(gz_file_name):
    # with zip file name download to temp file, open and read through about 10mb of zip file size
    # parse to list of xml in string format and pass each iteration through generator
    bucket = STORAGE_CLIENT.bucket('ref_data_1')
    temp_file_path = '/tmp/temp_gz_file.xml.gz'
    blob = bucket.get_blob(gz_file_name)    # example `gz_file_name`: 'gsm_update_muni_APFICC_GSMF10I.35.1_1.20201221T0800-05.xml.gz'
    blob.download_to_filename(temp_file_path)
    
    start = True    # TODO: what does this variable represent?
    last_partial_entry = ''
    keep_reading = True
    with gzip.open(temp_file_path, 'rt') as f:
        while keep_reading:
            # read_length equal to a zip file size of about 15mb
            read_length = int(1e8)
            file_str = f.read(read_length)
            file_str = last_partial_entry + file_str
            if len(file_str) < read_length:
                file_str = file_str.split('</payload>')[0]
            if start:
                file_str = file_str.split('<payload>')[1]
                start = False
            xml_list = file_str.split('</instrument>')
            # save last entry of the list for next loop, will be empty if string cleanly ends on an xml instrument
            last_partial_entry = xml_list.pop()

            if not file_str:
                keep_reading = False

            yield xml_list


def load_nested_table_uri_json(table_id, uri, ignore_values_bool, max_bad_records_num):
    # load from created json file into main table
    ice_nested_table = BQ_CLIENT.get_table(table_id)
    print('ICE Nested Table:', ice_nested_table)

    job_config = bigquery.LoadJobConfig(schema=ice_nested_table.schema,
                                        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                                        ignore_unknown_values=ignore_values_bool,
                                        max_bad_records=max_bad_records_num)

    try:
        load_job = BQ_CLIENT.load_table_from_uri(uri, table_id, job_config=job_config)
        load_job.result()
    except BadRequest as e:
        for error in load_job.errors:
            print(f'{type(error)}: {error["message"]}')
        raise e


def get_nested_ndjson_as_uri(file_name, xmls_generator, timestamp):
    '''Use `file_name` for saving the file, and use `timestamp` for recording time of upload to table.'''
    json_file_name = f'{file_name}nest.json'
    # save memory by writing one json string to file at a time rather than storing to a variable
    with open(f'/tmp/{json_file_name}', 'wb+') as f:    # TODO: why do we need to perform this series of `del` statements?
        for xmls in xmls_generator:    # loop through capped size xml lists
            for xml_entry in xmls:
                xml_str = xml_entry + '</instrument>'
                instrument_dict = xmltodict.parse(xml_str, attr_prefix='', cdata_key='text')
                instrument_dict['ice_file_date'] = timestamp    # add timestamp to dict, constant for now

                #### Modifications added by Developer March 12 2022 #####
                master_information = instrument_dict['instrument']['master_information']
                if 'instrument_xref' in master_information.keys():    # TODO: should we have another condition here to check if `'xref' in master_information['instrument_xref'].keys()` similar to the second condition checking whether `'market' in master_information['market_master'].keys()`?
                    xref = master_information['instrument_xref']['xref']
                    if isinstance(xref, list):
                        for xref_value in xref:
                            if 'long_name' in xref_value:
                                try:
                                    del xref_value['long_name']
                                except Exception as e:
                                    raise e
                            if 'valid' in xref_value:
                                try:
                                    del xref_value['valid']
                                except Exception as e:
                                    raise e

                if 'market_master' in master_information.keys() and 'market' in master_information['market_master'].keys():
                    market = master_information['market_master']['market']
                    try:
                        if 'linked_markets' in market.keys() and 'linked_market' in market['linked_markets'].keys() and isinstance(market['linked_markets']['linked_market'], list):
                            market['linked_markets']['linked_market'] = market['linked_markets']['linked_market'][0]
                    except Exception as e:
                        if isinstance(market, list):
                            master_information['market_master']['market'] = market[0]
                        else:
                            raise e
                
                try:
                    muni_details = instrument_dict['instrument']['debt']['muni_details']
                    if 'conduit_obligor_name' in muni_details:
                        muni_details['conduit_obligor_name'] = muni_details['conduit_obligor_name']['text']
                except Exception as e:
                    pass    # TODO: what is an appropriate print statement when this exception is ignored?

                try:
                    del master_information['market_master']['market']['related_markets']['related_market']['xref']['text']
                except Exception as e:
                    pass    # TODO: what is an appropriate print statement when this exception is ignored?

                for field in ['impact_bond_ind', 'impact_bond_type', 'impact_bond_third_party', 'impact_bond_framework', 'impact_bond_subtype']:
                    try:
                        del instrument_dict['instrument']['debt']['fixed_income'][field]
                    except Exception as e:
                        pass    # TODO: what is an appropriate print statement when this exception is ignored?

                try:
                    put_details = instrument_dict['instrument']['debt']['put_details']
                    if 'put_schedule' in put_details:
                        del put_details['put_schedule']
                except Exception as e:
                    pass    # TODO: what is an appropriate print statement when this exception is ignored?

                # conform fields with bq schema
                derepeat_bad_fields(bad_address_list, instrument_dict)
                instrument_dict = record_key_flatten(address_key_list, instrument_dict)
                instrument_dict = repeated_enforcement(all_repeated_nodes, instrument_dict)
                ndjson = json.dumps(instrument_dict) + '\n'
                f.write(ndjson.encode())

        f.seek(0)    # TODO: why is this necessary?
        bucket = STORAGE_CLIENT.bucket('ice_ndjsons')
        blob = bucket.blob(json_file_name)
        blob.upload_from_filename(f'/tmp/{json_file_name}')
    uri = 'gs://ice_ndjsons/' + json_file_name
    return uri


def get_files():
    get_files_query = f'''SELECT zip_file_name FROM {DATASET}.{LOADING_PROCESSING_TABLE} GROUP BY zip_file_name HAVING MAX(status) = 0'''
    files_list = list(BQ_CLIENT.query(get_files_query).result())
    files_list = [file[0] for file in files_list]
    print(f'Files list: {files_list} from making the BigQuery call: {get_files_query}')
    return files_list


def update_ice_files_loading_processing_table(zip_file_name, code):
    query = f"""INSERT INTO `{PROJECT}.{DATASET}.{LOADING_PROCESSING_TABLE}` VALUES('"""+ zip_file_name + f"""',{code}, CURRENT_DATE("US/Eastern"))"""
    query_job = BQ_CLIENT.query(query)
    query_job.result()
    print(f'Uploaded {zip_file_name} with code: {code} into {PROJECT}.{DATASET}.{LOADING_PROCESSING_TABLE}')


def main(args):
    files_to_process = get_files()
    for file_name in files_to_process:
        xmls_generator = ''
        uri = ''
        try:
            print(f'Processing {file_name}')
            xmls_generator = xmls_list_generator(file_name)
            uri = get_nested_ndjson_as_uri(file_name, xmls_generator, get_timestamp_from_file_name(file_name))
            load_nested_table_uri_json(f'{PROJECT}.{DATASET}.{ICE_NESTED_TABLE}', uri, False, 0)
            update_ice_files_loading_processing_table(file_name, 2)
        except Exception as e:
            print(f"Failed to upload file {file_name} on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            try:
                print('Trying to load again...')
                load_nested_table_uri_json(f'{PROJECT}.{DATASET}.{ICE_NESTED_TABLE}', uri, True, 5)
                update_ice_files_loading_processing_table(file_name, 2)
            except Exception as e:
                update_ice_files_loading_processing_table(file_name, 1)
                print(f'Failed to process file: {file_name} due to {type(e)}: {e}')
                break
    return 'Success'
