#import os
#gcloud functions deploy load_ice_data_from_gz_file \
#--runtime python37 \
#--trigger-resource ref_data_1 \
#--trigger-event google.storage.object.finalize
import xmltodict
import json
import time
import gzip
import shutil
import collections
import re

from datetime import datetime
from google.cloud import bigquery, storage
from datetime import datetime,timedelta
from google.api_core.exceptions import BadRequest

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/Gil/git/ficc/data_loader/Cusip Global Service Importer-2fdcdfc4edba.json"

bq_schema = ["ice_file_date","instrument_id",	"cusip",	"isin",	"security_typ",	"apex_asset_type",	"instrument_type",	"primary_name",	"delivery_date",	"issue_price",	"settlement_date",	"issue_date",	"outstanding_indicator",	"primary_currency_code",	"child_issue_ind",	"federal_tax_status",	"id",	"entity_level",	"type",	"type_id",	"text",	"incorporated_state_code",	"primary_name_abbreviated",	"organization_status",	"organization_type",	"org_country_code",	"country_code",	"maturity_amount",	"denom_increment_amount",	"min_denom_amount",	"accrual_date",	"bond_form",	"bond_insurance",	"coupon_type",	"current_coupon_rate",	"daycount_basis_type",	"debt_type",	"default_indicator",	"depository_type",	"first_coupon_date",	"interest_payment_frequency",	"issue_amount",	"last_period_accrues_from_date",	"maturity_date",	"next_coupon_payment_date",	"orig_principal_amount",	"original_yield",	"outstanding_amount",	"sale_type",	"settlement_type",	"principal_factor",	"principal_factort1",	"principal_factort2",	"principal_factort3",	"principal_factort4",	"previous_coupon_payment_date",	"bank_qualified",	"capital_type",	"dtcc_status",	"first_execution_date",	"formal_award_date",	"issue_key",	"issue_text",	"maturity_description_code",	"muni_security_type",	"other_enhancement_type",	"sale_date",	"series_name",	"state_tax_status",	"use_of_proceeds",	"asset_claim_code",	"purpose_class",	"purpose_sub_class",	"num",	"call_notice",	"call_timing",	"call_timing_in_part",	"extraordinary_make_whole_call",	"extraordinary_redemption",	"make_whole_call",	"mandatory_redemption_code",	"next_call_date",	"next_call_price",	"optional_redemption_code",	"par_call_date",	"par_call_price",	"date",	"amount_outstanding",	"amount_outstanding_decimal"]
dict_data_type = {"ice_file_date":"timestamp","instrument_id":"integer",	"cusip":"string",	"isin":"string",	"security_typ":"string",	"apex_asset_type":"integer",	"instrument_type":"integer",	"primary_name":"string",	"delivery_date":"date",	"issue_price":"numeric",	"settlement_date":"date",	"issue_date":"date",	"outstanding_indicator":"boolean",	"primary_currency_code":"string",	"child_issue_ind":"boolean",	"federal_tax_status":"integer",	"id":"integer",	"entity_level":"string",	"type":"string",	"type_id":"integer",	"text":"string",	"incorporated_state_code":"string",	"primary_name_abbreviated":"string",	"organization_status":"integer",	"organization_type":"integer",	"org_country_code":"string",	"country_code":"string",	"maturity_amount":"numeric",	"denom_increment_amount":"numeric",	"min_denom_amount":"numeric",	"accrual_date":"date",	"bond_form":"integer",	"bond_insurance":"string",	"coupon_type":"integer",	"current_coupon_rate":"numeric",	"daycount_basis_type":"integer",	"debt_type":"integer",	"default_indicator":"boolean",	"depository_type":"integer",	"first_coupon_date":"date",	"interest_payment_frequency":"integer",	"issue_amount":"numeric",	"last_period_accrues_from_date":"date",	"maturity_date":"date",	"next_coupon_payment_date":"date",	"orig_principal_amount":"numeric",	"original_yield":"numeric",	"outstanding_amount":"numeric",	"sale_type":"integer",	"settlement_type":"integer",	"principal_factor":"numeric",	"principal_factort1":"numeric",	"principal_factort2":"numeric",	"principal_factort3":"numeric",	"principal_factort4":"numeric",	"previous_coupon_payment_date":"date",	"bank_qualified":"boolean",	"capital_type":"integer",	"dtcc_status":"integer",	"first_execution_date":"timestamp",	"formal_award_date":"timestamp",	"issue_key":"integer",	"issue_text":"string",	"maturity_description_code":"integer",	"muni_security_type":"integer",	"other_enhancement_type":"integer",	"sale_date":"timestamp",	"series_name":"string",	"state_tax_status":"integer",	"use_of_proceeds":"integer",	"asset_claim_code":"integer",	"purpose_class":"integer",	"purpose_sub_class":"integer",	"num":"integer",	"call_notice":"integer",	"call_timing":"integer",	"call_timing_in_part":"integer",	"extraordinary_make_whole_call":"boolean",	"extraordinary_redemption":"boolean",	"make_whole_call":"boolean",	"mandatory_redemption_code":"integer",	"next_call_date":"date",	"next_call_price":"numeric",	"optional_redemption_code":"integer",	"par_call_date":"date",	"par_call_price":"numeric",	"date":"date",	"amount_outstanding":"integer",	"amount_outstanding_decimal":"numeric"}
bq_client = bigquery.Client()
    
def is_float(string):
    try: 
        float(string)
        return True

    except ValueError:
        return False

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        elif type(v) == list:
            for elem in v:
                try:
                    keys = list(elem.keys())
                    #not is_date(elem[keys[0]], fuzzy=True) and 
                    if not elem[keys[0]].isdecimal() and not is_float(elem[keys[0]]) and elem[keys[0]].find('/') == -1:
                        items.append((elem[keys[0]].lower(),elem[keys[len(keys)-1]]))
                    elif elem['@security_typ']:
                        items.append(('security_typ',elem['@security_typ']))    
                except Exception as e:
                    pass
        else:
            if new_key == '@id' and parent_key == 'instrument':
                items.append(('instrument_id',v))    
            elif new_key == '@id:':
                pass
            else:
                items.append((new_key.lower(), v))
    return dict(items)  

def data_type_casting(i_dict):
    dict_ice_instrument = i_dict
    for k,v in dict_ice_instrument.items():
        try:
            if(dict_data_type[k]=='date'):
                if v != None:
                    if len(v) == 10:
                        dict_ice_instrument[k] = v #datetime.strptime(v, '%Y-%m-%d').strftime('%Y-%m-%d')
                    elif len(v) == 20:
                        dict_ice_instrument[k] = v
                    else:
                        dict_ice_instrument[k] = None
            if(dict_data_type[k] == 'numeric' or dict_data_type[k] == 'float'):
                if v != '' and v != None:
                    dict_ice_instrument[k] = float(v)
                else: 
                    dict_ice_instrument[k] = None
            if(dict_data_type[k]=='string'):
                dict_ice_instrument[k] = v   
            if(dict_data_type[k]=='integer' and v != None):
                dict_ice_instrument[k] = int(v) 
            if(dict_data_type[k]=='boolean'):
                dict_ice_instrument[k] = (v == 'true')
            if(dict_data_type[k]=='timestamp'):
                dict_ice_instrument[k] = time.mktime(datetime.strptime(v, '%Y-%m-%dT%H:%M:%SZ').timetuple())
            elif(dict_data_type[k]=='time'):
                if v != None:
                    dict_ice_instrument[k] = datetime.strftime(datetime.strptime(v,'%H%M%S'),'%H:%M:%S')
        except Exception:
            dict_ice_instrument[k] = None

    return(dict_ice_instrument)

def get_timestamp_from_file_name(file_name):
    date = re.search('([0-9]{4}[0-9]{2}[0-9]{2}T[0-9]{4}-[0-9]{2})', file_name)[0]
    date = date.replace('-','')
    return(datetime.strftime(datetime.strptime(date, '%Y%m%dT%H%M%S'),'%Y-%m-%dT%H:%M:%SZ'))

def get_xmls_list(gz_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket('ref_data_1')
    blob = bucket.get_blob(gz_file_name) #e.g. 'gsm_update_muni_APFICC_GSMF10I.35.1_1.20201221T0800-05.xml.gz')
    file = blob.download_to_filename('/tmp/temp_gz_file.xml.gz')

    f = gzip.open('/tmp/temp_gz_file.xml.gz', 'rt')
    file_str = f.read()
    clean_file = file_str.split('<payload>')[1]
    return (clean_file.split('</instrument>'))

def load_table_uri_json(table_id,uri):
    schema_fields = [bigquery.SchemaField("ice_file_date","timestamp"),	bigquery.SchemaField("instrument_id","integer"),	bigquery.SchemaField("cusip","string"),	bigquery.SchemaField("isin","string"),	bigquery.SchemaField("security_typ","string"),	bigquery.SchemaField("apex_asset_type","integer"),	bigquery.SchemaField("instrument_type","integer"),	bigquery.SchemaField("primary_name","string"),	bigquery.SchemaField("delivery_date","date"),	bigquery.SchemaField("issue_price","numeric"),	bigquery.SchemaField("settlement_date","date"),	bigquery.SchemaField("issue_date","date"),	bigquery.SchemaField("outstanding_indicator","boolean"),	bigquery.SchemaField("primary_currency_code","string"),	bigquery.SchemaField("child_issue_ind","boolean"),	bigquery.SchemaField("federal_tax_status","integer"),	bigquery.SchemaField("id","integer"),	bigquery.SchemaField("entity_level","string"),	bigquery.SchemaField("type","string"),	bigquery.SchemaField("type_id","integer"),	bigquery.SchemaField("text","string"),	bigquery.SchemaField("incorporated_state_code","string"),	bigquery.SchemaField("primary_name_abbreviated","string"),	bigquery.SchemaField("organization_status","integer"),	bigquery.SchemaField("organization_type","integer"),	bigquery.SchemaField("org_country_code","string"),	bigquery.SchemaField("country_code","string"),	bigquery.SchemaField("maturity_amount","numeric"),	bigquery.SchemaField("denom_increment_amount","numeric"),	bigquery.SchemaField("min_denom_amount","numeric"),	bigquery.SchemaField("accrual_date","date"),	bigquery.SchemaField("bond_form","integer"),	bigquery.SchemaField("bond_insurance","string"),	bigquery.SchemaField("coupon_type","integer"),	bigquery.SchemaField("current_coupon_rate","numeric"),	bigquery.SchemaField("daycount_basis_type","integer"),	bigquery.SchemaField("debt_type","integer"),	bigquery.SchemaField("default_indicator","boolean"),	bigquery.SchemaField("depository_type","integer"),	bigquery.SchemaField("first_coupon_date","date"),	bigquery.SchemaField("interest_payment_frequency","integer"),	bigquery.SchemaField("issue_amount","numeric"),	bigquery.SchemaField("last_period_accrues_from_date","date"),	bigquery.SchemaField("maturity_date","date"),	bigquery.SchemaField("next_coupon_payment_date","date"),	bigquery.SchemaField("orig_principal_amount","numeric"),	bigquery.SchemaField("original_yield","numeric"),	bigquery.SchemaField("outstanding_amount","numeric"),	bigquery.SchemaField("sale_type","integer"),	bigquery.SchemaField("settlement_type","integer"),	bigquery.SchemaField("principal_factor","numeric"),	bigquery.SchemaField("principal_factort1","numeric"),	bigquery.SchemaField("principal_factort2","numeric"),	bigquery.SchemaField("principal_factort3","numeric"),	bigquery.SchemaField("principal_factort4","numeric"),	bigquery.SchemaField("previous_coupon_payment_date","date"),	bigquery.SchemaField("bank_qualified","boolean"),	bigquery.SchemaField("capital_type","integer"),	bigquery.SchemaField("dtcc_status","integer"),	bigquery.SchemaField("first_execution_date","timestamp"),	bigquery.SchemaField("formal_award_date","timestamp"),	bigquery.SchemaField("issue_key","integer"),	bigquery.SchemaField("issue_text","string"),	bigquery.SchemaField("maturity_description_code","integer"),	bigquery.SchemaField("muni_security_type","integer"),	bigquery.SchemaField("other_enhancement_type","integer"),	bigquery.SchemaField("sale_date","timestamp"),	bigquery.SchemaField("series_name","string"),	bigquery.SchemaField("state_tax_status","integer"),	bigquery.SchemaField("use_of_proceeds","integer"),	bigquery.SchemaField("asset_claim_code","integer"),	bigquery.SchemaField("purpose_class","integer"),	bigquery.SchemaField("purpose_sub_class","integer"),	bigquery.SchemaField("num","integer"),	bigquery.SchemaField("call_notice","integer"),	bigquery.SchemaField("call_timing","integer"),	bigquery.SchemaField("call_timing_in_part","integer"),	bigquery.SchemaField("extraordinary_make_whole_call","boolean"),	bigquery.SchemaField("extraordinary_redemption","boolean"),	bigquery.SchemaField("make_whole_call","boolean"),	bigquery.SchemaField("mandatory_redemption_code","integer"),	bigquery.SchemaField("next_call_date","date"),	bigquery.SchemaField("next_call_price","numeric"),	bigquery.SchemaField("optional_redemption_code","integer"),	bigquery.SchemaField("par_call_date","date"),	bigquery.SchemaField("par_call_price","numeric"),	bigquery.SchemaField("date","date"),	bigquery.SchemaField("amount_outstanding","integer"),	bigquery.SchemaField("amount_outstanding_decimal","numeric")]
    
    client = bigquery.Client()

    job_config = bigquery.LoadJobConfig(
        schema=schema_fields,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    )

    bq_client.load_table_from_uri(uri, table_id, job_config=job_config)# Make an API request.  
    
def get_ndjson_as_uri(file_name,xmls,timestamp):
    ndjson = ''
    #from tqdm.notebook import tqdm
    total = len(xmls)-1
    #pbar = tqdm(total=total)

    for n in range(len(xmls)-1):
        xml_str = xmls[n]+'</instrument>' 
        try:
            instrument_dict = xmltodict.parse(xml_str)
        except Exception as ex:
            pass
        flat = flatten(instrument_dict)
        instrument = {bq_schema[i]:flat[bq_schema[i]] if bq_schema[i] in flat else None for i in range(len(bq_schema))}
        instrument['ice_file_date'] = timestamp
        ndjson+= json.dumps(data_type_casting(instrument))+'\n'
        #pbar.update(n/total)
    json_file_name = '%s.json' % (file_name) 
    with open('/tmp/%s' % (json_file_name), 'wb+') as f:
        f.write(ndjson.encode())
        f.seek(0)
        storage_client = storage.Client()
        bucket = storage_client.bucket('ice_ndjsons')
        blob = bucket.blob(json_file_name)
        blob.upload_from_filename('/tmp/%s' % (json_file_name))
        f.close
    uri = 'gs://ice_ndjsons/' + json_file_name
    #pbar.close()
    return(uri)

def main(event, context):
    file_name = event['name']
    xmls = get_xmls_list(file_name)
    uri = get_ndjson_as_uri(file_name,xmls,get_timestamp_from_file_name(file_name))  
    load_table_uri_json('eng-reactor-287421.reference_data.ice',uri)