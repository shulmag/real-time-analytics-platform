'''
Description: Functions for parsing the XML from S&P.
'''
import io
from collections import defaultdict
from datetime import datetime
import pytz
import pandas as pd

from lxml import etree

from google.cloud import bigquery, storage

from auxiliary_functions import function_timer


BQ_CLIENT = bigquery.Client()
STORAGE_CLIENT = storage.Client()

TAGS_WITH_FLAT_ELEMENTS = {tag: [] for tag in ['clearingSchedule', 
                                               'creditEnhancement', 
                                               'dealInfo', 
                                               'derivedData', 
                                               'exdividendRecordDetails', 
                                               'holidaySchedule', 
                                               'instrument', 
                                               'instrumentRelation', 
                                               'organization',
                                               'principalDetails',  
                                               'regulationDetails', 
                                               'tradingAndSellingRestrictionDetails']}    # create a dictionary that maps to the sub-fields that will be populated later
TAGS_TO_PUT_AS_PREFIX = ['creditEnhancement', 'instrumentRelation', 'principalDetails']    # subset of keys in `TAGS_WITH_FLAT_ELEMENTS` that have elements that should be distinguished with a prefix for accuracy (e.g., 'effectiveDate' is present in both 'instrumentRelation' and 'principalDetails') and readability (e.g., 'endDate' is a subfield in 'creditEnhancement' but is unclear if it just written as 'endDate')

SP_REF_DATA_GOOGLE_CLOUD_BUCKET_NAME = 'sp_ref_data'
SP_REFERENCE_DATA_NESTED_TABLE_NAME = 'eng-reactor-287421.sp_reference_data.sp_nested'

EASTERN = pytz.timezone('US/Eastern')

TYPOS_IN_SP_SCHEMA = {'paymentCategorySubtype': 'paymentCategorySubType', 
                      'origIntRate': 'originalInterestRate'}


def upload_to_bigquery(data: list, schema, table_name: str):
    job_config = bigquery.LoadJobConfig(schema=schema)
    load_job = BQ_CLIENT.load_table_from_json(data, table_name, job_config=job_config)
    try:
        load_job.result()    # waits for the job to complete
    except Exception as e:
        print(load_job.errors)
        raise e


def convert_bq_schema_to_use_bigquery_constructor(bq_schema) -> list:
    def create_schema_field(field):
        if 'fields' in field:    # handles the fields with REPEATED values
            return bigquery.SchemaField(name=field['name'], 
                                        field_type=field['type'], 
                                        mode=field['mode'], 
                                        description=field.get('description'),    # using `.get(...)` so that if `description` does not exist, then it returns `None`
                                        fields=[create_schema_field(sub_field) for sub_field in field['fields']])
        else:
            return bigquery.SchemaField(name=field['name'], 
                                        field_type=field['type'], 
                                        mode=field['mode'], 
                                        description=field.get('description'))    # using `.get(...)` so that if `description` does not exist, then it returns `None`

    return [create_schema_field(field) for field in bq_schema]


def get_duplicates(lst: list) -> list:
    seen = set()
    duplicates = []
    for item in lst:
        if item in seen: duplicates.append(item)
        seen.add(item)
    return duplicates


def lowercase_first_letter(s: str) -> str:
    return s[0].lower() + s[1:]


def correct_name_if_typo(name: str) -> str:
    '''Fix typos from the S&P schema, i.e., instances of names in the schema that do not match that of the XML data.'''
    return name if name not in TYPOS_IN_SP_SCHEMA else TYPOS_IN_SP_SCHEMA[name]


def convert_sp_dtype_to_bigquery_dtype(dtype: str) -> str:
    if 'STRING' in dtype: return 'STRING'
    if dtype == 'DECIMAL' or dtype == 'LONG': return 'NUMERIC'
    if dtype == 'DATE': return 'DATE'
    return dtype


def get_sp_schema_dict() -> pd.DataFrame:
    bucket = STORAGE_CLIENT.get_bucket(SP_REF_DATA_GOOGLE_CLOUD_BUCKET_NAME)
    csv_content = bucket.blob('sp_schema_dict.csv').download_as_bytes()
    return pd.read_csv(io.BytesIO(csv_content))


def add_prefix_to_field(tag: str, field_name: str) -> str:
    return tag + '_' + field_name


def create_bq_schema():
    # initialize the BigQuery schema with the 'upload_datetime' field to indicate when this entry was created, the `file_received_from_sp_date` field to indicate when the file was received, and the `gcp_storage_file_path` field which has the file path in the GCP storage of the file
    bq_schema = [{'name': 'upload_datetime', 'type': 'DATETIME', 'mode': 'NULLABLE', 'description': 'Datetime when this entry was created.'}, 
                 {'name': 'file_received_from_sp_date', 'type': 'DATE', 'mode': 'NULLABLE', 'description': 'Date when the file was received from S&P.'}, 
                 {'name': 'gcp_storage_file_path', 'type': 'STRING', 'mode': 'NULLABLE', 'description': 'File path in GCP storage of the file from S&P.'}]

    # group by Table Name to handle nested fields
    df = get_sp_schema_dict()
    grouped = df.groupby('Table Name')

    for table_name, group in grouped:
        fields = []
        for _, row in group.iterrows():
            field_schema = {'name': correct_name_if_typo(lowercase_first_letter(row['Field Name'])),
                            'type': convert_sp_dtype_to_bigquery_dtype(row['Data Type'].upper()),
                            'mode': 'NULLABLE',    # assuming all fields are nullable; adjust as needed
                            'description': row['Descriptions']}
            fields.append(field_schema)
        
        table_schema = {'name': correct_name_if_typo(lowercase_first_letter(table_name)),
                        'type': 'RECORD',
                        'mode': 'REPEATED',    # assuming repeated records; adjust as needed
                        'fields': fields}
        bq_schema.append(table_schema)

    # flatten the elements in `TAGS_WITH_FLAT_ELEMENTS`
    element_indices_to_flatten = [idx for idx, element in enumerate(bq_schema) if element['name'] in TAGS_WITH_FLAT_ELEMENTS]
    element_indices_to_flatten = sorted(element_indices_to_flatten, reverse=True)    # need to remove each index in reversed numerical order to preserve the index positions of the ones to be removed later (e.g., for [A, B, C] if you remove index 1, then you cannot remove index 2 afterwards, since index 2 no longer exists)
    for idx in element_indices_to_flatten:
        element = bq_schema.pop(idx)
        name, fields = element['name'], element['fields']
        TAGS_WITH_FLAT_ELEMENTS[name] = [field['name'] for field in fields]
        if name in TAGS_TO_PUT_AS_PREFIX:
            for field in fields:
                field['name'] = add_prefix_to_field(name, field['name'])
        bq_schema.extend(fields)

    # check that all fields are unique
    all_fields = [schema_dict['name'] for schema_dict in bq_schema]
    if len(all_fields) != len(set(all_fields)):
        duplicate_fields = get_duplicates(all_fields)
        error_string = f'Not all of the fields in the schema are unique; non-unique fields: {get_duplicates(all_fields)}\n'
        for field in duplicate_fields:
            error_string += f'\nThe following tags have {field} as a duplicate field: {[tag for tag, fields in TAGS_WITH_FLAT_ELEMENTS.items() if field in fields]}'
        raise RuntimeError(error_string)
    
    return bq_schema


def convert_bigquery_dtype_to_python_dtype(dtype: str):
    if dtype == 'STRING' or dtype == 'DATE' or dtype == 'DATETIME': return str
    if dtype == 'NUMERIC': return float
    if dtype == 'BOOLEAN': return bool
    raise NotImplementedError(f'Unsupported dtype: {dtype}')


def create_type_dict(schema) -> dict:
    '''Convert `schema` which is in JSON format into a dictionary that maps the name of each item in `schema` 
    to the Python dtype.'''
    type_dict = dict()
    for schema_dict in schema:
        name, mode = schema_dict['name'], schema_dict['mode']
        if mode == 'REPEATED':
            type_dict[name] = create_type_dict(schema_dict['fields'])
        else:
            assert mode == 'NULLABLE'
            type_dict[name] = convert_bigquery_dtype_to_python_dtype(schema_dict['type'])
    return type_dict


BQ_SCHEMA = create_bq_schema()
BQ_SCHEMA_TYPES = create_type_dict(BQ_SCHEMA)    # map the name to the Python dtype; the dictionary may be nested if the name correspondes to a field that is REPEATED

TAGS_WITH_REPEATED_ELEMENTS = {schema_dict['name']: [field['name'] for field in schema_dict['fields']] for schema_dict in BQ_SCHEMA if schema_dict['mode'] == 'REPEATED'}
TAGS_WITHOUT_REPEATED_ELEMENTS = [schema_dict['name'] for schema_dict in BQ_SCHEMA if schema_dict['mode'] != 'REPEATED']
ALL_TAGS_SET = set(TAGS_WITHOUT_REPEATED_ELEMENTS + list(TAGS_WITH_REPEATED_ELEMENTS.keys()))

BQ_SCHEMA_TO_UPLOAD_TO_BIGQUERY = convert_bq_schema_to_use_bigquery_constructor(BQ_SCHEMA)


def process_repeated_element(child_name: str, child_element):
    child_details = dict()
    for subchild in child_element:
        subtag = subchild.tag.split('}')[-1]
        child_details[subtag] = subchild.text

    TAGS = TAGS_WITH_REPEATED_ELEMENTS if child_name in TAGS_WITH_REPEATED_ELEMENTS else TAGS_WITH_FLAT_ELEMENTS
    assert child_name in TAGS, f'{child_name} not in either `TAGS_WITH_REPEATED_ELEMENTS` or `TAGS_WITH_FLAT_ELEMENTS`'
    for missing_child_tag in set(TAGS[child_name]) - set(child_details.keys()):
        child_details[missing_child_tag] = None    # set tags with missing values to `None` by default to store as NULL in BigQuery
    
    if child_name in TAGS_TO_PUT_AS_PREFIX:
        child_details = {add_prefix_to_field(child_name, name): subchild for name, subchild in child_details.items()}
    return child_details


def process_element(element, file_name: str):
    element_data = dict()
    repeated_elements = defaultdict(list)    # since these are repeated elements, the underlying data structure will be a list that stores each of them

    for child in element:
        tag = child.tag.split('}')[-1]    # Remove namespace
        if tag in TAGS_WITH_REPEATED_ELEMENTS:
            repeated_elements[tag].append(process_repeated_element(tag, child))
        elif tag in TAGS_WITH_FLAT_ELEMENTS:
            element_data = element_data | process_repeated_element(tag, child)    # concatenate two dictionaries together
        else:
            element_data[tag] = child.text

    for tag in TAGS_WITH_REPEATED_ELEMENTS:
        element_data[tag] = repeated_elements[tag] if tag in repeated_elements else None

    for missing_tag in ALL_TAGS_SET - set(element_data.keys()):
        element_data[missing_tag] = None    # set tags with missing values to `None` by default to store as NULL in BigQuery
    
    #element_data['file_received_from_sp_date'] = file_name[:10].replace('/', '-')    # assumes that the first 10 characters are the date because it is coming from a folder of format: YYYY/MM/DD, e.g., `2024/07/02`

    # Parse date from the file name format "Muni_Full Universe-2024110835.xml"
    # 2024110835 breaks down as: 2024 (year) 11 (month) 08 (day) 35 (timestamp)
    timestamp_str = file_name.split('-')[1][:8]  # Gets '20241108' from '2024110835.xml'
    element_data['file_received_from_sp_date'] = f"{timestamp_str[:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]}"  # Formats as '2024-11-08'

    element_data['upload_datetime'] = datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S')
    element_data['gcp_storage_file_path'] = SP_REF_DATA_GOOGLE_CLOUD_BUCKET_NAME + '/' + file_name
    return element_data


def parse_xml(xml_as_bytes, file_name: str):
    print(f'Parsing: {file_name}')
    context = etree.iterparse(io.BytesIO(xml_as_bytes), events=('end',), tag='{http://dictionary.markit.com/ddgen/firef_xsd_consumer_1}Instrument')
    data = []
    for event, elem in context:    # each file is of size 10000
        data.append(process_element(elem, file_name))
        elem.clear()    # clear the contents of the current element to free memory
        while elem.getprevious() is not None:    # delete previous siblings of the current element from its parent to free up more memory
            del elem.getparent()[0]
    return data


def convert_schema_type_to_python_type(dtype, value: str):
    if value is None: return None
    # no type conversions need to occur for DATE or STRING since the string representation of the value is sufficient to load into BigQuery
    if dtype == bool:
        if value == 'false': return False
        if value == 'true': return True
        raise NotImplementedError(f'Unsupported boolean value: {value}')
    if dtype == float:
        return round(float(value), 9)    # `NUMERIC` data type only supports up to 9 decimal places: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#numeric-type
    return value


def convert_row_from_schema_type_to_python_type(row_dict: dict, schema_type_dict: dict):
    converted_row = dict()
    for name, value in row_dict.items():
        schema_type = schema_type_dict[name]
        is_nested_row = isinstance(schema_type, dict) and value is not None
        converted_row[name] = [convert_row_from_schema_type_to_python_type(nested_row_dict, schema_type) for nested_row_dict in value] if is_nested_row else convert_schema_type_to_python_type(schema_type, value)
    return converted_row


def convert_parsed_xml_to_bigquery_data(schema_type_dict: dict, data) -> list:
    return [convert_row_from_schema_type_to_python_type(row, schema_type_dict) for row in data]
