'''
'''
from datetime import datetime
import logging as python_logging    # to not confuse with google.cloud.logging

from auxiliary_variables import BQ_CLIENT, USAGE_DATA_LOGGING_TABLE, USAGE_DATA_INTERNAL_LOGGING_TABLE, LOGGING_JOB_CONFIG, LOGGING_FEATURES, RECENT_FEATURES


YEAR_MONTH_DAY_HOUR_MIN_SEC = '%Y-%m-%d %H:%M:%S'


def get_table_name(internal_usage_table: bool = False) -> str:
    '''Return the appropriate table name based on whether `internal_usage_table` is `True` or `False`. 
    This is used to determine which BigQuery table to upload to.'''
    return USAGE_DATA_INTERNAL_LOGGING_TABLE if internal_usage_table else USAGE_DATA_LOGGING_TABLE


def upload_usage_dicts_to_bigquery(list_of_usage_dicts: list, internal_usage_table: bool = False) -> None:
    '''Upload `list_of_usage_dicts` to the BigQuery table for logs.'''
    load_job = BQ_CLIENT.load_table_from_json(list_of_usage_dicts, get_table_name(internal_usage_table), job_config=LOGGING_JOB_CONFIG)
    ## taken directly from `ficc/app_engine/demo/server/modules/logging_functions.py`
    try:
        load_job.result()    # waits for the job to complete
    except Exception as e:
        python_logging.info(f'{type(e)}: {e} in `upload_usage_dicts_to_bigquery(...)`')
        python_logging.info(load_job.errors)
        raise e
    

def upload_usage_dicts_to_bigquery_stream(list_of_usage_dicts: list, internal_usage_table: bool = False, chunk_size: int = None) -> None:
    '''Upload `list_of_usage_dicts` to the BigQuery table for logs using streaming inserts (`insert_rows_json`). 
    Avoids load job quota limits, which result in the following error:
    `google.api_core.exceptions.Forbidden: 403 Quota exceeded: Your table exceeded quota for imports or query appends per table. For more information, see https://cloud.google.com/bigquery/docs/troubleshoot-quotas; reason: quotaExceeded, location: load_job_per_table.long`.
    Streaming inserts costs $0.01 per 200 MB inserted and is buffered for ~90 minutes before being fully available in BigQuery.'''
    table_name = get_table_name(internal_usage_table)    # get the table name based on whether it's internal or not
    table = BQ_CLIENT.get_table(table_name)    # retrieve table reference
    if chunk_size is None: chunk_size = len(list_of_usage_dicts)
    for start_idx in range(0, len(list_of_usage_dicts), chunk_size):
        try:
            errors = BQ_CLIENT.insert_rows_json(table, list_of_usage_dicts[start_idx : start_idx + chunk_size])    # insert rows into BigQuery; if the list inserted is too large (> 500 items) then the function requires > 8GB of memory (causing a memory issue) because the entire list is serialized as part of the function call which is why it is called in chunks
            if errors:    # check for and handle errors
                python_logging.error(f'BigQuery streaming insert errors: {errors}')
                raise RuntimeError(f'Streaming insert errors: {errors}')
        except Exception as e:
            python_logging.error(f'{type(e)}: {e} in `upload_usage_dicts_to_bigquery()`')
            raise e
    
    python_logging.info(f'Successfully inserted {len(list_of_usage_dicts)} rows into {table_name} using streaming insert')


def upload_usage_dicts_to_bigquery_append(list_of_usage_dicts: list, internal_usage_table: bool = False) -> None:
    '''Upload usage logs to the BigQuery table for logs using the BigQuery Storage Write API's
    `AppendRowsStream` method. This approach is ideal for high-throughput, real-time streaming
    inserts and avoids load job quota limits, which result in the following error:
    `google.api_core.exceptions.Forbidden: 403 Quota exceeded: Your table exceeded quota for imports or query appends per table. For more information, see https://cloud.google.com/bigquery/docs/troubleshoot-quotas; reason: quotaExceeded, location: load_job_per_table.long`.
    After investigation, the quota cannot be increased (see https://cloud.google.com/bigquery/docs/troubleshoot-quotas). 
    Based on https://cloud.google.com/bigquery/docs/write-api-streaming#at-least-once, the best way 
    to avoid the quota is to use AppendRowsStream.

    The JSON rows must be converted into the Protobuf format that matches the BigQuery table schema.
    NOTE: this function does not work due to Segmentation Fault errors associated with `Protobuf`. Many 
    hacky solutions were attempted, but none resulted in success. In the logs the error is: `Uncaught signal: 11, pid=8, tid=10, fault_addr=8.`
    Hypothesis: there is an issue with creating the `repeated` record holding the last five yield spreads in the history.'''
    # lazy loading
    from google.cloud.bigquery_storage_v1 import BigQueryWriteClient
    from google.cloud.bigquery_storage_v1.types import ProtoRows, AppendRowsRequest, ProtoSchema
    from google.protobuf.descriptor_pb2 import DescriptorProto
    from google.protobuf import message_factory

    try:
        table_name = get_table_name(internal_usage_table)    # get the table name based on whether it's internal or not
        project_id, dataset_id, table_id = table_name.split('.')    # extract project, dataset, and table names from the fully qualified table ID
        table_path = f'projects/{project_id}/datasets/{dataset_id}/tables/{table_id}'
        write_stream = f'{table_path}/streams/_default'    # the special `_default` stream commits rows immediately

        bq_write_client = BigQueryWriteClient()    # initialize the BigQuery Storage Write client
        
        descriptor_proto = DescriptorProto()    # define Protobuf descriptor for the schema; acts as a blueprint for the Protobuf message analogous to a BigQuery schema
        descriptor_proto.name = 'LoggingEntry'    # can be anything since it does not affect how the data is stored in BigQuery

        # map BigQuery data types to Protobuf types
        bigquery_to_protobuf = {'STRING': 9,       # TYPE_STRING
                                'BOOLEAN': 8,      # TYPE_BOOL
                                'INTEGER': 5,      # TYPE_INT32
                                'NUMERIC': 2,      # TYPE_FLOAT  (Protobuf does not support NUMERIC, so use FLOAT)
                                'FLOAT': 2,        # TYPE_FLOAT
                                'TIMESTAMP': 3}    # TYPE_INT64 (Unix timestamp in microseconds)

        for field_idx, (field_name, field_type) in enumerate(LOGGING_FEATURES.items()):    # create Protobuf schema based on `LOGGING_FEATURES`
            proto_field = descriptor_proto.field.add()
            proto_field.name = field_name
            proto_field.number = field_idx + 1    # need to increment by 1 since each field must have a unique field number starting from 1 (not 0)
            proto_field.label = 1    # 1: label is optional, 2: label is required (deprecated in Protobuf 3+), 3: label is repeated (multiple values, e.g. array, list)
            proto_field.type = bigquery_to_protobuf.get(field_type.upper(), 9)    # default to STRING

        # handle repeated RECORD field `recent`
        recent_proto_field = descriptor_proto.field.add()
        recent_proto_field.name = 'recent'
        recent_proto_field.number = len(LOGGING_FEATURES) + 1    # need to increment by 1 since each field must have a unique field number starting from 1 (not 0) and we have already assigned all of the `LOGGING_FEATURES` upstream
        recent_proto_field.label = 3    # 1: label is optional, 2: label is required (deprecated in Protobuf 3+), 3: label is repeated (multiple values, e.g. array, list)
        recent_proto_field.type = 11    # TYPE_MESSAGE; analagous to RECORD in BigQuery
        recent_proto_field.type_name = 'RecentFields'    # can be anything since it does not affect how the data is stored in BigQuery

        # define `recent` sub-message (nested RECORD)
        recent_descriptor_proto = descriptor_proto.nested_type.add()
        recent_descriptor_proto.name = 'RecentFields'
        for recent_feature_idx, recent_feature in enumerate(RECENT_FEATURES):
            recent_field = recent_descriptor_proto.field.add()
            recent_field.name = recent_feature
            recent_field.number = recent_feature_idx + 1    # need to increment by 1 since each field must have a unique field number starting from 1 (not 0)
            recent_field.label = 1    # 1: label is optional, 2: label is required (deprecated in Protobuf 3+), 3: label is repeated (multiple values, e.g. array, list)
            recent_field.type = bigquery_to_protobuf['NUMERIC']    # store `NUMERIC` as `FLOAT`

        LoggingEntry = message_factory.GetMessageClass(descriptor_proto)    # create a dynamic Protobuf message class using the copied version

        # convert each row into Protobuf format
        proto_rows = ProtoRows()
        for row in list_of_usage_dicts:
            proto_row = LoggingEntry()
            
            # convert standard fields
            for field_name, field_type in LOGGING_FEATURES.items():
                if field_name in row:
                    if field_type.lower() == 'timestamp':
                        dt_obj = datetime.strptime(row[field_name], YEAR_MONTH_DAY_HOUR_MIN_SEC)    # convert string to datetime object; this is from the server logging having `time` as a string
                        setattr(proto_row, field_name, int(dt_obj.timestamp() * 1_000_000))    # convert to Unix timestamp (microseconds since epoch)
                    else:
                        setattr(proto_row, field_name, row[field_name])

            # convert `recent` field (nested RECORD)
            if 'recent' in row and isinstance(row['recent'], dict):
                recent_proto = proto_row.recent.add()
                for recent_feature in RECENT_FEATURES:
                    if recent_feature in row['recent']:
                        setattr(recent_proto, recent_feature, float(row['recent'][recent_feature]))

            proto_rows.serialized_rows.append(proto_row.SerializeToString())    # append serialized `Protobuf` row

        # create the `AppendRowsRequest`
        request = AppendRowsRequest(write_stream=write_stream,
                                    proto_rows=AppendRowsRequest.ProtoData(rows=proto_rows),
                                    proto_schema=ProtoSchema(proto_descriptor=descriptor_proto.SerializeToString()))
        responses = bq_write_client.append_rows(iter([request]))    # send the request using `AppendRowsStream`
        rows_appended = len(list_of_usage_dicts)    # track successfully appended rows

        for response in responses:
            if response.error and response.error.code != 0:
                python_logging.error(f'AppendRowsStream error: {response.error.message}')
                raise Exception(f'AppendRowsStream error: {response.error.message}')

        python_logging.info(f'Successfully appended {rows_appended} rows into {table_name} using `AppendRowsStream`')

    except Exception as e:
        python_logging.error(f'{type(e)}: {e} in `upload_usage_dicts_to_bigquery_append()`')
        raise e
