import os
from io import StringIO, BytesIO
import pandas as pd 
import numpy as np
from lxml import etree
from google.cloud import bigquery
import pytz
from datetime import datetime
from google.api_core.exceptions import BadRequest

my_timezone = pytz.timezone('America/New_York')

# Main change will hapen here as the file should be read either from the FTP or cloud storage
path = "../../gsm_init_muni_APFICC_GSMF10I.1.1_1.20201203T1300-05.xml"


PROJECT = 'eng-reactor-287421'
dataset = 'FULL_ICE'
table = 'events'
bqclient = bigquery.Client(project=PROJECT,)


def load_data(bq,data,project,dataset,table):
    bq = bq
    table_id = '{}.{}.{}'.format(project,dataset,table)
    job_config = bigquery.LoadJobConfig(schema =[])
    job = bq.load_table_from_dataframe(data, table_id,job_config=job_config)

    try:
        job.result() # Waits for the job to complete.
        return 'success'  
    except BadRequest as ex:
        print(ex) 

def data_loaded_threading(bqclient,doc_df,PROJECT,dataset,table,count):
    try:
        test = load_data(bqclient,doc_df,PROJECT,dataset,table)
        print(test)
    except Exception as e:
        print(e)


def events(new_elem,final_list):
    
    list_of_elem = new_elem.xpath("//instrument/debt")
    instrument_id = new_elem.xpath("//instrument")[0].attrib["id"]
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            if element.tag == "event":
                new_dict = {}
                new_dict.update({"instrument_id":instrument_id})
                new_dict.update({"event_code":element.attrib["code"]})
                new_dict.update({"event_time":element.attrib["date"]})
                new_dict.update({"event_type":element.attrib["type"]})
                new_dict.update({"event_text":element.text})

                final_list.append(new_dict)
                
def create_df_events(new_elem,final_list):
    events(new_elem,final_list)

def upload_dataframe(final_list,count,data_timestamp):
    doc_df = pd.DataFrame(final_list)
    if len(doc_df)>0:
        new_columns = []
        for columns in doc_df.columns:
            columns = columns.replace("&","")
            columns = columns.replace(" ","_")
            columns = columns.replace("-","_")
            new_columns.append(columns)
        doc_df.columns = new_columns
        doc_df["upload_date"] = my_timezone.localize(datetime.now()).date() 
        doc_df["data_timestamp"] = pd.to_datetime(data_timestamp[0])
        doc_df["data_timestamp"] = doc_df["data_timestamp"].dt.date
        doc_df["event_time"] = pd.to_datetime(doc_df["event_time"])

        doc_df["event_date"] = doc_df["event_time"].dt.date
        del doc_df["event_time"]
        

        doc_df.replace(np.nan,None,inplace = True)
        doc_df.replace([np.nan],[None],inplace = True)
        doc_df.replace("nan",None,inplace = True)
        data_loaded_threading(bqclient,doc_df,PROJECT,dataset,table,count)
    

count = 0
total = 0
context = etree.iterparse(path,events=("start",),tag = "instrument")

for j,k in context:
    count += 1
    while k.getprevious() is not None:    
        del k.getparent()[0]
else:
    total = count

data_timestamp = []
head_context = etree.iterparse(path,events=("start",),tag = "header")
for event,elem in head_context:
    new_elem = etree.parse(BytesIO(etree.tostring(elem)))
    list_of_elem = elem.xpath("//header")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
#             print(element.tag)
            if element.tag == "timestamp":
                print("hello")
                data_timestamp.append(element.text)
    break


def new_fast_iter(context,data_timestamp, *args, **kwargs):
  
    final_list = []
    count = 0

    for _,elem in context:
        new_elem = etree.parse(BytesIO(etree.tostring(elem)))
        create_df_events(new_elem,final_list)
        count += 1
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if count %30000== 0:
            upload_dataframe(final_list,count,data_timestamp)


            final_list = []


        if count == total:
            upload_dataframe(final_list,count,data_timestamp)


            final_list = []
            



context = etree.iterparse(path,events=("start",),tag = "instrument")
new_fast_iter(context,data_timestamp)            
                    
