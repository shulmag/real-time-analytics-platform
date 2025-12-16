import os
from io import StringIO, BytesIO
import pandas as pd 
import numpy as np
from lxml import etree
from google.cloud import bigquery
import pytz
from datetime import datetime
from google.api_core.exceptions import BadRequest


#Setting up the global variables to be used.
my_timezone = pytz.timezone('America/New_York')

# Main change will hapen here as the file should be read either from the FTP or cloud storage
path = "../../gsm_init_muni_APFICC_GSMF10I.1.1_1.20201203T1300-05.xml"



# Project, data and table configuration
PROJECT = 'eng-reactor-287421'
dataset = 'FULL_ICE'
table = 'call_schedule'



bqclient = bigquery.Client(project=PROJECT,)

#helper function to load the final dataframe to bigquery
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

def load_dataframe(bqclient,doc_df,PROJECT,dataset,table,count):
    try:
        test = load_data(bqclient,doc_df,PROJECT,dataset,table)
        print(test)
    except Exception as e:
        print(e)
        for columns in doc_df.columns:
            print(columns)
        print(doc_df.head)
        print(count)

def upload_dataframe(final_list,count,data_timestamp):
    doc_df = pd.DataFrame(final_list)
    
    new_columns = []
     # Changing columns name to remove special characters
    for columns in doc_df.columns:
        columns = columns.replace("&","")
        columns = columns.replace(" ","_")
        columns = columns.replace("-","_")
        new_columns.append(columns)
    doc_df.columns = new_columns
    doc_df["upload_date"] = my_timezone.localize(datetime.now()).date() 
    doc_df["data_timestamp"] = pd.to_datetime(data_timestamp[0])
    doc_df["data_timestamp"] = doc_df["data_timestamp"].dt.date
    doc_df.replace(np.nan,None,inplace = True)
    doc_df.replace([np.nan],[None],inplace = True)
    doc_df.replace("nan",None,inplace = True)
    load_dataframe(bqclient,doc_df,PROJECT,dataset,table,count)

def call_schedule(new_elem,final_list):
    """Function to parse the call_schedule tag."""
    
    list_of_elem = new_elem.xpath("//instrument/debt/call_details")
    instrument_id = new_elem.xpath("//instrument")[0].attrib["id"]
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            if element.tag == "call_schedule":
                new_dict = {}
                new_dict.update({"instrument_id":instrument_id})

                for childs in element.getchildren():           
                    new_dict.update({childs.tag:childs.text})
                    
                final_list.append(new_dict)

def create_df_call_schedule(new_elem,final_list):    
    call_schedule(new_elem,final_list)

count = 0
total = 0
context = etree.iterparse(path,events=("start",),tag = "instrument")
"""Loops over the context to find the exact number of instruments in the xml file. This is used when creating batch size
    and uploading the final residual amount at the end"""
for j,k in context:
    count += 1
    while k.getprevious() is not None:    
        del k.getparent()[0]
else:
    total = count



data_timestamp = []
"""Creating an iterator to get the timestamp of the data."""
head_context = etree.iterparse(path,events=("start",),tag = "header")
for event,elem in head_context:
    new_elem = etree.parse(BytesIO(etree.tostring(elem)))
    list_of_elem = elem.xpath("//header")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
#             print(element.tag)
            if element.tag == "timestamp":
               
                data_timestamp.append(element.text)
    break

def new_fast_iter(context,data_timestamp,):
    """Main iterator function to loop over the elements """
    final_list = []
    count = 0

    for _,elem in context:
        new_elem = etree.parse(BytesIO(etree.tostring(elem)))
        create_df_call_schedule(new_elem,final_list)
        
        count += 1

        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if count %10000== 0:
            upload_dataframe(final_list,count,data_timestamp)
            final_list = []


        if count == total:
            upload_dataframe(final_list,count,data_timestamp)
            final_list = []
            



context = etree.iterparse(path,events=("start",),tag = "instrument")
new_fast_iter(context,data_timestamp)