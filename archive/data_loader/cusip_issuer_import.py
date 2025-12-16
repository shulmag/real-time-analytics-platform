# Cloud function implementation
import paramiko
import os
from google.cloud import bigquery
from google.cloud import storage
from io import StringIO
from time import sleep
import pandas as pd
import pytz
from datetime import datetime
import table_schema as table_schema

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/FICC.AI/Documents/Code/ficc/credentials.json"
# Helper function to create a new table with a specified schema
def create_table_with_schema(bq,project_id,dataset,table_id,schema = None):
    PROJECT = project_id
    bq = bq
    table_id = '{}.{}.{}'.format(PROJECT,dataset,table_id)
    table = bq.create_table(table_id, exists_ok=True)
    print('{} created on {}'.format(table.table_id, table.created))
    table = bq.get_table(table_id)
    table.schema = schema
    table = bq.update_table(table, ["schema"])

# Helper function to load a pandas dataframe to a bigquery table.    
def load_data(bq,data,project,dataset,table):
    bq = bq
    table_id = '{}.{}.{}'.format(project,dataset,table)
    job_config = bigquery.LoadJobConfig(schema = table_schema.get_table_schema(table))
    job = bq.load_table_from_dataframe(data, table_id,job_config=job_config)
    job.result() # blocks and waits
    print("Loaded {} rows into {}".format(job.output_rows, table_id))
    print('Num rows = ', bq.get_table(table_id).num_rows)
# Helper function to chech if a table exists
def doesTableExist(bq,project_id, dataset_id, table_id):
    try:
        table_id = '{}.{}.{}'.format(project_id,dataset_id,table_id)
        bq.get_table(table_id)
        return True
    except:
        return False
#Helper function which fetches a schema for a particular table


# The below code is for loading historical issuer data
if __name__ == "__main__":
   
    my_timezone = pytz.timezone('America/New_York')


    host = 'edx.standardandpoors.com'
    port = 22
    user = 'Ficc6333'
    pwd = '4dquWX$2'

    transport = paramiko.Transport((host, port))
    transport.connect(username = user, password = pwd)
    sftp = paramiko.SFTPClient.from_transport(transport)
    PROJECT = 'eng-reactor-287421'
    dataset = 'cusip_global_services'
    table = 'cusip_issuer'
    bq = bigquery.Client(project=PROJECT)
    if not doesTableExist(bq,PROJECT,dataset,table):
        create_table_with_schema(bq,PROJECT,dataset,table,table_schema.get_table_schema(table))
    if sftp.getcwd() != '/Inbox':
        sftp.chdir(r'.\Inbox')
    
    file_list = sftp.listdir()
    file_stats = [sftp.stat(file) for file in file_list] # Returns file metadata for each file.
 
    for i in range(0,len(file_list)):
        if file_list[i] != "ACUMASTER_ISSUER.PIP.zip" and file_list[i] != "ACUMASTER_ISSUE.PIP.gz" and ('R' in file_list[i]):
            print(file_list[i])
            data = pd.read_csv(sftp.open(file_list[i]),sep = "|",names = ['issuer_num',"issuer_check","issuer_name","issuer_adl","issuer_type",
            "issuer_status","domicile","state_cd","cabre_id","cabre_status","lei_gmei","legal_entity_name","previous_name","issuer_entry_date","cp_institution_type_desc","issuer_transaction",
            "issuer_update_date","reserved_1","reserved_2","reserved_3","reserved_4","reserved_5","reserved_6","reserved_7","reserved_8","reserved_9","reserved_10"],header = None)
           
            year_month_day = str(my_timezone.localize(datetime.fromtimestamp(file_stats[i].st_mtime)).date().year) +"-"+file_list[i].split("-")[0][-2:]+"-"+file_list[i].split("-")[1][:2]
            data["upload_date"] = datetime.strptime(year_month_day, '%Y-%m-%d') 
            
            data = data[~pd.isnull(data["issuer_check"])]
            data["issuer_check"] = data["issuer_check"].astype(int)
            data["issuer_type"] = data["issuer_type"].replace({"M": "Municipal Issuer"})
            data["cabre_status"] = data["cabre_status"].astype(str)
            data["cabre_status"] = data["cabre_status"].replace({"A":"Active","D":"Delete"})
            data["issuer_status"] = data["issuer_status"].replace({"A": "Active", "S": "Suspend","D": "Suspend"})
            # data["issuer_transaction"] = data["issuer_transaction"].astype(str)
            # data["issuer_transaction"]  = data["issuer_transaction"].replace({"A": "New Issuer", "R": "Suspend Drop","B": "New Issuer Company Name Change","T":"Suspend Drop",""})
            
            for columns in ["issuer_entry_date","issuer_update_date"]:
                data[columns] = pd.to_datetime(data[columns],format = "%Y-%m-%d",exact = True,errors = 'coerce').dt.date
            
            load_data(bq,data,PROJECT,dataset,table)
        elif file_list[i] == "ACUMASTER_ISSUER.PIP.zip" :
            print(file_list[i]) 
            path = "/Users/FICC.AI/Documents/Data/CUSIP/ACUMASTER_ISSUER.PIP"
            data = pd.read_csv(path,sep = "|",names = ['issuer_num',"issuer_check","issuer_name","issuer_adl","issuer_type",
            "issuer_status","domicile","state_cd","cabre_id","cabre_status","lei_gmei","legal_entity_name","previous_name","issuer_entry_date","cp_institution_type_desc","issuer_transaction",
            "issuer_update_date","reserved_1","reserved_2","reserved_3","reserved_4","reserved_5","reserved_6","reserved_7","reserved_8","reserved_9","reserved_10"],header = None)
            data["upload_date"] = my_timezone.localize(datetime.fromtimestamp(file_stats[i].st_mtime)).date() 
        
            data = data[~pd.isnull(data["issuer_check"])]
            data["issuer_check"] = data["issuer_check"].astype(int)
            data["issuer_type"] = data["issuer_type"].replace({"M": "Municipal Issuer"})
            data["cabre_status"] = data["cabre_status"].replace({"A":"Active","D":"Delete"})
            data["issuer_status"] = data["issuer_status"].replace({"A": "Active", "S": "Suspend","D": "Suspend"})
            # data["issuer_transaction"]  = data["issuer_transaction"].replace({"A": "New Issuer", "R": "Name Change","W":"Registration Withdrawn"})
            for columns in ["issuer_entry_date","issuer_update_date"]:
                data[columns] = pd.to_datetime(data[columns],format = "%Y-%m-%d",exact = True,errors = 'coerce').dt.date
            load_data(bq,data,PROJECT,dataset,table)
    sftp.close()
    transport.close()