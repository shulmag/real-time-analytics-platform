
import paramiko
import os
from google.cloud import bigquery
from google.cloud import storage
from io import StringIO
from time import sleep
import pandas as pd
import table_schema as table_schema
import pytz
import numpy as np
from datetime import datetime
 

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



# The below code is for loading historical issues data
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
    table = 'cusip_issues'
    bq = bigquery.Client(project=PROJECT)
    if not doesTableExist(bq,PROJECT,dataset,table):
        create_table_with_schema(bq,PROJECT,dataset,table,table_schema.get_table_schema(table))
    if sftp.getcwd() != '/Inbox':
        sftp.chdir(r'.\Inbox')
    
    file_list = sftp.listdir()
    file_stats = [sftp.stat(file) for file in file_list] # Returns file metadata for each file.
    column_header = ["issuer_num","issue_num","issue_check","issue_description","issue_additional_info","issue_status",
            "issue_type_code","dated_date","maturity_date","partial_maturity","coupon_rate",
            "currency_code","security_type_description","fisn","issue_group","isin","where_traded",
            "ticker_symbol","us_cfi_code","iso_cfi_code","issue_entry_date","alternative_minimum_tax","bank_qualified",
            "callable","first_coupon_date","initial_public_offering","payment_frequency_code","closing_date",
            "dtc_eligible","pre_refunded","refundable","remarketed","sinking_fund","taxable","bond_form","enhancements",
            "fund_distribution_policy","fund_investment_policy","fund_type","guarantee","income_type","insured_by",
            "ownership_restriction","payment_status","preferred_type","putable","rate_type","redemption",
            "source_document","sponsoring","voting_rights","warrant_assets","warrant_status","warrant_type",
            "underwriter","auditor","paying_agent","tender_agent","transfer_agent","bond_counsel","financial_advisor",
            "municipal_sale_date","sale_type","offering_amount","offering_amount_code","issue_transaction","issue_last_update_date",
            "obligator_name","obligor_cusip_issuer_number","co_obligor_name","co_obligor_cusip_issuer_number","government_stimulus_program",
            "reserved_6","reserved_7","reserved_8","reserved_9","reserved_10"]
    for i in range(0,len(file_list)):
        if file_list[i] != "ACUMASTER_ISSUER.PIP.zip" and file_list[i] != "ACUMASTER_ISSUE.PIP.gz" and ('E' in file_list[i]):
            # print(file_list[i])
            data = pd.read_csv(sftp.open(file_list[i]),sep = "|",names = column_header,header = None)
            
            data = data[data["issuer_num"]!= '999999']
          
            data = data[~pd.isnull(data["issue_num"]) & ~pd.isnull(data["issue_check"])]
            
            data["issue_check"] = data["issue_check"].astype(int)
            float_columns = ["offering_amount","coupon_rate"]
            for columns in data.columns:
                if columns not in float_columns: 
                    if data[columns].dtype == "float64":
                        data[columns] = data[columns].astype(str)
            year_month_day = str(my_timezone.localize(datetime.fromtimestamp(file_stats[i].st_mtime)).date().year) +"-"+file_list[i].split("-")[0][-2:]+"-"+file_list[i].split("-")[1][:2]
            data["upload_date"] = datetime.strptime(year_month_day, '%Y-%m-%d')

            for columns in ["dated_date","maturity_date","issue_entry_date","first_coupon_date","closing_date","municipal_sale_date","issue_last_update_date"]:
                data[columns] = pd.to_datetime(data[columns],format = "%Y-%m-%d",exact = True,errors = 'coerce').dt.date
            data = data.replace("nan",np.nan)
            load_data(bq,data,PROJECT,dataset,table)
        elif file_list[i] == "ACUMASTER_ISSUE.PIP.gz":
            
            # print(file_list[i])
            
            path = "/Users/FICC.AI/Documents/Data/CUSIP/ACUMASTER_ISSUE.PIP"
            chunksize = 10 ** 5
            for data in pd.read_csv(path,sep = "|",names = column_header ,header = None, chunksize=chunksize):
                data = data[data["issuer_num"]!= '999999']
            # print(len(data))
                data = data[~pd.isnull(data["issue_num"]) & ~pd.isnull(data["issue_check"])]
                # data.replace([np.nan],[None],inplace = True)
                data["issue_check"] = data["issue_check"].astype(int)
                float_columns = ["offering_amount","coupon_rate"]
                for columns in data.columns:
                    if columns not in float_columns:    
                        if data[columns].dtype == "float64" or data[columns].dtype == "object":
                            data[columns] = data[columns].astype(str)
                     

                # year_month_day = str(my_timezone.localize(datetime.fromtimestamp(file_stats[i].st_mtime)).date().year) +"-"+file_list[i].split("-")[0][-2:]+"-"+file_list[i].split("-")[1][:2]
                data["upload_date"] = my_timezone.localize(datetime.fromtimestamp(file_stats[i].st_mtime)).date()
                # print(data["upload_date"].head)
                for columns in ["dated_date","maturity_date","issue_entry_date","first_coupon_date","closing_date","municipal_sale_date","issue_last_update_date"]:
                    data[columns] = pd.to_datetime(data[columns],format = "%Y-%m-%d",exact = True,errors = 'coerce').dt.date
                data = data.replace("nan",np.nan)
                load_data(bq,data,PROJECT,dataset,table)
            
    sftp.close()
    transport.close()