import pandas as pd
from google.cloud import bigquery
import decimal
import numpy as np
import datetime as dt
from datetime import datetime
import table_schema as table_schema
import pytz
import os

# my comment

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/FICC.AI/Documents/Code/ficc/Credentials.json"

def create_table_with_schema(bq,project_id,dataset,table_id,schema = None):
    PROJECT = project_id
    bq = bq
    table_id = '{}.{}.{}'.format(PROJECT,dataset,table_id)
    table = bq.create_table(table_id, exists_ok=True)
    print('{} created on {}'.format(table.table_id, table.created))
    table = bq.get_table(table_id)
    table.schema = schema
    table = bq.update_table(table, ["schema"])
def load_data(bq,data,project,dataset,table):
    bq = bq
    table_id = '{}.{}.{}'.format(project,dataset,table)
    job = bq.load_table_from_dataframe(data, table_id,)
    job.result() # blocks and waits
    print("Loaded {} rows into {}".format(job.output_rows, table_id))
    print('Num rows = ', bq.get_table(table_id).num_rows)

def doesTableExist(bq,project_id, dataset_id, table_id):
    try:
        table_id = '{}.{}.{}'.format(project_id,dataset_id,table_id)
        bq.get_table(table_id)
        return True
    except:
        return False
        
if __name__ == "__main__":
    path = "/Users/FICC.AI/Documents/Data/2019"
    delta_path = "/Users/FICC.AI/Documents/Data/Delta_data"
    PROJECT = 'eng-reactor-287421'
    dataset = 'MSRB'
    table = 'msrb_trade_t_20'
    column_header = ['rtrs_control_number','trade_type','cusip','security_description',
                                                        'dated_date','coupon','maturity_date','is_trade_before_issue','assumed_settlement_date','trade_date',
                                                        'time_of_trade','settlement_date','par_traded','dollar_price','yield_to_worst','brokers_broker','is_weighted_average_price','is_lop_or_takedown','publish_date','publish_time','version',
                                                        'unable_to_verify_dollar_price','is_alternative_trading_system','is_non_transaction_based_compensation',]
    bq = bigquery.Client(project=PROJECT)
    if not doesTableExist(bq,PROJECT,dataset,table):
        create_table_with_schema(bq,PROJECT,dataset,table,table_schema.get_table_schema(table))
    
    my_timezone = pytz.timezone('America/New_York')
    total_rows = 0
    total_records_saved = 0
    total_records_not_saved = 0

    for dir in os.listdir(path):
         if not dir.startswith('.'):
            for filename in os.listdir(os.path.join(path,dir)):
                print(os.path.join(path,dir,filename))
                data = pd.read_csv(os.path.join(path,dir,filename),sep = ",",    names = column_header,)
                data = data[data['par_traded'].notna() & data['dollar_price'].notna() & data['trade_date'].notna() & data['time_of_trade'].notna() ]
                data["yield_to_worst"] = data["yield_to_worst"].round(decimals = 4)
                data["par_traded"] = data["par_traded"].round(decimals = 4)
                data["coupon"] = data["coupon"].round(decimals = 4)
                data["dollar_price"] = data["dollar_price"].round(decimals = 4)
                data["is_alternative_trading_system"] = data["is_alternative_trading_system"] == 'Y'
                data["is_non_transaction_based_compensation"] = data["is_non_transaction_based_compensation"] == "Y"
                data["unable_to_verify_dollar_price"] = ~pd.isnull(data["unable_to_verify_dollar_price"])
                data["is_lop_or_takedown"] = data["is_lop_or_takedown"] == "Y"
                data["is_trade_before_issue"] = data["is_trade_before_issue"] == "Y"
                data["is_weighted_average_price"] = data["is_weighted_average_price"] == "Y"
                data["is_brokers_broker"] = (data["brokers_broker"] == "P") | (data["brokers_broker"] == "S")
                data["transaction_type"] = np.nan
                data["message_type"] = np.nan
                data["sequence_number"] = np.nan
                data["is_trade_with_a_par_amount_over_5MM"] = np.nan
                data["upload_date"] = my_timezone.localize(datetime.now()).date() 
                data["upload_time"] = my_timezone.localize(datetime.now()).time()
                for columns in ["trade_type","brokers_broker"]:
                    data[columns] = data[columns].astype(str)

                data.replace({'trade_type': {"S": "STC", "P": "PFC"},'brokers_broker':{"P":"PBB","S":"SBB"}},inplace = True)
                data[["trade_type","brokers_broker"]].replace("nan",np.nan,inplace = True)
                for columns in ["time_of_trade","publish_time"]:
                    data[columns] = data[columns].astype(str)
                    data[columns] = data[columns].apply(lambda x:"0" + x if len(x) == 5 else x)
                    data[columns] = pd.to_datetime(data[columns],format = "%H%M%S",exact = True,errors = 'coerce').dt.tz_localize(my_timezone).dt.time

                for columns in ["settlement_date","publish_date","dated_date","maturity_date","assumed_settlement_date","trade_date"]:
                    data[columns] = pd.to_datetime(data[columns],format = "%Y%m%d",exact = True,errors = 'coerce').dt.date
                total_rows += len(data)
                print("Toal recorded avaialbale {}".format(total_rows))
                # data_csv = data[data['par_traded'].isna()| data['dollar_price'].isna() | data['trade_date'].isna() | data['time_of_trade'].isna() ]
                # if len(data_csv) != 0:
                #     data_csv = data.join(data_csv, lsuffix='_caller', rsuffix='_other',how = "inner")
                #     data_csv[["par_traded_caller","dollar_price_caller","trade_date_caller","time_of_trade_caller"]].to_csv(os.path.join(delta_path,dir + filename),index = False)
                data = data[data['par_traded'].notna() & data['dollar_price'].notna() & data['trade_date'].notna() & data['time_of_trade'].notna() ]
                total_records_saved += len(data)
                print("Total records loaded {}".format(total_records_saved))
                data["time_of_trade"] = data.apply(lambda r : dt.datetime.combine(r['trade_date'],r['time_of_trade']),1)
                data["publish_time"] = data.apply(lambda r : dt.datetime.combine(r['publish_date'],r['publish_time']),1)
                data["upload_time"] =  data.apply(lambda r : dt.datetime.combine(r['upload_date'],r['upload_time']),1)
                data.replace([np.nan],[None],inplace = True)
                load_data(bq,data,PROJECT,dataset,table) 
        