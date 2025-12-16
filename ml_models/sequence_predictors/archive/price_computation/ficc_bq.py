import os
import pandas as pd
from google.cloud import bigquery

bq_client = bigquery.Client()
PROJECT_ID = "eng-reactor-287421"

def get_trades_data(date_str):
    DATA_QUERY = f'''Select * from `eng-reactor-287421.primary_views.trade_history_for_training_no_neg_yields` 
                  where trade_date ='{date_str}' and yield >0 and yield <=3 and par_traded is not null and sp_long is not null
                  AND msrb_valid_to_date > current_date ORDER BY trade_date'''

    PATH = f"trades_data_{date_str}.pkl"
    
    if os.path.isfile(PATH):
        print("Reading from file")
        ref_data = pd.read_pickle(PATH)
        print("File read")
    
    else:
        print("Running query")
        ref_data = bq_client.query(DATA_QUERY).result().to_dataframe()
        print("Saving data")
        ref_data.to_pickle(PATH)
    
    return(ref_data)