import functions_framework
from auxiliary_functions import *
import datetime
import pytz
import os
import regex as re

#for pub/sub
import base64
import json
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/jupyter/ficc/isaac_creds.json"

# INITIALIZE WHEN CLOUD FUNCTION INSTANCE BEGINS 

last_updated = None
trades_df = pd.DataFrame(columns = trades_df_cols)
tz = pytz.timezone('US/Eastern')
processed_files = set()

@functions_framework.cloud_event    
def main(cloud_event):
    start = time.time()
    now = datetime.datetime.now(tz)
    
#     # for eventarc trigger
#     bucket = cloud_event.data["bucket"]
#     name = cloud_event.data["name"]
    
    #for pub/sub
    data = json.loads(base64.b64decode(cloud_event.data["message"]["data"]))
    bucket = data['bucket']
    name = data['name']
    
    if "from_fast_redis_update" in name: 
        print(f'{name} is a test file. Exiting function.')
        return

    try:
        timestamp = re.search(re.compile('\d{2}:\d{2}'), name)[0]

        #the timestamp on a file received can never be larger than the current time, because such a file does not exist 
        #it must hence be the same time or earlier, and if too long ago we exit the function
        if (now.minute - int(timestamp[3:]) > 3) or (now.hour != int(timestamp[:2])):
            print(f'{name} is asynchronous from time now, {now}. Exiting function.')
            return name
        
    except Exception as e: 
        print(f'Error handling file {name} . Exception: {e}')
        return
        
        
    if name in processed_files:
        return (f'{name} already processed. Exiting function.')
        
    else:
        print(f'Function executing now, {now}, for {name}.')
    
    if not bucket or not name: 
        raise ValueError(f'Eventarc did not send a valid cloud storage bucket or file. Cloud event arguments parsed were {(bucket, name)}')
        
    global last_updated, trades_df
    fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
    
    #TODO: simplify the if-else logic in the following blocks 
    
    #if it is a new day or cloud function just initialized, note down the current date and reset the trade dataframe
    if not last_updated or last_updated < now.date(): 
        if not last_updated:
            print('No last update time, cloud function has just been initialized.')
        else:
            if last_updated < now.date():
                print(f'Trades_df last updated {last_updated}, current date is {datetime.datetime.now(tz).date()}. Refreshing trades_df.')
        reset_cloud_function_state(now)
        print('Cloud function state reset')
       
    #Only run function from 1 hour before market open till 1 hour after market close
    if (now.hour <=8 and now.minute <= 30):
        print('Before market open, function will not run.')
        return name
    elif now.hour >= 22:
        print('After market close, function will not run.')    
        return name
    #If trades_df is empty, it is either the start of the data or the cloud function instance was just initialized
    #If cloud function just initialized and it is during market open hours, then we check for a checkpoint file first
    #If that checkpoint file is not available, then we can safely start trades_df from scratch
    else: 
        trade_log_path = f"gs://biases/trade_log_{now.strftime('%Y-%m-%d')}.pkl"
        if not len(trades_df):            
            try:
                print('Loading checkpoint')
                trades_df = load_from_cloud_storage(trade_log_path, fs)
                print(f"Successfully loaded checkpoint {trade_log_path} from cloud storage, with {len(trades_df)} rows, from {trades_df['published_datetime'].min()} to {trades_df['published_datetime'].max()}")
                
            except Exception as e:
                print(f'Cannot load {trade_log_path} from cloud storage, resetting trades_df. Exception: {e}')
                reset_cloud_function_state(now)

    print('Beginning to process and price trade messages.')
    N = len(trades_df)
    update_intraday_cusips(*get_trade_messages(os.path.join(bucket,name), fs), trades_df)
    print(f'Trades processed, trades_df rows changed by {len(trades_df) - N}.')
    if len(trades_df):    
        upload_to_cloud_storage(trade_log_path, trades_df, fs)
        # bias = simulate_weighted_average(trades_df, 
        #                                  weighting_col = 'error', 
        #                                  error_col = 'error', 
        #                                  groupby_cols=['trade_date'], 
        #                                  window_size=2000, 
        #                                  weighting_method='simple_average', 
        #                                  mask_large=0.3)
        # bias = bias_warm_start(bias, 500)
        # bias_path = f"gs://biases/bias_{now.strftime('%Y-%m-%d')}.pkl"
        # bias_path_demo = f"gs://biases/bias_for_demo.pkl"
        # upload_to_cloud_storage(bias_path, bias, fs)
        # upload_to_cloud_storage(bias_path_demo, bias, fs)
        processed_files.add(name)
        # print(f'Biases updated with trades from {name} to {bias_path}.') 
        
    print(f'Entire function took {time.time() - start}')
    return 'SUCCESS'


def reset_cloud_function_state(now):
    '''Resets the state of the cloud function and sets last_updated to the current date'''
    
    global last_updated, trades_df, processed_files
    print('Resetting cloud function state') #DEBUG
    
    last_updated = now.date()
    trades_df = pd.DataFrame(columns = trades_df_cols)
    processed_files = set()
