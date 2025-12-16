import functions_framework
from auxiliary_functions import *
import datetime
import pytz
import os
import regex as re
from modules.ficc.utils.diff_in_days import *
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/jupyter/ficc/isaac_creds.json"

# INITIALIZE WHEN CLOUD FUNCTION INSTANCE BEGINS 
YYYYMMDD = "%Y-%m-%d"
last_updated = None
tz = pytz.timezone('US/Eastern')
cusip_ref_set = set()
cusip_ref_df = pd.DataFrame()
relevant_cols = ['cusip', 'next_call_date', 'maturity_date', 'refund_date', 'interest_payment_frequency', 'coupon_type']

weighting_col = 'error'
error_col = 'error'
window_size = 2000
weighting_method = 'simple_average'
mask_large = 35
N_warm = 500


def get_cusip_ref(cusip_list, redis_client):
    global cusip_ref_df, cusip_ref_set
    
    if not len(cusip_list):
        return
    
    cusip_ref = redis_client.mget(cusip_list)
    cusip_ref = [pickle.loads(x) for x in cusip_ref]
    
    if not len(cusip_ref_df):
        cusip_ref_df = pd.DataFrame(cusip_ref)[relevant_cols]
        
    else: 
        temp = pd.DataFrame(cusip_ref)[relevant_cols]
        cusip_ref_df = pd.concat([cusip_ref_df, temp])
        
    cusip_ref_set.update(cusip_list)

def get_days(row, col):
    if pd.isna(row[col]):
        return 0
    else: 
        diff = diff_in_days_two_dates(row[col], row.trade_datetime+pd.Timedelta(days=2))
        if diff <= 0:
            return -1
        else:
            return diff
    
@functions_framework.cloud_event    
def main(cloud_event):
    redis_client = redis.Redis(host='10.14.140.37', port=6379, db=0)
    fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
    
    start = time.time()
    now = datetime.datetime.now(tz)
    
    global last_updated, cusip_ref_set
    if not last_updated or last_updated < now.date():
        if not last_updated:
            print('Function just initialized. Resetting state')
        else: 
            print(f'Function last executed on {last_updated}, now it is {now.date()}. Resetting state.')
        last_updated = now.date()
        cusip_ref_set = set()
        print('Cloud function state reset')
        
        
    # for eventarc trigger
    bucket = cloud_event.data["bucket"]
    name = cloud_event.data["name"]
    if "trade_log" not in name: 
            return
    if now.strftime(YYYYMMDD) not in name:
        print(f'Error in filename; current date is {now.strftime(YYYYMMDD)} but trade_log is titled {name}')
        return

    with fs.open(f'gs://biases/{name}', 'rb') as f:
        trades_df = pickle.load(f)
    print(f'Function executing for {name}, with {len(trades_df)} trades')
        
    for col in ['yield', 'prediction', 'error']:
        trades_df[col] = trades_df[col]*100
        
    absent_cusips = []
    for cusip in trades_df.cusip.unique():
        if cusip not in cusip_ref_set:
            absent_cusips.append(cusip)
    print(f'Retrieving reference data for {len(absent_cusips)} cusips. Data for these cusips not previously loaded in memory.')
    redis_start = time.time()
    get_cusip_ref(absent_cusips, redis_client)
    print(f'Data retrieval took {time.time() - redis_start:.2f} seconds.')

    print('Calculating days to maturity, next call and refund and filtering trades.')
    trades_df = pd.merge(trades_df, cusip_ref_df, left_on='cusip', right_on='cusip')
    trades_df['days_to_call'] = trades_df[['trade_datetime','next_call_date']].apply(lambda x: get_days(x, 'next_call_date'), axis = 1)
    trades_df['days_to_maturity'] = trades_df[['trade_datetime','maturity_date']].apply(lambda x: get_days(x, 'maturity_date'), axis = 1)
    trades_df['days_to_refund'] = trades_df[['trade_datetime','refund_date']].apply(lambda x: get_days(x, 'refund_date'), axis = 1)
    
    data = trades_df.copy()
    N = len(data)
    data = data[(data.days_to_call == 0) | (data.days_to_call > 400)]
    data = data[(data.days_to_refund == 0) | (data.days_to_refund > 400)]
    data = data[(data.days_to_maturity == 0) | (data.days_to_maturity > 400)]
    data = data[data.days_to_maturity < 30000]
    data = data[data.par_traded >= 10000]
    data = data[data['yield'] >= 0]
    data = data[trades_df.interest_payment_frequency.isin([1,2,3,5,16])]
    data = data[trades_df.coupon_type.isin([8, 4, 10, 17])]
    # data['quantity'] = np.log10(data.par_traded)
    print(f'Filtering removed {N-len(data)} trades, down from {N} to {len(data)}.')
    
    print(f'Calculating biases. Parameters: Weight Column:{weighting_col}, Window Size: {window_size}, Weighting Method: {weighting_method}, Mask Large: {mask_large}, Warm Start: {N_warm}')
    bias = bias_warm_start(simulate_weighted_average(data, 
                                                     weighting_col = weighting_col, 
                                                     error_col = error_col, 
                                                     window_size=window_size, 
                                                     weighting_method=weighting_method, 
                                                     mask_large=mask_large),
                           N_warm)   
    data['bias'] = bias
    
    trades_df_final = pd.merge(trades_df, 
         data[['bias']], 
         left_index=True, 
         right_index=True,
                          how='outer')
    
    trades_df_final['bias'] = trades_df_final['bias'].ffill()
    
    print(f'Debiasing on filtered trades consistent with model training data (N={len(data)}):')
    original_MAE_filtered, corrected_MAE_filtered = debias_series(data.prediction, data['yield'], bias)
    print(f'Debiasing on all trades (N={len(trades_df_final)}):')
    original_MAE_all, corrected_MAE_all = debias_series(trades_df_final.prediction, trades_df_final['yield'], trades_df_final['bias'])
    
    bias_calculations_path = f"gs://biases/bias_calculations_{now.strftime(YYYYMMDD)}.pkl"
    bias_path_demo = f"gs://biases/bias_for_demo.pkl"
    upload_to_cloud_storage(bias_calculations_path, (data, trades_df_final), fs)
    upload_to_cloud_storage(bias_path_demo, bias, fs)
    
    uploadData(pd.DataFrame({'Timestamp':now.replace(second=0),
                            "MAE_all_trades" : original_MAE_all,
                            "MAE_new_ys_trades": original_MAE_filtered,
                            "MAE_all_trades_debiased": corrected_MAE_all,
                            "MAE_new_ys_trades_debiased":corrected_MAE_filtered}, index=[0]),
              "eng-reactor-287421.debiasing.MAE_tracking"
              )
    print(f'Biases updated with trades from {name}. Function took {time.time() - start:.2f} seconds.') 
