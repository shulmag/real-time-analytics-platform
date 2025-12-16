'''
 # The SQL arrays from BigQuery are converted to a dictionary when read as a pandas dataframe. 
 # 
 # A few blunt normalization are performed on the data. We will experiment with others as well. 
 # Multiplying the yield spreads by 100 to convert into basis points. 
 # Taking the log of the size of the trade to reduce the absolute scale 
 # Taking the log of the number of seconds between the historical trade and the latest trade
 '''
import numpy as np
from datetime import datetime

from modules.ficc.utils.nelson_siegel_model import yield_curve_level
from modules.ficc.utils.diff_in_days import diff_in_days_two_dates
from modules.ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from modules.ficc.utils.calendars import get_day_before

# Please comment before deploying 
# from utils.nelson_siegel_model import yield_curve_level
# from utils.diff_in_days import diff_in_days_two_dates
# from utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
# from utils.calendars import get_day_before

def trade_dict_to_list(trade_dict: dict, 
                       trade_datetime, 
                       treasury_rate_df) -> list:
                      
    trade_type_mapping = {'D':[0,0],'S': [0,1],'P': [1,0]}
    trade_list = []
    features = ['rtrs_control_number', 
                'par_traded',
                'trade_type',
                'trade_datetime',
                'settlement_date',
                'calc_date', 
                'maturity_date']
    for key in features:
        try:
            temp = trade_dict[key]
            if temp is None:
                print(f'{key} is missing, skipping this trade')
                return None, None
        except Exception as e:
            return None, None
    
    # We do not have weighted average maturity before August 4 for ficc yc
    if trade_dict['trade_datetime'] < datetime(2021,8,4):
        target_date = datetime(2021,8,4,15,57)
    else:
        target_date = trade_dict['trade_datetime']
 

    calc_date = trade_dict['calc_date']
    time_to_maturity = diff_in_days_two_dates(calc_date,target_date)/NUM_OF_DAYS_IN_YEAR
    #calculating the time to maturity in years from the trade_date
 
    yield_at_that_time = yield_curve_level(time_to_maturity,
                                           target_date.strftime('%Y-%m-%d:%H:%M'))

    if trade_dict['yield'] is not None and yield_at_that_time is not None:
        yield_spread = trade_dict['yield'] * 100 - yield_at_that_time
        trade_list.append(yield_spread)
    else:
        print('Yield is missing, skipping trade')
        return None, None
        
    
    ###### Adding the treasury spreads ######
    treasury_maturities = np.array([1,2,3,5,7,10,20,30])
    maturity = min(treasury_maturities, key=lambda x:abs(x-time_to_maturity))
    maturity = 'year_' + str(maturity)
    day_before = get_day_before(target_date)
    t_rate = treasury_rate_df.iloc[treasury_rate_df.index.get_loc(day_before, method='backfill')][maturity]
    t_spread = (trade_dict['yield'] - t_rate) * 100
    trade_list.append(np.round(t_spread,3))
    ####### treasury spread added #######

    ####### Adding par traded and trade type ######
    trade_list.append(np.float32(np.log10(trade_dict['par_traded'])))        
    trade_list += trade_type_mapping[trade_dict['trade_type']]
    ###############################################

    # For some trades the seconds ago feature is negative.
    # This is because the publish time is after the trade datetime.
    # We have verified that this is an anomaly on MSRBs end.
    seconds_ago = (trade_datetime - trade_dict['trade_datetime']).total_seconds()
    trade_list.append(np.log10(1 + seconds_ago))
    
    # trade_list.append(time_to_maturity * NUM_OF_DAYS_IN_YEAR)

    return np.stack(trade_list) , (yield_spread,
                                   yield_at_that_time,
                                   trade_dict['rtrs_control_number'],
                                   trade_dict['yield'],
                                   trade_dict['dollar_price'], 
                                   seconds_ago, 
                                   float(trade_dict['par_traded']),
                                   trade_dict['calc_date'], 
                                   trade_dict['maturity_date'], 
                                   trade_dict['next_call_date'], 
                                   trade_dict['par_call_date'], 
                                   trade_dict['refund_date'], 
                                   trade_dict['trade_datetime'], 
                                   trade_dict['calc_day_cat'], 
                                   trade_dict['settlement_date'], 
                                   trade_dict['trade_type'])