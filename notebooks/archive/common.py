import pandas as pd
from enum import Enum
from dateutil.relativedelta import relativedelta

MUNI_SECURITY_TYPE_DICT = {0:'Unknown',
                           1:'Special assessment',
                           2:'Double barreled',
                           3:'Lease-rental',
                           4:'Fuel/vehicle tax',
                           5:'Unlimited G.O.',
                           6:'Limited G.O.',
                           7:'Other',
                           8:'Revenue',
                           9:'Sales/excise tax',
                           10:'Tax allocation/increment',
                           11:'U.S. government-backed',
                           12:'Tobacco state appropriated',
                           13:'Tobacco settlement non-appropriated',
                           14:'Federal grant',
                           15:'Cigarette tax',
                           16:'Hotel tax',
                           17:'Miscellaneous tax',
                           18:'Personal income tax',
                           19:'Pilot'}


COUPON_TYPE_DICT = {0:'Unknown',
                    1:'Short term discount',
                    2:'Fixed rate - unconfirmed',
                    3:'Adjustable rate',
                    4:'Zero coupon',
                    5:'Floating rate',
                    6:'Index Linked',
                    7:'Stepped coupon',
                    8:'Fixed rate',
                    9:'Stripped convertible',
                    10:'Deferred interest',
                    11:'Floating rate @ floor',
                    12:'Stripped tax credit',
                    13:'Inverse floating',
                    14:'Stripped coupon principal',
                    15:'Linked inverse floater',
                    16:'Flexible rate',
                    17:'Original issue discount',
                    18:'Stripped principal',
                    19:'Reserve CUSIP',
                    20:'Variable rate',
                    21:'Stripped coupon',
                    22:'Floating auction rate',
                    23:'Tax credit',
                    24:'Tax credit OID',
                    25:'Stripped coupon payment',
                    26:'Stepped up stepped down',
                    27:'Credit Sensitive',
                    28:'Pay in kind',
                    29:'Range',
                    30:'Digital',
                    31:'Reset'}



COUPON_FREQUENCY_DICT = {0:None,
                         1:2,
                         2:12,
                         3:1,
                         4:52,
                         5:4,
                         6:0.5,
                         7:0.33333,
                         8:0.25,
                         9:0.2,
                         10:1/7,
                         11:1/8,
                         13:44,
                         14:360,
                         16:0,
                         23:None}


class REDEMPTION_TYPE(Enum):
    MATURITY = 1
    PAR_CALLABLE = 2
    NEXT_CALLABLE = 3
    CALLED = 4

# Assumed Redemption Date definition
class ARD:
    def __init__(self, estimated_date, actual_date, redemption_type):
        self.estimated_date = estimated_date
        self.actual_date = actual_date
        self.redemption_type = redemption_type
        self.additional = []

    def __str__(self):
        return str(self.estimated_date) + " " + str(self.actual_date) + " " + str(self.redemption_type) + " " + str(self.additional)

    def add(self,ard):
        self.additional.append(ard)

    def __repr__(self):
      return str(self)



# We determine if the trade price is equal to the last discounted cash flow
def is_one_coupon_or_less_till_redemption(settlement_date,start_date,dollar_price,
                                            interest_payment_frequency,yld,coupon_rate,redemption_value):
    if pd.isnull(redemption_value) or interest_payment_frequency == 0:
        return False
    
    A = diff_in_days(settlement_date,start_date)
    if A < 0:
        A = 0
    B = 360
    P = round(dollar_price,3)
    M = interest_payment_frequency
    Y = yld/100
    R = coupon_rate/100
    E = B/M
    
    last_discounted_cash_flow = 100*((redemption_value/100 + R/M)/(1 + ((E-A)*(Y/M)/E)) - A*R/B)
    last_discounted_cash_flow = round(last_discounted_cash_flow,3)
    
    if (last_discounted_cash_flow == P):
        return True
    return False

# These transformations do not include selection. i.e. the number of output examples
# is the same as the number of input examples
def transform_trade_data(df):
    df['muni_security_type']= df['muni_security_type'].map(MUNI_SECURITY_TYPE_DICT)
    df['coupon_type']= df['coupon_type'].map(COUPON_TYPE_DICT) 
    df['interest_payment_frequency']= df['interest_payment_frequency'].map(COUPON_FREQUENCY_DICT)
    
    df['coupon_rate'] = df['coupon_rate'].astype(float)
    df['next_call_price'] = df['next_call_price'].astype(float)
    transform_to_datetime(df, ['settlement_date', 'first_coupon_date', 
                                'previous_coupon_payment_date', 'next_coupon_payment_date', 
                                'maturity_date', 'refund_date'])
    df['yld'] = df.apply(lambda x: getattr(x, 'yield'), axis=1)
    #df = df[df.apply(lambda x: x["coupon_type"] in ['Fixed rate', 'Original issue discount', 'Zero coupon',],axis=1)]

    
def get_trade_data(bqclient, trade_date):
    query = ''' SELECT
                IFNULL(settlement_date,
                assumed_settlement_date) AS settlement_date,
                trade_date,
                time_of_trade,
                cusip,
                dated_date,
                dollar_price,
                issue_price,
                coupon AS coupon_rate,
                interest_payment_frequency,
                next_call_date,
                par_call_date,
                next_call_price,
                par_call_price,
                basic_assumed_maturity_date,
                maturity_date AS maturity_date,
                previous_coupon_payment_date,
                next_coupon_payment_date,
                first_coupon_date,
                coupon_type,
                muni_security_type,
                called_redemption_type,
                refund_date,
                refund_price,
                is_callable,
                is_called,
                call_timing,
                yield
                FROM `eng-reactor-287421.primary_views.trade_history_with_reference_data`
                WHERE trade_date = \'''' + trade_date + '''\'
                      AND NOT default_indicator
                      AND maturity_date > DATE_ADD(IFNULL(settlement_date,assumed_settlement_date),INTERVAL 6 month)
                      AND next_call_date > DATE_ADD(IFNULL(settlement_date,assumed_settlement_date),INTERVAL 6 month)
             '''
    dataframe = bqclient.query(query).result().to_dataframe()
    return dataframe   

def diff_in_days(end_date,start_date, convention="360/30"):
    #See MSRB Rule 33-G for details
    Y2 = end_date.year
    Y1 = start_date.year
    M2 = end_date.month
    M1 = start_date.month
    D2 = end_date.day #(end_date - relativedelta(days=1)).day 
    D1 = start_date.day
    if convention == "360/30":
        D1 = min(D1, 30)
        if D1 == 30: D2 = min(D2,30)
        difference_in_days = (Y2 - Y1) * 360 + (M2 - M1) * 30 + (D2 - D1)
    else: 
        print("unknown convention", convention)
    return difference_in_days


def get_end_date(start_date, diff_days):
    num_months = diff_days//30
    num_days = diff_days%30
    d = start_date + relativedelta(months=num_months, days=num_days)
    return d

def actual_diff_in_days(end_date,start_date):
    end_date = datetime(end_date.year, end_date.month, end_date.day)
    start_date = datetime(start_date.year, start_date.month, start_date.day)
    return (end_date - start_date).days

def get_next_coupon_date(first_coupon_date,settlement_date,time_delta):
    date = first_coupon_date
    while date < settlement_date:
        date = date + time_delta
    return date

def get_previous_coupon_date(first_coupon_date,settlement_date,time_delta):
    date = first_coupon_date
    while date < settlement_date:
        date = date + time_delta
    return date - time_delta


def transform_to_datetime(df, fields):
    for f in fields:
        df[f] = pd.to_datetime(df[f])
            

def get_time_delta_from_interest_frequency(interest_payment_frequency):
    #raise an error if interest_payment_frequency does not divide into 12: 
    if interest_payment_frequency != 0: 
        #delta = 12/interest_payment_frequency 
        if interest_payment_frequency <= 12:
            delta = 12/interest_payment_frequency
            time_delta = relativedelta(months=delta)
        elif interest_payment_frequency > 12 and interest_payment_frequency <= 52:
            delta = 52/interest_payment_frequency
            time_delta = relativedelta(weeks=delta)
    else: 
        time_delta = 0
    return time_delta
