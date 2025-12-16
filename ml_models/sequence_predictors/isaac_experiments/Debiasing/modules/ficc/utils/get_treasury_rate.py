'''
 '''

import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar
# from pandas.tseries.offsets import BDay

from modules.ficc.utils.auxiliary_functions import sqltodf
from modules.ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from modules.ficc.utils.diff_in_days import diff_in_days_two_dates
# from modules.ficc.utils.calendars import get_day_before
# import modules.ficc.utils.globals as globals

#Please comment before deploying
# from utils.auxiliary_functions import sqltodf
# from utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
# from utils.diff_in_days import diff_in_days_two_dates
# from utils.calendars import get_day_before
# import utils.globals as globals

def get_treasury_rate(client, treasury_rate_df=None):
    query = '''SELECT * FROM `eng-reactor-287421.treasury_yield.daily_yield_rate` order by Date desc'''
    treasury_rate_df = sqltodf(query, client)
    treasury_rate_df.set_index('Date', drop=True, inplace=True)
    treasury_rate_df = treasury_rate_df[~treasury_rate_df.index.duplicated(keep='first')]
    return treasury_rate_df


def get_all_treasury_rate(trade_date, treasury_rate_df):
    t_rate = treasury_rate_df.iloc[treasury_rate_df.index.get_loc(trade_date.values[0], method='backfill')]
    return list(t_rate.values)


def current_treasury_rate(trade, treasury_rate_df):
    treasury_maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30])
    if trade['last_calc_date'] is not None:
        time_to_maturity = diff_in_days_two_dates(trade['last_calc_date'],trade['settlement_date']) / NUM_OF_DAYS_IN_YEAR
    else:
        time_to_maturity = diff_in_days_two_dates(trade['maturity_date'],trade['settlement_date']) / NUM_OF_DAYS_IN_YEAR
    maturity = min(treasury_maturities, key=lambda x: abs(x - time_to_maturity))
    maturity = 'year_' + str(maturity)
    t_rate = treasury_rate_df.iloc[treasury_rate_df.index.get_loc(trade['trade_date'], method='backfill')][maturity]
    return t_rate