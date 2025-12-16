import datetime as dt

from pandas.tseries.holiday import get_calendar,AbstractHolidayCalendar, Holiday, HolidayCalendarFactory,  nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay

from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar

# import modules.ficc.utils.globals as globals

#This is for testing purposes comment when deploying
# import utils.globals as globals

# Juneteenth = Holiday(
#             'Juneteenth National Independence Day',
#             month=6,
#             day=19,
#             start_date='2021-06-18',
#             observance=nearest_workday,
#         )

cal = get_calendar('USFederalHolidayCalendar')  # Create calendar instance

TradingCalendar = HolidayCalendarFactory('TradingCalendar', cal, GoodFriday)

# class USTradingCalendar(AbstractHolidayCalendar):
#     rules = [
#         USFederalHolidayCalendar,
#         GoodFriday,
#         Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
#         USMartinLutherKingJr,
#         USPresidentsDay,
#         GoodFriday,
#         USMemorialDay,
#         Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
#         USLaborDay,
#         USThanksgivingDay,
#         Holiday('vDay', month=11, day=11),
#         Holiday('Christmas', month=12, day=25, observance=nearest_workday)
#         # Holiday('Columbus Day', month=10, day=1, offset=<DateOffset: kwds={'weekday': MO(+2))
#     ]


def get_trading_close_holidays(year):
    inst = TradingCalendar()
    
    return inst.holidays(dt.datetime(year-1, 12, 31), dt.datetime(year, 12, 31))

def get_day_before(trade_date):
    day_before = (trade_date - BDay(1)).date()  
    trading_close_holidays = [t.date() for t in get_trading_close_holidays(trade_date.year)]
    if day_before in trading_close_holidays:
        day_before = (day_before - BDay(1)).date()  
    
    # while day_before not in globals.treasury_rate.keys():
    #     print(f"Warning, missing value for tresury_rate on: {day_before}")
    #     day_before = (day_before - BDay(1)).date()

    return day_before