'''
 # frequency information of a bond.
 '''
from dateutil.relativedelta import relativedelta

from modules.ficc.utils.auxiliary_variables import COUPON_FREQUENCY_DICT, COUPON_FREQUENCY_TYPE

# This is used to debug, please comment before deploying
# from utils.auxiliary_variables import COUPON_FREQUENCY_DICT, COUPON_FREQUENCY_TYPE

def get_frequency(identifier):
    '''This function returns the frequency of coupon payments based on 
    the interest payment frequency identifier in the bond reference data.'''
    if type(identifier) != str: identifier = COUPON_FREQUENCY_DICT[identifier]    # check whether the frequency dict has already been applied to the identifier
    return COUPON_FREQUENCY_TYPE[identifier]


def get_time_delta_from_interest_frequency(interest_payment_frequency):
    '''This function returns a time delta object based on the interest payment frequency. 
    The first step is to identify whether the interest payment frequency passed in is in 
    terms of the number of months in a year or the number of weeks in a year. Then, based 
    on this the time delta object is returned.'''
    error_string = lambda num: f'The interest payment frequency of {interest_payment_frequency} is invalid, since it must divide {num}'

    NUM_OF_MONTHS_IN_YEAR = 12
    NUM_OF_WEEKS_IN_YEAR = 52
    NUM_OF_DAYS_IN_YEAR = 360

    time_delta = 0
    if interest_payment_frequency != 0:
        if interest_payment_frequency <= 1:
            delta = 1 / interest_payment_frequency
            time_delta = relativedelta(years=delta)
        elif interest_payment_frequency > 1 and interest_payment_frequency <= NUM_OF_MONTHS_IN_YEAR:
            if NUM_OF_MONTHS_IN_YEAR % interest_payment_frequency != 0:
                raise ValueError(error_string(NUM_OF_MONTHS_IN_YEAR))
            delta = NUM_OF_MONTHS_IN_YEAR / interest_payment_frequency
            time_delta = relativedelta(months=delta)
        elif interest_payment_frequency > NUM_OF_MONTHS_IN_YEAR and interest_payment_frequency <= NUM_OF_WEEKS_IN_YEAR:
            if NUM_OF_WEEKS_IN_YEAR % interest_payment_frequency != 0:
                raise ValueError(error_string(NUM_OF_WEEKS_IN_YEAR))
            delta = NUM_OF_WEEKS_IN_YEAR / interest_payment_frequency
            time_delta = relativedelta(weeks=delta)
        elif interest_payment_frequency > NUM_OF_WEEKS_IN_YEAR and interest_payment_frequency <= NUM_OF_DAYS_IN_YEAR:
            if NUM_OF_DAYS_IN_YEAR % interest_payment_frequency != 0:
                raise ValueError(error_string(NUM_OF_DAYS_IN_YEAR))
            delta = NUM_OF_DAYS_IN_YEAR / interest_payment_frequency
            time_delta = relativedelta(days=delta)
    return time_delta