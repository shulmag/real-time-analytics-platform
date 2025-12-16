'''
'''
from dateutil.relativedelta import relativedelta

import pandas as pd

from auxiliary_variables import NUM_OF_DAYS_IN_YEAR, NUM_OF_WEEKS_IN_YEAR, NUM_OF_MONTHS_IN_YEAR, COUPON_FREQUENCY_TYPE, COUPON_FREQUENCY_DICT
from auxiliary_functions import compare_dates, dates_are_equal


def end_date_for_called_bond(trade):
    '''This function provides the end date for a called bond.'''
    if not pd.isnull(trade.refund_date): return trade.refund_date
    raise ValueError(f'Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) is called, but no refund date.')


def refund_price_for_called_bond(trade):
    '''This function provides the par value for a called bond.'''
    if not pd.isnull(trade.refund_price): return trade.refund_price
    raise ValueError(f'Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) is called, but no refund price.')


def get_frequency(identifier):
    '''This function returns the frequency of coupon payments based on 
    the interest payment frequency identifier in the bond reference data.'''
    if type(identifier) == str: return COUPON_FREQUENCY_TYPE[identifier]    # check whether the frequency dict has already been applied to the identifier
    return COUPON_FREQUENCY_TYPE[COUPON_FREQUENCY_DICT[identifier]]


def get_time_delta_from_interest_frequency(interest_payment_frequency):
    error_string = lambda num: f'The interest payment frequency of {interest_payment_frequency} is invalid, since it must divide {num}'

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


def get_next_coupon_date(first_coupon_date, start_date, time_delta):
    '''This function computes the next time a coupon is paid. Note that this function could return 
    a `next_coupon_date` that is after the end_date. This does not create a problem since we 
    deal with the final coupon separately in `price_of_bond_with_multiple_periodic_interest_payments`. 
    Note that it may be that this function is not necessary because the reference data field `next_coupon_date` 
    is never null when there is a 'next coupon date.' In the future, we should confirm whether this 
    is the case.'''
    date = first_coupon_date
    while compare_dates(date, start_date) < 0:
        date = date + time_delta
    return date
#     cannot use the below code since division is not valid between datetime.timedelta and relativedelta, and converting between types introduces potential for errors
#     num_of_time_periods = int(np.ceil((start_date - first_coupon_date) / time_delta))    # `int` wraps the `ceil` function because the `ceil` function returns a float
#     return first_coupon_date + time_delta * num_of_time_periods


def get_previous_coupon_date(first_coupon_date, start_date, accrual_date, time_delta, next_coupon_date=None):
    '''This function computes the previous time a coupon was paid for this bond 
    by relating it to the next coupon date.
    Note that it may be that this function is not necessary because the reference data field 
    `previous_coupon_date` is never null when `next_coupon_date` exists. In the 
    future, we should confirm whether this is the case.'''
    if next_coupon_date == None: next_coupon_date = get_next_coupon_date(first_coupon_date, start_date, time_delta)

    if dates_are_equal(next_coupon_date, first_coupon_date): return accrual_date
    return next_coupon_date - time_delta


def get_prev_coupon_date_and_next_coupon_date(trade, frequency, time_delta):
    '''This function is valid for bonds that don't pay coupons, whereas the previous 
    two functions assume the bond pays coupons.
    Note: the reference data field of `next_coupon_payment_date` corresponds to our variable of 
    `next_coupon_date` (removing the word `payment`) for more concise and readable 
    code, and similarly with `previous_coupon_date`.'''
    if frequency == 0:
        next_coupon_date = trade.maturity_date
        prev_coupon_date = trade.accrual_date
    else:
        if pd.isnull(trade.next_coupon_payment_date):
            next_coupon_date = get_next_coupon_date(trade.first_coupon_date, trade.settlement_date, time_delta)
        else:
            next_coupon_date = pd.to_datetime(trade.next_coupon_payment_date)

        if pd.isnull(trade.previous_coupon_payment_date):
            prev_coupon_date = get_previous_coupon_date(trade.first_coupon_date, trade.settlement_date, trade.accrual_date, time_delta, next_coupon_date)
        else:
            prev_coupon_date = pd.to_datetime(trade.previous_coupon_payment_date)

    return prev_coupon_date, next_coupon_date


def get_num_of_interest_payments_and_final_coupon_date(next_coupon_date, end_date, time_delta):
    '''This function returns the number of interest payments and the final coupon 
    date based on the next coupon date, the end date, and the gap between coupon 
    payments. This function returns both together because one is always a 
    byproduct of computing the other.
    Note that the special case of an odd final coupon is handled below in 
    `price_of_bond_with_multiple_periodic_interest_payments`.'''
    if compare_dates(next_coupon_date, end_date) > 0: return 0, next_coupon_date    # return 1, end_date (would be valid in isolation)
    
    num_of_interest_payments = 1
    final_coupon_date = next_coupon_date
    while compare_dates(final_coupon_date + time_delta, end_date) <= 0:
        num_of_interest_payments += 1
        final_coupon_date += time_delta
    return num_of_interest_payments, final_coupon_date
