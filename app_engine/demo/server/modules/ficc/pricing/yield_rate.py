'''
 # given the price.
 '''
import pandas as pd
import scipy.optimize as optimize

from modules.ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from modules.ficc.utils.auxiliary_functions import compare_dates
from modules.ficc.utils.diff_in_days import diff_in_days_two_dates
from modules.ficc.utils.truncation import trunc_and_round_yield
from modules.ficc.pricing.auxiliary_functions import compute_value, \
                                                     get_num_of_interest_payments_and_final_coupon_date, \
                                                     price_of_bond_with_multiple_periodic_interest_payments, \
                                                     price_of_bond_with_interest_at_maturity


def get_yield(cusip, 
              prev_coupon_date, 
              first_coupon_date, 
              next_coupon_date, 
              end_date, 
              settlement_date, 
              accrual_date,
              frequency, 
              price, 
              coupon, 
              RV, 
              time_delta, 
              last_period_accrues_from_date):
    '''This function is a helper function for `compute_yield`. This function calculates the yield of a trade given the price and other trade features, using
    the MSRB Rule Book G-33 linked at: https://www.msrb.org/pdf.aspx?url=https%3A%2F%2Fwww.msrb.org%2FRules-and-Interpretations%2FMSRB-Rules%2FGeneral%2FRule-G-33.aspx.

    When referring to the formulas in the MSRB handbook (link above), the below variables map to the code.
    B: NUM_OF_DAYS_IN_YEAR
    E: number of days in interest payment period
    M: frequency
    N: num_of_interest_payments
    R: coupon'''
    settlement_date_to_end_date = diff_in_days_two_dates(end_date, settlement_date)    # hold period days

    if frequency == 0:
        # See description for `price_of_bond_with_interest_at_maturity`
        yield_func = lambda yield_rate: -price + price_of_bond_with_interest_at_maturity(cusip, 
                                                                                         settlement_date, 
                                                                                         accrual_date, 
                                                                                         end_date, 
                                                                                         yield_rate, 
                                                                                         coupon, 
                                                                                         RV)
        yield_estimate = optimize.newton(yield_func, x0=0, maxiter=int(1e3))
    else:
        num_of_interest_payments, final_coupon_date = get_num_of_interest_payments_and_final_coupon_date(next_coupon_date, 
                                                                                                         end_date, 
                                                                                                         time_delta)
        prev_coupon_date_to_settlement_date = diff_in_days_two_dates(settlement_date, prev_coupon_date)    # accrued days from beginning of the interest payment period, used to be labelled `A`
        prev_coupon_date_to_end_date = diff_in_days_two_dates(end_date, prev_coupon_date)    # accrual days for final paid coupon
        num_of_days_in_period = NUM_OF_DAYS_IN_YEAR / frequency    # number of days in interest payment period 
        assert num_of_days_in_period == round(num_of_days_in_period)

        if compare_dates(end_date, next_coupon_date) <= 0:
            # MSRB Rule Book G-33, rule (b)(ii)(B)(1)
            # Recall: number of interest payments per year * number of days in interest payment period = number of days in a year, i.e., E * M = NUM_OF_DAYS_IN_YEAR
            yield_estimate = (((RV + (coupon * prev_coupon_date_to_end_date / NUM_OF_DAYS_IN_YEAR)) / \
                            (price + (coupon * prev_coupon_date_to_settlement_date / NUM_OF_DAYS_IN_YEAR))) - 1) * \
                            (NUM_OF_DAYS_IN_YEAR / settlement_date_to_end_date)
        else:
            # MSRB Rule Book G-33, rule (b)(ii)(B)(2)
            yield_func = lambda yield_rate: -price + price_of_bond_with_multiple_periodic_interest_payments(cusip, 
                                                                                                   settlement_date, 
                                                                                                   accrual_date, 
                                                                                                   first_coupon_date, 
                                                                                                   prev_coupon_date, 
                                                                                                   next_coupon_date, 
                                                                                                   final_coupon_date, 
                                                                                                   end_date,  
                                                                                                   frequency,
                                                                                                   num_of_interest_payments, 
                                                                                                   yield_rate,
                                                                                                   coupon, 
                                                                                                   RV, 
                                                                                                   time_delta, 
                                                                                                   last_period_accrues_from_date)
            yield_estimate = optimize.newton(yield_func, x0=0, maxiter=int(1e3))
    return trunc_and_round_yield(yield_estimate * 100)


def compute_yield(trade, price=None):
    '''This function computes the yield of a trade.'''
    if type(price) == str: raise ValueError('Price argument cannot be a string. It must be a numerical value.')

    if price is None: price = trade.dollar_price
    return compute_value(trade, price, get_yield)
