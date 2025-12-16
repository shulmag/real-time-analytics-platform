'''
 # given the yield.
 '''
import pandas as pd

from modules.ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from modules.ficc.utils.auxiliary_functions import compare_dates
from modules.ficc.utils.diff_in_days import diff_in_days_two_dates
from modules.ficc.utils.truncation import trunc_and_round_price
from modules.ficc.pricing.auxiliary_functions import compute_value, \
                                                     get_num_of_interest_payments_and_final_coupon_date, \
                                                     price_of_bond_with_multiple_periodic_interest_payments, \
                                                     price_of_bond_with_interest_at_maturity


def get_price(cusip, 
              prev_coupon_date, 
              first_coupon_date, 
              next_coupon_date, 
              end_date, 
              settlement_date, 
              accrual_date, 
              frequency, 
              yield_rate, 
              coupon, 
              RV, 
              time_delta, 
              last_period_accrues_from_date):
    '''This function is a helper function for `compute_price`. This function calculates the price of a trade, where `yield_rate` 
    is a specific yield and `end_date` is a fixed repayment date. All dates must be valid relative to the settlement 
    date, as opposed to the trade date. Note that 'yield' is a reserved word in Python and should not be used as the name 
    of a variable or column.

    Formulas are from https://www.msrb.org/pdf.aspx?url=https%3A%2F%2Fwww.msrb.org%2FRules-and-Interpretations%2FMSRB-Rules%2FGeneral%2FRule-G-33.aspx.
    For all bonds, `base` is the present value of future cashflows to the buyer. 
    The clean price is this price minus the accumulated amount of simple interest that the buyer must pay to the seller, which is called `accrued`.
    Zero-coupon bonds are handled first. For these, the yield is assumed to be compounded semi-annually, i.e., once every six months.
    For bonds with non-zero coupon, the first and last interest payment periods may have a non-standard length,
    so they must be handled separately.

    When referring to the formulas in the MSRB handbook (link above), the below variables map to the code.
    A: prev_coupon_date_to_settlement_date
    B: NUM_OF_DAYS_IN_YEAR
    Y: yield_rate
    N: num_of_interest_payments
    E: num_of_days_in_period
    F: settlement_date_to_next_coupon_date
    P: price
    D: settlement_date_to_end_date
    H: prev_coupon_date_to_end_date
    R: coupon'''
    yield_rate = yield_rate / 100
    
    # Right now we do not disambiguate zero coupon from interest at maturity. More specfically, 
    # we should add logic that separates the cases of MSRB Rule Book G-33, rule (b) and rule (c)
    if frequency == 0:
        # See description for `price_of_bond_with_interest_at_maturity`
        price = price_of_bond_with_interest_at_maturity(cusip, 
                                                        settlement_date, 
                                                        accrual_date, 
                                                        end_date, 
                                                        yield_rate, 
                                                        coupon, 
                                                        RV)
    else:
        num_of_interest_payments, final_coupon_date = get_num_of_interest_payments_and_final_coupon_date(next_coupon_date, 
                                                                                                         end_date, 
                                                                                                         time_delta)
        prev_coupon_date_to_settlement_date = diff_in_days_two_dates(settlement_date, prev_coupon_date)
            
        num_of_days_in_period = NUM_OF_DAYS_IN_YEAR / frequency    # number of days in interest payment period 
        assert num_of_days_in_period == round(num_of_days_in_period)
         
        if compare_dates(end_date, next_coupon_date) <= 0:    # MSRB Rule Book G-33, rule (b)(i)(B)(1)
            if diff_in_days_two_dates(end_date, final_coupon_date) == 0:    # end date is the same (within one day) as the final coupon date, but the arithmetic later essentially includes one too many coupon payments; to fix this, one payment is subtracted from the redemption value (RV) and the final coupon date is shifted earlier
                final_coupon_date = prev_coupon_date
                RV = RV - coupon / frequency
            
            settlement_date_to_end_date = diff_in_days_two_dates(end_date, settlement_date)
            final_coupon_date_to_end_date = diff_in_days_two_dates(end_date, final_coupon_date)
            interest_due_at_end_date = coupon * final_coupon_date_to_end_date / NUM_OF_DAYS_IN_YEAR
            base = (RV + coupon / frequency + interest_due_at_end_date) / \
                   (1 + (yield_rate / frequency) * settlement_date_to_end_date / num_of_days_in_period)
            accrued = coupon * prev_coupon_date_to_settlement_date / NUM_OF_DAYS_IN_YEAR
            price = base - accrued
        else:    # MSRB Rule Book G-33, rule (b)(i)(B)(2)
            price = price_of_bond_with_multiple_periodic_interest_payments(cusip, 
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
    return trunc_and_round_price(price)


def compute_price(trade, yield_rate=None):
    '''This function computes the price of a trade. For bonds that have not been called, the price is the lowest of
    three present values: to the next call date (which may be above par), to the next par call date, and to maturity.'''
    if type(yield_rate) == str: raise ValueError('Yield rate argument cannot be a string. It must be a numerical value.')
    
    if yield_rate is None: yield_rate = trade.ficc_ytw
    return compute_value(trade, yield_rate, get_price)
