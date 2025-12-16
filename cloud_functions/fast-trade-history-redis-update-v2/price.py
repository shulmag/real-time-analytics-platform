'''
'''
import pandas as pd

from auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from auxiliary_functions import diff_in_days_two_dates, compare_dates, dates_are_equal, trunc_and_round_price
from bond_characteristics import get_num_of_interest_payments_and_final_coupon_date, get_time_delta_from_interest_frequency, end_date_for_called_bond, get_prev_coupon_date_and_next_coupon_date, refund_price_for_called_bond


def price_of_bond_with_interest_at_maturity(cusip,    # can be used for debugging purposes
                                            settlement_date, 
                                            accrual_date, 
                                            end_date, 
                                            yield_rate, 
                                            coupon, 
                                            RV):
    '''This function is called when the interest is only paid at maturity (which is represented 
    in the transformed dataframe as interest payment frequency equaling 0). There are two 
    cases when interest is paid at maturity. The first case is for short term bonds where 
    there is a single coupon payment at maturity, and this logic will reduce to the logic 
    in MSRB Rule Book G-33, rule (b)(i)(A). The second case is when when there is a compounding 
    accreted value (i.e., capital appreciation bonds) which accrues semianually. Then, to get 
    the price of this bond, we need to account for the accrued interest. This can be thought 
    of as a bond that pays a coupon semiannually through the duration of the bond, but all the 
    coupon payments are made as a single payment at the time the bond is called / maturity. 
    For more info and an example, see the link: https://www.investopedia.com/terms/c/cav.asp#:~:text=Compound%20accreted%20value%20(CAV)%20is,useful%20metric%20for%20bond%20investors.'''
    NOMINAL_FREQUENCY = 2    # semiannual interest payment frequency
    accrual_date_to_settlement_date = diff_in_days_two_dates(settlement_date, accrual_date)
    settlement_date_to_end_date = diff_in_days_two_dates(end_date, settlement_date)
    accrued = coupon * accrual_date_to_settlement_date / NUM_OF_DAYS_IN_YEAR
    num_of_periods_from_settlement_date_to_end_date = settlement_date_to_end_date / (NUM_OF_DAYS_IN_YEAR / NOMINAL_FREQUENCY)
    denom = (1 + yield_rate / NOMINAL_FREQUENCY) ** num_of_periods_from_settlement_date_to_end_date
    accrual_date_to_end_date = diff_in_days_two_dates(end_date, accrual_date)
    base = (RV + coupon * accrual_date_to_end_date / NUM_OF_DAYS_IN_YEAR) / denom
    return base - accrued


def price_of_bond_with_multiple_periodic_interest_payments(cusip,    # can be used for debugging purposes
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
                                                           last_period_accrues_from_date):
    '''This function computes the price of a bond with multiple periodic interest 
    payments using MSRB Rule Book G-33, rule (b)(i)(B)(2). Comments with capital 
    letter symbols represent those same symbols seen in formula in MSRB rule book.'''
    num_of_days_in_period = NUM_OF_DAYS_IN_YEAR / frequency
    discount_rate = 1 + yield_rate / frequency    # 1 + Y / M
    final_coupon_date_to_end_date = diff_in_days_two_dates(end_date, final_coupon_date)
    prev_coupon_date_to_settlement_date = diff_in_days_two_dates(settlement_date, prev_coupon_date)    # A
    interest_due_at_end_date = coupon * final_coupon_date_to_end_date / NUM_OF_DAYS_IN_YEAR
    
    RV_and_interest_due_at_end_date = RV + interest_due_at_end_date
    settlement_date_to_next_coupon_date = diff_in_days_two_dates(next_coupon_date, settlement_date)    # E - A
    settlement_date_to_next_coupon_date_frac = settlement_date_to_next_coupon_date / num_of_days_in_period    # (E - A) / E
    final_coupon_date_to_end_date_frac = final_coupon_date_to_end_date / num_of_days_in_period
    num_of_periods_from_settlement_date_to_end_date = num_of_interest_payments - 1 + settlement_date_to_next_coupon_date_frac + final_coupon_date_to_end_date_frac
    
    RV_and_interest_due_at_end_date_discounted = RV_and_interest_due_at_end_date / (discount_rate ** num_of_periods_from_settlement_date_to_end_date)
    
    # The following logic statements are necessary to address odd first and final coupons
    if dates_are_equal(next_coupon_date, first_coupon_date):
        num_of_days_in_current_interest_payment_period = diff_in_days_two_dates(first_coupon_date, accrual_date)
    elif pd.notna(last_period_accrues_from_date) and compare_dates(settlement_date, last_period_accrues_from_date + time_delta) > 0:    # this logic has not been tested
        num_of_days_in_current_interest_payment_period = 0
    else:
        num_of_days_in_current_interest_payment_period = num_of_days_in_period

    coupon_payments_discounted_total = (coupon * num_of_days_in_current_interest_payment_period / NUM_OF_DAYS_IN_YEAR) / \
                                       (discount_rate ** settlement_date_to_next_coupon_date_frac)
    coupon_payment = coupon / frequency
    for k in range(1, num_of_interest_payments):
        coupon_payment_discounted = coupon_payment / (discount_rate ** (settlement_date_to_next_coupon_date_frac + k))
        coupon_payments_discounted_total += coupon_payment_discounted
        
    accrued = coupon * prev_coupon_date_to_settlement_date / NUM_OF_DAYS_IN_YEAR    # R * A / B
    return RV_and_interest_due_at_end_date_discounted + coupon_payments_discounted_total - accrued


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
         
        if compare_dates(end_date, next_coupon_date) <= 0:
            # MSRB Rule Book G-33, rule (b)(i)(B)(1)
            settlement_date_to_end_date = diff_in_days_two_dates(end_date, settlement_date)
            final_coupon_date_to_end_date = diff_in_days_two_dates(end_date, final_coupon_date)
            interest_due_at_end_date = coupon * final_coupon_date_to_end_date / NUM_OF_DAYS_IN_YEAR
            base = (RV + coupon / frequency + interest_due_at_end_date) / \
                   (1 + (yield_rate / frequency) * settlement_date_to_end_date / num_of_days_in_period)
            accrued = coupon * prev_coupon_date_to_settlement_date / NUM_OF_DAYS_IN_YEAR
            price = base - accrued
        else:
            # MSRB Rule Book G-33, rule (b)(i)(B)(2)
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
    old_date = pd.to_datetime('2000-01-01').date()     # NOTE: change this error value because the yield to worst below will always be to this date
    return_values_if_missing_data = -100, old_date, -100, -100, -100, -1
    
    # below code is commented out in `server/modules/ficc/pricing/price.py` because this situation is handled in `get_data_from_redis(...)` in `server/modules/finance.py`
    if trade.interest_payment_frequency != 0 and pd.isna(trade.first_coupon_date):    # checks if data is faulty
        print(f'Trade (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) has a coupon but no first coupon date.')    # printing instead of raising an error to not disrupt processing large quantities of trades
        return return_values_if_missing_data
    
    if yield_rate == None:
        yield_rate = trade['yield']
    elif isinstance(yield_rate, str):
        print('Yield rate argument cannot be a string. It must be a numerical value.')    # raise ValueError('Yield rate argument cannot be a string. It must be a numerical value.')
        return return_values_if_missing_data
    
    variables_to_null_check = [(yield_rate, 'yield'),
                               (trade.coupon, 'coupon'), 
                               (trade.maturity_date, 'maturity date')]
    for value, name in variables_to_null_check:
        if pd.isna(value):
            print(f'Trade (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) has a {name} value of {value} which should not be null.')    # printing instead of raising an error to not disrupt processing large quantities of trades
            return return_values_if_missing_data

    try:
        frequency = trade.interest_payment_frequency
        time_delta = get_time_delta_from_interest_frequency(frequency)
        my_prev_coupon_date, my_next_coupon_date = get_prev_coupon_date_and_next_coupon_date(trade, frequency, time_delta)

        get_price_caller = lambda end_date, redemption_value: get_price(trade.cusip, 
                                                                        my_prev_coupon_date, 
                                                                        trade.first_coupon_date, 
                                                                        my_next_coupon_date, 
                                                                        end_date, 
                                                                        trade.settlement_date, 
                                                                        trade.accrual_date, 
                                                                        frequency, 
                                                                        yield_rate, 
                                                                        trade.coupon, 
                                                                        redemption_value, 
                                                                        time_delta, 
                                                                        trade.last_period_accrues_from_date)

        redemption_value_at_maturity = 100
        if (not trade.is_called) and (not trade.is_callable):
            yield_to_maturity = get_price_caller(trade.maturity_date, redemption_value_at_maturity)
            return yield_to_maturity, trade.maturity_date, 0, 0, 0, 2
        elif trade.is_called:
            end_date = end_date_for_called_bond(trade)

            if compare_dates(end_date, trade.settlement_date) < 0:
                print(f'Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) has an end date ({end_date}) which is after the settlement date ({trade.settlement_date}).')    # printing instead of raising an error to not disrupt processing large quantities of trades
                # raise ValueError(f'Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) has an end date ({end_date}) which is after the settlement date ({trade.settlement_date}).')
            
            redemption_value_at_refund = refund_price_for_called_bond(trade)
            return get_price_caller(end_date, redemption_value_at_refund), end_date, 0, 0, 0, 3
        else:
            next_price, to_par_price, maturity_price = float('inf'), float('inf'), float('inf')

            if pd.notna(trade.par_call_date) and pd.notna(trade.par_call_price):
                to_par_price = get_price_caller(trade.par_call_date, trade.par_call_price)
                if pd.isna(to_par_price): to_par_price = float('inf')    # necessary conversion because null values silently fail in a `min` or `max` function by returning null (`min` function is used downstream)
            
            if pd.notna(trade.next_call_date) and pd.notna(trade.next_call_price):
                next_price = get_price_caller(trade.next_call_date, trade.next_call_price)
                if pd.isna(next_price): next_price = float('inf')    # necessary conversion because null values silently fail in a `min` or `max` function by returning null (`min` function is used downstream)
            
            maturity_price = get_price_caller(trade.maturity_date, redemption_value_at_maturity)
            if pd.isna(maturity_price):    # `maturity_date` null check already done upstream
                print(f'Trade (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) has a maturity_price value of {maturity_price}, which should not be null, after computing it based on an end date of {trade.maturity_date} and redemption value of {redemption_value_at_maturity}.')    # printing instead of raising an error to not disrupt processing large quantities of trades
                if next_price == float('inf') and to_par_price == float('inf'):    # returning missing values only in this case so that if one these values do exist, the price can be returned even if the maturity price is missing
                    return return_values_if_missing_data
                else:
                    maturity_price = float('inf')

            prices_and_dates = [(next_price, trade.next_call_date), 
                                (to_par_price, trade.par_call_date), 
                                (maturity_price, trade.maturity_date)]
            calc_price, calc_date = min(prices_and_dates, key=lambda pair: pair[0]) # this function is stable and will choose the pair which appears first in the case of ties for the lowest price
        
        if calc_date == trade.next_call_date: calc_day_cat = 0
        elif calc_date == trade.par_call_date: calc_day_cat = 1
        elif calc_date == trade.maturity_date: calc_day_cat = 2
        elif calc_date == trade.refund_date: calc_day_cat = 3
        else: calc_day_cat = 4
        return calc_price, calc_date, next_price, to_par_price, maturity_price, calc_day_cat
    except Exception as e:
        print(f'{type(e)}: {e} when computing price for trade (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}).')
        return return_values_if_missing_data
