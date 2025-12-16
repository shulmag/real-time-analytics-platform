'''
 '''
def trunc(x, decimal_places):
    '''This file truncations an input to a specified number of decimal places.
    
    >>> trunc(3.33333, 3)
    3.333
    >>> trunc(3.99499, 3)
    3.994
    >>> trunc(30.99499, 3)
    30.994
    '''
    ten_places = 10 ** decimal_places
    return ((x * ten_places) // 1) / ten_places


def trunc_and_round_price(price):
    '''This function rounds the final price according to MSRB Rule Book G-33, rule (d).'''
    return trunc(price, 3)


def trunc_and_round_yield(yield_rate):
    '''This function rounds the final yield according to MSRB Rule Book G-33, rule (d).'''
    return round(trunc(yield_rate, 4), 3)
