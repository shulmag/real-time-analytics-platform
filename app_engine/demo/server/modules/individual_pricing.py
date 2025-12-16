'''
Description: Functions that support individual pricing, i.e., pricing a single CUSIP.
'''
from typing import Union     # importing types for hinting
from datetime import datetime
import numpy as np
import pandas as pd
from flask import jsonify, make_response

from modules.ficc.utils.diff_in_days import diff_in_days_two_dates    # , diff_in_days


from modules.errors import CustomMessageError, DEFAULT_PRICING_ERROR_MESSAGE

from modules.auxiliary_variables import PRICE_BOTH_DIRECTIONS_TO_CORRECT_INVERSION, \
                                        YEAR_MONTH_DAY, \
                                        YEAR_MONTH_DAY_HOUR_MIN_SEC, \
                                        DISPLAY_PRECISION, \
                                        DOLLAR_PRICE_MODEL_DISPLAY_TEXT, \
                                        MONTH_DAY_YEAR, \
                                        REDEMPTION_VALUE_AT_MATURITY, \
                                        HOUR_MIN_SEC
from modules.auxiliary_functions import get_current_datetime, datetime_as_string, get_settlement_date, get_outstanding_amount, round_for_logging
from modules.logging_functions import log_usage
from modules.data_preparation_for_pricing import pre_processing, get_inputs_for_nn, reverse_direction_concat, process_data_for_pricing, get_data_from_redis
from modules.pricing_functions import predict_spread, predict_dollar_price, get_trade_price_from_yield_spread_model
from modules.exclusions import CUSIP_ERROR_MESSAGE


def _get_spread_or_dollar_price(df: pd.DataFrame, spread_or_dollar_price='spread') -> Union[float, tuple[float, float, pd.DataFrame]]:
    '''Gets the prediction for a single CUSIP. If the `trade_type` is dealer-dealer, then one prediction 
    is returned. Otherwise, both sides are priced and returned along with the modified dataframe, which 
    is required for the different trade history derived features.'''
    assert spread_or_dollar_price in ('spread', 'dollar_price')
    using_dollar_price_model = spread_or_dollar_price == 'dollar_price'
    predict_func = predict_spread if spread_or_dollar_price == 'spread' else predict_dollar_price
    trade_type = df['trade_type'].iloc[0]
    if not PRICE_BOTH_DIRECTIONS_TO_CORRECT_INVERSION or trade_type == 'D':
        df = pre_processing(df)
        input_dict = get_inputs_for_nn(df, using_dollar_price_model)
        spread_or_dollar_price = predict_func(input_dict)[0][0]
        return spread_or_dollar_price
    else:
        # creating a DataFrame with two rows, one for 'S' and one for 'P'
        df = reverse_direction_concat(df)    # ensures that the order is 'S' and then 'P'
        df = pre_processing(df)   
        input_dicts = get_inputs_for_nn(df, using_dollar_price_model)

        # send `input_dict`s to the model
        spreads_or_dollar_prices = predict_func(input_dicts)
        spread_or_dollar_price_S, spread_or_dollar_price_P = spreads_or_dollar_prices[0][0], spreads_or_dollar_prices[1][0]
        return spread_or_dollar_price_S, spread_or_dollar_price_P, df


def _get_spread(df: pd.DataFrame) -> Union[float, tuple[float, float, pd.DataFrame]]:
    '''Gets the predicted spread for a single cusip.'''
    return _get_spread_or_dollar_price(df, 'spread')


def _get_dollar_price(df: pd.DataFrame) -> Union[float, tuple[float, float, pd.DataFrame]]:
    '''Gets the predicted dollar price for a single cusip.'''
    return _get_spread_or_dollar_price(df, 'dollar_price')


def get_data(cusip, quantity, current_date, current_datetime, settlement_date, trade_type):
    '''This function retrives reference data from our managed Redis instance. Currently,
    the key for this Redis instance is simply a CUSIP, and the reference data retrieved 
    is the most recent reference data for that CUSIP. In future, this will be CUSIP and
    timestamp.'''
    cusip_can_be_priced_df, cusip_cannot_be_priced_df = get_data_from_redis(cusip, current_date)
    if len(cusip_cannot_be_priced_df) != 0:
        error_key = cusip_cannot_be_priced_df['message'].values[0]
        raise CustomMessageError(CUSIP_ERROR_MESSAGE[error_key])
    
    maturity_date_or_refund_date = cusip_can_be_priced_df['maturity_date'].values[0]
    if cusip_can_be_priced_df['is_called'].values[0] is True:
        maturity_date_or_refund_date = cusip_can_be_priced_df['refund_date'].values[0]
    if diff_in_days_two_dates(maturity_date_or_refund_date, settlement_date, convention='exact') <= 0:
        # raise CustomMessageError(f'This cusip matures on {datetime_as_string(maturity_date, precision="day")}')
        raise CustomMessageError(CUSIP_ERROR_MESSAGE['maturing_soon'])    # the above error message provides more detail but this keeps it consistent with that in batch pricing
    
    outstanding_amount = get_outstanding_amount(cusip_can_be_priced_df)
    if not pd.isna(outstanding_amount) and outstanding_amount > 0 and outstanding_amount < quantity:
        raise CustomMessageError(CUSIP_ERROR_MESSAGE['quantity_greater_than_outstanding_amount'](outstanding_amount))
    
    cusip_can_be_priced_df = process_data_for_pricing(cusip_can_be_priced_df, quantity, trade_type, current_date, settlement_date, current_datetime)
    return cusip_can_be_priced_df


def check_inversion_and_flip_prices(dealer_sell, dealer_buy, sell_calc_date=None, buy_calc_date=None, ys_dealer_sell=None, ys_dealer_purchase=None):
    '''Invert dealer buy and dealer sell prices and calc dates for a single trade if sell price is less than 
    buy price. Calc date defaults to zero in the case of dollar price predictions.'''
    print(f'dealer_sell {dealer_sell}, dealer_buy {dealer_buy}')
    if dealer_buy > dealer_sell:
        print('dealer buy more than dealer sell, inverting prices')
        dealer_sell, dealer_buy = dealer_buy, dealer_sell  # Invert prices
        if sell_calc_date and buy_calc_date:
            sell_calc_date, buy_calc_date = buy_calc_date, sell_calc_date
        if ys_dealer_sell and ys_dealer_purchase:
            ys_dealer_sell, ys_dealer_purchase = ys_dealer_purchase, ys_dealer_sell
    
    if sell_calc_date and buy_calc_date and ys_dealer_sell and ys_dealer_purchase:
        return dealer_sell, dealer_buy, sell_calc_date, buy_calc_date, ys_dealer_sell, ys_dealer_purchase
    else: 
        return dealer_sell, dealer_buy


def get_prediction_from_individual_pricing(cusip, quantity, trade_type, user, api_call):
    '''This function takes aspects of a hypothetical trade for a particular CUSIP
    and returns estimations of price and Yield to Worst.'''
    try:    # wrap in try except to perform logging even when there is an error
        current_datetime = get_current_datetime()
        current_date = datetime_as_string(current_datetime, precision='day')
        current_datetime = datetime_as_string(current_datetime)    # current_datetime is now a string; making it in precision minutes since this is what get_yield_curve requires

        settlement_date = get_settlement_date(current_date)
        quantity = int(quantity) * 1000    # wrap this in int since `quantity` is passed in as a string, but for this function, we need `quantity` to be a numerical value

        # FIXME: remove calling `process_data` inside `get_data(...)` and instead call `get_ytw_dollar_price_for_list(...)` inside this function
        # Passing the date for the trade
        df = get_data(cusip, 
                      quantity, 
                      pd.to_datetime(current_date, format=YEAR_MONTH_DAY),
                      pd.to_datetime(current_datetime, format=YEAR_MONTH_DAY_HOUR_MIN_SEC),
                      settlement_date,
                      trade_type)
        row = df.iloc[0]

        # Changed yield_curve_level to ficc_ycl. This now comes from the data package
        yield_curve_level = row.get('ficc_ycl') / 100    # extract the yield_curve_level from the dataframe, and since the dataframe only has one item, so we are isolating the value by doing .iloc[0]

        if type(df) == str: return df
        model_used = row.get('model_used')

        if model_used == 'yield_spread':
            if not PRICE_BOTH_DIRECTIONS_TO_CORRECT_INVERSION or trade_type == 'D':
                ys = _get_spread(df) / 100    # estimated yield spread from the model
                if ys == 'error': raise ValueError(f'Cannot compute the spread since certain columns are null:\n{df}')
                ytw = yield_curve_level + ys
                ytw = np.abs(ytw)    # ytw should always be positive
                df['ficc_ytw'] = ytw
                price, calc_date = get_trade_price_from_yield_spread_model(df.iloc[0])
            else:
                ys_dealer_sell, ys_dealer_purchase, df = _get_spread(df)
                ys_dealer_purchase /= 100
                ys_dealer_sell /= 100
    
                ytw_dealer_sell = ys_dealer_sell + yield_curve_level
                ytw_dealer_sell = np.abs(ytw_dealer_sell)    # ytw should always be positive
                ytw_dealer_purchase = ys_dealer_purchase + yield_curve_level
                ytw_dealer_purchase = np.abs(ytw_dealer_purchase)    # ytw should always be positive

                df['ficc_ytw'] = [ytw_dealer_sell, ytw_dealer_purchase]
                prices = df.apply(get_trade_price_from_yield_spread_model, axis=1)

                # checking inversion and flipping prices if necessary
                price_sell, sell_calc_date = prices.iloc[0]
                price_buy, buy_calc_date = prices.iloc[1]
                price_sell, price_buy, sell_calc_date, buy_calc_date, ys_dealer_sell, ys_dealer_purchase = check_inversion_and_flip_prices(price_sell, price_buy, sell_calc_date, buy_calc_date, ys_dealer_sell, ys_dealer_purchase)

                # selecting appropriate values based on trade type
                if trade_type == 'S':
                    ys = ys_dealer_sell
                    price = price_sell
                    calc_date = sell_calc_date
                elif trade_type == 'P':
                    ys = ys_dealer_purchase
                    price = price_buy
                    calc_date = buy_calc_date

                # only take the relevant row of the dataframe
                ytw = ys + yield_curve_level
                ytw = np.abs(ytw)    # ytw should always be positive
                df = df[df.trade_type == trade_type]
                df['ficc_ytw'] = ytw

            # # refuse to price CUSIP within 60 days of the calc date
            # days_to_calc_date = diff_in_days_two_dates(calc_date, df['settlement_date'].iloc[0], convention='exact')
            # if days_to_calc_date <= 60: raise CustomMessageError(CUSIP_ERROR_MESSAGE['maturing_soon'])
        else:
            if not PRICE_BOTH_DIRECTIONS_TO_CORRECT_INVERSION or trade_type == 'D':
                price = _get_dollar_price(df)
            else:
                # checking inversion and flipping prices if necessary
                price_sell, price_buy, df = _get_dollar_price(df)    # handling two values for non-'D' case
                price_sell = np.abs(price_sell)    # price_sell should always be positive
                price_buy = np.abs(price_buy)    # price_buy should always be positive
                price_sell, price_buy = check_inversion_and_flip_prices(price_sell, price_buy)
                
                if trade_type == 'S':
                    price = price_sell
                elif trade_type == 'P':
                    price = price_buy
                df = df[df.trade_type == trade_type]

            ytw, calc_date = None, None
            df['ficc_ytw'] = DOLLAR_PRICE_MODEL_DISPLAY_TEXT[row.get('reason_for_using_dollar_price_model')]
        
        df['price'] = np.round(price, DISPLAY_PRECISION)    # NOTE: this does not round `price` which needs to be rounded further downstream prior to logging
        calc_date_display = datetime.strftime(calc_date, MONTH_DAY_YEAR) if calc_date is not None else None    # calc_date is now changed to the predicted calc_date instead of last_calc_date which is what it was assigned in `_add_calc_date_and_ficc_treasury_spread(...)`; this format is MONTH_DAY_YEAR, instead of YEAR_MONTH_DAY for presentation purposes
        df['calc_date'] = calc_date_display
        maturity_date = datetime.strptime(np.datetime_as_string(df.maturity_date.values[0], unit='s'), YEAR_MONTH_DAY + 'T' + HOUR_MIN_SEC)
        df['maturity_date'] = datetime.strftime(maturity_date, YEAR_MONTH_DAY)    # use this maturity date for similar bonds; change to MONTH_DAY_YEAR on the front end since the backend needs it in format YEAR_MONTH_DAY to find similar bonds
        print(f'ficc_ycl: {yield_curve_level}\testimated ytw: {ytw}\testimated price: {price}\testimated calc_date: {calc_date_display}\tmodel used: {model_used}')
        try:
            next_call_date = datetime.strptime(np.datetime_as_string(df.next_call_date.values[0], unit='s'), YEAR_MONTH_DAY + 'T' + HOUR_MIN_SEC)
            df['next_call_date'] = datetime.strftime(next_call_date, YEAR_MONTH_DAY)    # use this to display the call date on the front end
        except ValueError as e:
            next_call_date = None
            df['next_call_date'] = None

        try:
            refund_date = datetime.strptime(np.datetime_as_string(df.refund_date.values[0], unit='s'), YEAR_MONTH_DAY + 'T' + HOUR_MIN_SEC)
            df['refund_date'] = datetime.strftime(refund_date, YEAR_MONTH_DAY)    # use this to display the call date on the front end
        except ValueError as e:
            refund_date = None
            df['refund_date'] = None

        display_text_for_ytw = 'Worst'    # display the yield to _ on the front end
        display_price = REDEMPTION_VALUE_AT_MATURITY    # display the redemption value for the associated redemption date on the front end
        if calc_date is not None:
            if calc_date == maturity_date:
                display_text_for_ytw = 'Maturity'
            elif calc_date == refund_date:    # give priority to refund instead of call because there are bonds that are called at CAV and called at premium where the call date and fefund date and prices are the same, but the bond is called so we should show refund date and price
                display_text_for_ytw = 'Refund'
                display_price = df.refund_price
            elif calc_date == next_call_date:
                display_text_for_ytw = 'Call'
                display_price = df.next_call_price

        try:    # wrap this in a try...except statement so that an invalid `df.next_call_price` or `df.refund_price` are not sent to the front end
            display_price = np.round(float(display_price), DISPLAY_PRECISION)
            if display_price % 1 == 0: display_price = int(display_price)    # shave off the .0 if it exists
        except ValueError as e:
            display_price = REDEMPTION_VALUE_AT_MATURITY

        issue_date = datetime.strptime(np.datetime_as_string(df.issue_date.values[0], unit='s'), YEAR_MONTH_DAY + 'T' + HOUR_MIN_SEC)
        df['issue_date'] = datetime.strftime(issue_date, YEAR_MONTH_DAY)    # use this to display the dated date on the front end

        # extract the previous trade features which is a pandas series of length 1
        previous_trade_features_array = df.previous_trades_features.iloc[0]

        # map of each feature in `previous_trade_features_array` to position in array
        feature_to_idx = {'yield_spread': 0, 
                          'ficc_ycl': 1, 
                          # 'rtrs_control_number': 2, 
                          'yield_to_worst': 3, 
                          'dollar_price': 4, 
                          # 'seconds_ago': 5, 
                          'size': 6, 
                          'calc_date': 7, 
                          # 'maturity_date': 8, 
                          # 'next_call_date': 9, 
                          # 'par_call_date': 10, 
                          # 'refund_date': 11, 
                          'trade_datetime': 12, 
                          # 'calc_day_cat': 13, 
                          # 'settlement_date': 14, 
                          'trade_type': 15}

        # since now we process trades which do not have a history
        if len(previous_trade_features_array) > 0:
            # convert all trade_datetime features to a easy to read string for display on the front end
            previous_trade_features_array[:, feature_to_idx['trade_datetime']] = np.vectorize(lambda datetime: datetime_as_string(datetime, display=True))(previous_trade_features_array[:, feature_to_idx['trade_datetime']])    # np.vectorize allows us to apply this function to each item in the array

            num_trades, _ = previous_trade_features_array.shape    # second item in the shape tuple is the number of features per trade, which should be equal to len(feature_to_idx)
            num_trades = min(num_trades, 32)    # 32 is the maximum number of trades in the same CUSIP trade history; change this value to display fewer than 32 trades on the front end if desired

            # put previous trade features into a list of dictionaries
            def create_previous_trade_dict(trade_idx):
                previous_trade = {}
                for feature, feature_idx in feature_to_idx.items():
                    feature_value = previous_trade_features_array[trade_idx, feature_idx]
                    if feature == 'calc_date': feature_value = datetime_as_string(feature_value, precision='day', display=True)
                    previous_trade[feature] = feature_value
                return previous_trade
            
            previous_trades = [create_previous_trade_dict(trade_idx) for trade_idx in range(num_trades)]
        else:
            previous_trades = []

        df['previous_trades_features'] = [previous_trades]    # wrapping this in a list since the original df expects a single value for each column
        # create these next two variables as sets so we can easily perform an intersection
        features_to_display = {'security_description', 'price', 'ficc_ytw', 'calc_date', 'coupon', 'issue_date', 'next_call_date', 'previous_trades_features', 'model_used', 'reason_for_using_dollar_price_model'}
        features_for_similar_bonds = {'purpose_class', 'purpose_sub_class', 'coupon', 'incorporated_state_code', 'rating', 'maturity_date'}

        df['rating'] = df['rating'].replace('MR', 'NR')    # change rating from MR to NR since MR means no rating from any agency, and NR means no rating from S&P, but for the user, this should be the same
        if df['ficc_ytw'].iloc[0] is not None and type(df['ficc_ytw'].iloc[0]) != str: df['ficc_ytw'] = np.round(df['ficc_ytw'], DISPLAY_PRECISION)    # this is to display the predicted yield to worst with the correct number of decimal points on the front end
        df_dict_list = df[list(features_to_display | features_for_similar_bonds)].to_dict('records')    # returns a dictionary inside a list; since there is only one row in `df`, there is only one item in this list (the dictionary maps the feature to the feature value for each row)
        df_dict = df_dict_list[0]    # extract the dictionary from the list by choosing the 0-th index
        features_to_remove_where_feature_value_is_null = [feature for feature, feature_value in df_dict.items() if type(feature_value) != list and pd.isnull(feature_value)]    # first check whether the feature value is a list which is the case with previous_trade_features, since we cannot check if a list is null type
        for feature in features_to_remove_where_feature_value_is_null:    # need to perform this procedure since purpose_sub_class sometimes has a null feature_value
            df_dict.pop(feature)    # remove all features from the dictionary where the feature value is null

        df_dict['display_text_for_ytw'] = display_text_for_ytw
        df_dict['display_price'] = display_price
        response = make_response(jsonify([df_dict]), 200)    # need to listify the resulting set after the intersection when selecting columns in `df` due to `FutureWarning: Passing a set as an indexer is deprecated and will raise in a future version. Use a list instead.`

        log_usage(user=user, 
                  api_call=api_call, 
                  cusip=cusip, 
                  direction=trade_type, 
                  quantity=quantity // 1000, 
                  ficc_price=round_for_logging(price),    # need to be rounded to LOGGING_PRECISION (can perhaps do more but not too many more digits) decimal places otherwise will be detected as an invalid numerical value
                  ficc_ytw=round_for_logging(ytw) if ytw is not None else None,    # need to be rounded to LOGGING_PRECISION (can perhaps do more but not too many more digits) decimal places otherwise will be detected as an invalid numerical value
                  yield_spread=round_for_logging(ys) if model_used == 'yield_spread' else None,    # need to be rounded to LOGGING_PRECISION (can perhaps do more but not too many more digits) decimal places otherwise will be detected as an invalid numerical value; `ys` does not exist if dollar price model is used
                  ficc_ycl=round_for_logging(yield_curve_level),    # need to be rounded to LOGGING_PRECISION (can perhaps do more but not too many more digits) decimal places otherwise will be detected as an invalid numerical value
                  calc_date=datetime.strftime(calc_date, MONTH_DAY_YEAR) if calc_date is not None else None, 
                  model_used=model_used, 
                  recent=[np.float64(trade[0]) for trade in df['trade_history'].iloc[0]])    # `.iloc[0]` selects the first (and only) row in the dataframe since we only priced a single CUSIP, and trade[0] chooses the first item (corresponding to the yield spread) in each trade in the trade history; need to convert to float otherwise the following exception is raised: `TypeError: Object of type int64 is not JSON serializable`

        return response
    except Exception as e:
        calc_date = e.message if type(e) == CustomMessageError else None
        log_usage(user=user, 
                  api_call=api_call, 
                  cusip=cusip, 
                  direction=trade_type, 
                  quantity=quantity // 1000, 
                  calc_date=calc_date, 
                  error=True)
        print(f'{type(e)} inside `get_prediction_from_individual_pricing(...)`: {e}')    # error message will be present in the logs
        return e.get_json_response() if type(e) == CustomMessageError else CustomMessageError(DEFAULT_PRICING_ERROR_MESSAGE).get_json_response()
