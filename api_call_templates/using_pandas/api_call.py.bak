import requests
import pandas as pd


# fill out the EMAIL and PASSWORD variables below, as well as DEFAULT_TRADE_TYPE
EMAIL = 'YOUR EMAIL'
PASSWORD = 'YOUR PASSWORD'
DEFAULT_TRADE_TYPE = 'S'    # P: Purchase from Customer, S: Sale to Customer, D: Inter-Dealer


# do NOT change this function
def ficc_prices(cusips_quantities, use_default_trade_type=True):
    url = 'https://api.ficc.ai/api/batchpricing'
    data = {'username': EMAIL, 
            'password': PASSWORD}
    if use_default_trade_type:
        data['tradeType'] = DEFAULT_TRADE_TYPE
    data['cusipList'] = [item['CUSIP'] for item in cusips_quantities]
    data['quantityList'] = [item['Quantity'] for item in cusips_quantities]
    if not use_default_trade_type:
        data['tradeTypeList'] = [item['Trade Type'] for item in cusips_quantities_tradetypes]
    resp = requests.post(url, data=data)
    
    if resp.ok:
        try:
            return pd.read_json(resp.json())
        except Exception:
            pass
    print('batch error')


# example CUSIP, quantity (in thousands) pairs; feel free to put your own using the below format
cusips_quantities = [
    {'CUSIP': '64971XQM3', 'Quantity': 5},
    {'CUSIP': '64971XQM3', 'Quantity': 500},
    {'CUSIP': '13063DU89', 'Quantity': 100},
    {'CUSIP': '13063DLJ5', 'Quantity': 250},
    {'CUSIP': '54466HJM9', 'Quantity': 500},
    {'CUSIP': '650036CJ3', 'Quantity': 5000},        
]


# since we do not provide a trade type for each CUSIP, quantity pair in `cusips_quantities`, we set the `use_default_trade_type` to `True` and use `DEFAULT_TRADE_TYPE` as the trade type for each line item
print(ficc_prices(cusips_quantities, use_default_trade_type=True))


# example CUSIP, quantity (in thousands), trade type triples; feel free to put your own using the below format
# for trade type, use the following mapping. P: Purchase from Customer, S: Sale to Customer, D: Inter-Dealer
cusips_quantities_tradetypes = [
    {'CUSIP': '64971XQM3', 'Quantity': 5, 'Trade Type': 'S'},
    {'CUSIP': '64971XQM3', 'Quantity': 500, 'Trade Type': 'D'},
    {'CUSIP': '13063DU89', 'Quantity': 100, 'Trade Type': 'P'},
    {'CUSIP': '13063DLJ5', 'Quantity': 250, 'Trade Type': 'S'},
    {'CUSIP': '54466HJM9', 'Quantity': 500, 'Trade Type': 'P'},
    {'CUSIP': '650036CJ3', 'Quantity': 5000, 'Trade Type': 'S'},        
]


# since we provide a corresponding trade type for each CUSIP and quantity line item in `cusips_quantities_tradetypes`, we set the `use_default_trade_type` to `False`
print(ficc_prices(cusips_quantities_tradetypes, use_default_trade_type=False))
