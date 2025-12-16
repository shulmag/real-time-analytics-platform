import requests

import pandas as pd


# fill out the EMAIL and PASSWORD variables below, as well as the trade type
EMAIL = 'YOUR EMAIL'
PASSWORD = 'YOUR PASSWORD'
TRADE_TYPE = 'S'    # P: Purchase from Customer, S: Sale to Customer, D: Inter-Dealer


# do NOT change this function
def ficc_prices(cusips_quantities):
    url = 'https://api.ficc.ai/api/batchpricing'
    data = {'username': EMAIL, 
            'password': PASSWORD, 
            'tradeType': TRADE_TYPE}
    cusip_list = [item['CUSIP'] for item in cusips_quantities]
    quantity_list = [item['Quantity'] for item in cusips_quantities]
    data['cusipList'] = cusip_list
    data['quantityList'] = quantity_list
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


print(ficc_prices(cusips_quantities))
