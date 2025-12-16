import requests

import pandas as pd


# fill out the EMAIL and PASSWORD variables below
EMAIL = 'YOUR EMAIL'
PASSWORD = 'YOUR PASSWORD'


# do NOT change this function
def ficc_prices(cusips_quantities_tradetypes):
    url = 'https://api.ficc.ai/api/batchpricing'
    data = {'username': EMAIL, 
            'password': PASSWORD}
    cusip_list = [item['CUSIP'] for item in cusips_quantities_tradetypes]
    quantity_list = [item['Quantity'] for item in cusips_quantities_tradetypes]
    trade_type_list = [item['Trade Type'] for item in cusips_quantities_tradetypes]
    data['cusipList'] = cusip_list
    data['quantityList'] = quantity_list
    data['tradeTypeList'] = trade_type_list
    resp = requests.post(url, data=data)

    if resp.ok:
        try:
            return pd.read_json(resp.json())
        except Exception:
            pass
    print('batch error')


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


print(ficc_prices(cusips_quantities_tradetypes))
