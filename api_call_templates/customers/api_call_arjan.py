import requests

import pandas as pd


REQUEST_URL = 'https://server-3ukzrmokpq-uc.a.run.app/'
USERNAME = 'aroghanchi@rseelaus.com'
PASSWORD = 'arjan2023'

NUMERICAL_ERROR = -1


def cusipListMaker(cusips):
    cusips_quantities = cusips.split()
    cusips = cusips_quantities[::2]
    quantities = cusips_quantities[1::2]
    cusips_quantities = [{'CUSIP': cusip, 'Quantity': int(quantity)} for cusip, quantity in zip(cusips, quantities)]
    return cusips_quantities


def getFiccPrices(cusips_quantities):
    url = REQUEST_URL + '/api/batchpricing'    # 'https://server-3ukzrmokpq-uc.a.run.app/api/batchpricing'
    data = {'username' : USERNAME,     # 'aroghanchi@rseelaus.com',
            'password' : PASSWORD,     # 'arjan2023',
            'tradeType' : 'D'}
    cusip_list = [item['CUSIP'] for item in cusips_quantities]
    quantity_list = [item['Quantity'] for item in cusips_quantities]
    data['cusipList'] = cusip_list
    data['quantityList'] = quantity_list
    resp = requests.post(url, data=data)
    if resp.ok:
        print(resp.json())
        return pd.read_json(resp.json())
    else:
        print('batch error')


def getFiccPricesDf(df):
    '''Input is a dataframe instead of CSV list.'''
    df['security_description'] = df['security_description'].apply(lambda desc: desc[:16])
    df['quantity'] = df['quantity'].astype(int)
    df['price'] = df['price'].astype(float)
    df['ytw'] = df['ytw'].astype(float)
    columns_to_keep = ['cusip', 'security_description', 'quantity', 'price', 'ytw', 'maturity_date', 'yield_to_worst_date']
    columns_to_rename = {'cusip': 'Cusip', 
                        'security_description': 'Des', 
                        'quantity': 'Qty', 
                        'price': 'FICCprice', 
                        'ytw': 'FICCyield', 
                        'maturity_date': 'Maturity', 
                        'yield_to_worst_date': 'W/O date'}
    return df[columns_to_keep].rename(columns=columns_to_rename)


cusips_quantities = [
    {'CUSIP': '64971XQM3', 'Quantity': 5},
    {'CUSIP': '64971XQM3', 'Quantity': 500},
    {'CUSIP': '13063DU89', 'Quantity': 100},
    {'CUSIP': '13063DLJ5', 'Quantity': 250},
    {'CUSIP': '54466HJM9', 'Quantity': 500},
    {'CUSIP': '650036CJ3', 'Quantity': 5000},        
]

df = getFiccPricesDf(getFiccPrices(cusips_quantities))
assert df['Cusip'].tolist() == [cusip_quantity['CUSIP'] for cusip_quantity in cusips_quantities]
assert df['Qty'].astype(int).tolist() == [int(cusip_quantity['Quantity']) * 1000 for cusip_quantity in cusips_quantities]
assert (df['FICCprice'] != NUMERICAL_ERROR).all()
assert (df['FICCyield'] != NUMERICAL_ERROR).all()
print(df)