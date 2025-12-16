import csv
import requests

import pandas as pd

def cusipListMaker(cusips):
    cusips_quantities = cusips.split()
    cusips = cusips_quantities[::2]
    quantities = cusips_quantities[1::2]
    cusips_quantities = [{'CUSIP': cusip, 'Quantity': int(quantity)} for cusip, quantity in zip(cusips, quantities)]
    return cusips_quantities


def getFiccPrices(cusips_quantities):
    batchQuantity = 500
    url = 'https://server-3ukzrmokpq-uc.a.run.app/api/batchpricing'
    data = {'username' : 'aroghanchi@rseelaus.com',
            'password' : 'arjan2023',
            'tradeType' : 'D',
            'amount': 1
           }
    cusip_quantity_list = cusips_quantities
    with open('tmp.csv', 'w', newline='') as csvfile:
        fieldnames = ['CUSIP','Quantity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for item in cusip_quantity_list:
            writer.writerow(item)
    with open('tmp.csv') as csvfile:
        files = {'file' : csvfile}
        resp = requests.post(url,
                             data = data,
                             files = files
                            )
    if resp.ok:
        csvData = resp.content.decode('utf-8')
        csvReader = csv.DictReader(csvData.splitlines(),
                                   delimiter = ','
                                  )
        csvList = list(csvReader)
        return csvList
    else:
        print('batch error')


def getFiccPricesDf(csv):
    df = pd.DataFrame(csv)
    df['security_description'] = df['security_description'].apply(lambda desc: desc[:16])
    df['quantity'] = df['quantity'].astype(int)
    df['price'] = df['price'].astype(float)
    df['ytw'] = df['ytw'].astype(float)
    columns_to_keep = ['cusip', 'security_description', 'quantity', 'price', 'ytw']
    columns_to_rename = {'cusip': 'Cusip', 
                         'security_description': 'Des', 
                         'quantity': 'Qty', 
                         'price': 'FICCprice', 
                         'ytw': 'FICCyield'}
    return df[columns_to_keep].rename(columns=columns_to_rename)


cusips_quantities = [
    {'CUSIP': '64971XQM3', 'Quantity': 5},
    {'CUSIP': '64971XQM3', 'Quantity': 500},
    {'CUSIP': '13063DU89', 'Quantity': 100},
    {'CUSIP': '13063DLJ5', 'Quantity': 250},
    {'CUSIP': '54466HJM9', 'Quantity': 500},
    {'CUSIP': '650036CJ3', 'Quantity': 5000},        
]

print(getFiccPricesDf(getFiccPrices(cusips_quantities)))