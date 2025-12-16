'''
'''
from modules.auxiliary_variables import NUMERICAL_ERROR

from modules.test.auxiliary_functions import REQUEST_URL
from modules.test.auxiliary_variables import USERNAME, PASSWORD


def test_arjan_2023_05_09():
    import requests

    import pandas as pd


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


def _test_arjan_2023_05_09():
    import requests

    import pandas as pd


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
            return pd.read_json(resp.json())
        else:
            print('batch error')


    def getFiccPricesDf(df):
        '''Input is a dataframe instead of CSV list.'''
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

    df = getFiccPricesDf(getFiccPrices(cusips_quantities))
    assert df['Cusip'].tolist() == [cusip_quantity['CUSIP'] for cusip_quantity in cusips_quantities]
    assert df['Qty'].astype(int).tolist() == [int(cusip_quantity['Quantity']) * 1000 for cusip_quantity in cusips_quantities]
    assert (df['FICCprice'] != NUMERICAL_ERROR).all()
    assert (df['FICCyield'] != NUMERICAL_ERROR).all()


def test_arjan_2023_05_01():
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
        url = REQUEST_URL + '/api/batchpricing'    # 'https://server-3ukzrmokpq-uc.a.run.app/api/batchpricing'
        data = {'username' : USERNAME,     # 'aroghanchi@rseelaus.com',
                'password' : PASSWORD,     # 'arjan2023',
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

    df = getFiccPricesDf(getFiccPrices(cusips_quantities))
    assert df['Cusip'].tolist() == [cusip_quantity['CUSIP'] for cusip_quantity in cusips_quantities]
    assert df['Qty'].astype(int).tolist() == [int(cusip_quantity['Quantity']) * 1000 for cusip_quantity in cusips_quantities]
    assert (df['FICCprice'] != NUMERICAL_ERROR).all()
    assert (df['FICCyield'] != NUMERICAL_ERROR).all()


def _test_arjan_2023_04_17():
    '''Runs the script that Arjan from R. Seelaus uses to price bonds with our product. If this  
    test fails, email Arjan at aroghanchi@rseelaus.com an email on how the code should be fixed  
    in order to continue using the product without error.'''
    import numpy as np
    import requests
    import csv

    def getFiccPrices(cusips):
        batchQuantity = 500
        url = REQUEST_URL + '/api/batchpricing'    # 'https://server-3ukzrmokpq-uc.a.run.app/api/batchpricing'
        data = {'username': USERNAME,     # 'aroghanchi@rseelaus.com',
                'password': PASSWORD,     # 'arjan2023',
                'amount': batchQuantity, 
                'tradeType': 'S'}

        np.savetxt('tmp.csv',
                   cusips,
                   delimiter=", ",
                   fmt='% s')

        with open('tmp.csv') as csvfile:
            files = {'file' : csvfile}
            resp = requests.post(url,
                                 data=data,
                                 files=files)

        if resp.ok:
            csvData = resp.content.decode('utf-8')
            csvReader = csv.DictReader(csvData.splitlines(),
                                       delimiter=',')

            csvList = list(csvReader)
            return csvList

        else:
            return 'batch error'


    def main():
        cusip_list = ['64971XQM3', '6461367J4', '13063DU89', '160429B88', '13063DLJ5', '54466HJM9', '650036CJ3']
        ficc_prices = getFiccPrices(cusip_list)
        assert ficc_prices != 'batch error', 'Response was not successful for batch pricing'

    
    main()
