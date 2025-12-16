import requests


# fill out the EMAIL and PASSWORD variables below
EMAIL = 'YOUR EMAIL'
PASSWORD = 'YOUR PASSWORD'


# do NOT change this function
def ficc_prices(cusips_quantities_tradetypes):
    base_url = 'https://api.ficc.ai/api/batchpricing'
    cusip_list = [row['CUSIP'] for row in cusips_quantities_tradetypes]
    quantity_list = [row['Quantity'] for row in cusips_quantities_tradetypes]
    trade_type_list = [row['Trade Type'] for row in cusips_quantities_tradetypes]
    response = requests.post(base_url, data={'username': EMAIL, 
                                             'password': PASSWORD, 
                                             'cusipList': cusip_list, 
                                             'quantityList': quantity_list, 
                                             'tradeTypeList': trade_type_list})
    return response.json()


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


ficc_prices(cusips_quantities_tradetypes)
