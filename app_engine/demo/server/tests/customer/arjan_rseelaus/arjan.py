import numpy as np
import requests
import csv

def getFiccPrices(cusips):

    batchQuantity = 500

    url = 'https://server-3ukzrmokpq-uc.a.run.app/api/batchpricing'

    data = {'username' : 'aroghanchi@rseelaus.com',
            'password' : 'arjan2023',
            'amount' : batchQuantity, 
            'tradeType': 'S'}

    cusipList = cusips
    np.savetxt('tmp.csv',
           cusipList,
           delimiter =", ",
           fmt ='% s'
          )

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

def main():
    cusip_list = ['64971XQM3', '6461367J4', '13063DU89', '160429B88', '13063DLJ5', '54466HJM9', '650036CJ3']
    print(getFiccPrices(cusip_list))
    # batch_pricing()
    # res = price_cusip()
    # print(f"predicted yield: {res[0]['ficc_ytw']}, predicted price: {res[0]['price']}")

main()