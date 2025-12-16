import csv
import requests

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
        fieldnames = ['CUSIP', 'Quantity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
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


input_cusips_quantities = input('?')
cusips_quantities = input_cusips_quantities.split()
cusips = cusips_quantities[::2]
quantities = cusips_quantities[1::2]
cusips_quantities = [{'CUSIP': cusip, 'Quantity': int(quantity)} for cusip, quantity in zip(cusips, quantities)]    # cusips_and_quantities has been overwritten to be the dictionary values
result = getFiccPrices(cusips_quantities)
print(result)