# NOTE: THIS CODE DOES NOT WORK

import requests

import pandas as pd


def ficc_prices_request(request):
    url = 'https://server-dev-3ukzrmokpq-uc.a.run.app/api/price'
    resp = requests.get(url=url, params=request)

    print(resp, '\n')    # new line character to separate from other outputs
    print(resp.headers, '\n')    # new line character to separate from other outputs
    print(resp.json(), '\n')    # new line character to separate from other outputs

    if resp.ok:
        try:
            return pd.read_json(resp.json())
        except Exception:
            pass
    print('individual pricing error')


request = {
  'username': 'user@adajeinc.com',
  'password': 'Adaje2023',
  'cusip': '64971XQM3', 
  'quantity': 5, 
  'tradeType': 'S'
}


print(ficc_prices_request(request))
