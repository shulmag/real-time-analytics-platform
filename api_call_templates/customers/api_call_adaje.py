import requests

import pandas as pd


def ficc_prices_request(request):
    url = 'https://server-dev-3ukzrmokpq-uc.a.run.app/api/batchpricing'
    resp = requests.post(url, data=request)

    print(resp, '\n')    # new line character to separate from other outputs
    print(resp.headers, '\n')    # new line character to separate from other outputs
    print(resp.json(), '\n')    # new line character to separate from other outputs

    if resp.ok:
        try:
            return pd.read_json(resp.json())
        except Exception:
            pass
    print('batch error')


request = {
  'username': 'user@adajeinc.com',
  'password': 'Adaje2023',
  'cusipList': [
    '64971XQM3',
    '64971XQM3',
    '13063DU89',
    '13063DLJ5'
  ],
  'quantityList': [
    5,
    500,
    100,
    250
  ],
  'tradeType': 'S'
}


print(ficc_prices_request(request))
