import requests
import pandas as pd

api_key  = 'enh1r7bXrpiR3lNOBhvBZV4RRVqIN3ZI'
limit = 500
start_date = '2022-08-22'
end_date = '2022-08-22'
ticker = 'FMHI'

if __name__ == '__main__':
    query = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}?adjusted=false&sort=desc&limit={limit}&apiKey={api_key}'
    response = requests.get(query)
    response = response.json()
    response = response['results']
    response = pd.DataFrame(response)
    response['t'] = pd.to_datetime(response['t'], unit='ms')
    response['t'] = response['t'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    print(response)
