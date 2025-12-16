'''Updates the hourly ETF prices. Used only for recordkeeping. Moved to archive on 2024-04-15 by Developer after discussion with team members.'''
import pandas as pd
import time
import requests
import csv
from google.cloud import bigquery

key = 'P51TNSAJMMUPP1W2'
interval = '15min'
etfs = [
    'HYD',
    'HYMB',
    'IBMJ',
    'IBMK',
    'IBML',
    'IBMM',
    'ITM',
    'MLN',
    'MUB',
    'PZA',
    'SHM',
    'SHYD',
    'SMB',
    'SUB',
    'TFI',
    'VTEB',
]
project_id = "eng-reactor-287421"
sp_etf_hourly_dataset = 'ETF_hourly_alphavantage'
bq_project_dataset = 'eng-reactor-287421.ETF_hourly_alphavantage'


def load_hourly_etf_prices_bq():
    '''
    This function loads the maturity data from the specified bigquery tables in the global etfs list and concatenates them
    into a single dataframe.
    '''

    client = bigquery.Client()
    etf_data = {}

    for table in etfs:
        query = '''
                SELECT * FROM {}.{}  ORDER BY Date DESC LIMIT 1 

                '''.format(
            sp_etf_hourly_dataset, table
        )

        df = pd.read_gbq(query, project_id=project_id, dialect='standard')

        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True, drop=True)
        etf_data[table] = df

    assert list(etf_data.keys()) == etfs

    return etf_data


def get_hourly_prices(intraday_df):
    df = intraday_df.copy().fillna(method='bfill')
    df.sort_index(ascending=True, inplace=True)
    df['minute'] = df.index.minute
    df['hour'] = df.index.hour

    df = df[
        ((df['hour'].isin([9, 10, 11, 12, 13, 14, 15])) & (df['minute'] == 30))
        | ((df['minute'] == 0) & (df['hour'] == 16))
    ]
    df.drop(['minute', 'hour'], axis=1, inplace=True)

    return df


def download_intraday_prices():
    count = 1
    times = [time.time()]

    dataframes = {}
    for symbol in etfs:
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={}&interval={}&slice=year1month1&apikey={}&adjusted=False'.format(
            symbol, interval, key
        )
        with requests.Session() as s:
            download = s.get(url)
            decoded_content = download.content.decode('utf-8')
            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            temp_df = pd.DataFrame(list(cr))
            temp_df.columns = temp_df.iloc[0]
            temp_df = temp_df.iloc[1:, :]

            temp_df = temp_df.rename(
                {
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'time': 'Date',
                },
                axis=1,
            )
            temp_df.set_index('Date', inplace=True)
            temp_df.columns = temp_df.columns + '_' + symbol
            temp_df.index = pd.to_datetime(temp_df.index)

            for col in temp_df:
                temp_df[col] = temp_df[col].astype(float)

            dataframes[symbol] = temp_df
        count += 1
        time.sleep(12)

    return dataframes


def get_col_names(df):
    return (
        df.filter(regex='Open').columns[0],
        df.filter(regex='Close').columns[0],
        df.filter(regex='Volume').columns[0],
        df.filter(regex='High').columns[0],
        df.filter(regex='Low').columns[0],
    )


def get_schema_intraday(Open, Close, Volume, High, Low):
    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField('Date', bigquery.enums.SqlTypeNames.DATETIME),
            bigquery.SchemaField(Open, bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField(Close, bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField(Volume, bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField(High, bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField(Low, bigquery.enums.SqlTypeNames.FLOAT),
        ],
        write_disposition="WRITE_APPEND",
    )

    return job_config


def get_hour_index(date):
    a = [pd.to_datetime('{} {}:30'.format(date, x)) for x in range(9, 16)]
    b = [pd.to_datetime('{} 16:00'.format(date))]

    return pd.DataFrame(index=a + b)


def main(args):
    bq_data = load_hourly_etf_prices_bq()
    intraday_data = download_intraday_prices()

    client = bigquery.Client()

    excluded = []

    data = {}

    for name, df in intraday_data.items():
        df = df.sort_index(ascending=True)
        data_last_date = str(df.index[-1].date())

        dates = pd.to_datetime(bq_data[name].index)
        bq_last_date = str(dates[0].date())

        if data_last_date == bq_last_date:
            excluded.append(name)
            continue

        df = df.loc[data_last_date:]
        df = pd.merge(
            get_hour_index(data_last_date),
            df,
            left_index=True,
            right_index=True,
            how='outer',
        )
        df = get_hourly_prices(df.fillna(method='bfill'))

        Open, Close, Volume, High, Low = get_col_names(df)
        job_config = get_schema_intraday(Open, Close, Volume, High, Low)
        table_id = bq_project_dataset + '.' + name
        df = df.reset_index(drop=False).rename({'index': 'Date'}, axis=1)

        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()

    if excluded:
        return 'Data for {} already available'.format(excluded)
    else:
        return 'Upload Successful for all ETFs'
