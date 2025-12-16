import pandas as pd
import random
import matplotlib.pyplot as plt
import json
import requests
from google.cloud import bigquery
from collections import defaultdict
import io
from datetime import datetime, timedelta
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday

EMAIL = 'gil@ficc.ai'
PASSWORD = ''

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'

class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]

def get_last_business_day():
    today = datetime.now().date()
    business_day = CustomBusinessDay(calendar=USHolidayCalendarWithGoodFriday())
    last_business_date = today - business_day
    return last_business_date.strftime('%Y-%m-%d')

last_business_date = get_last_business_day()

query = f"""
SELECT
  cusip,
  maturity_date,
  sp_long,
  -- Calculate years to maturity using last business day
  DATE_DIFF(maturity_date, DATE('{last_business_date}'), DAY) / 365.25 AS years_to_maturity,
  -- Bucket assignment
  CASE
    WHEN DATE_DIFF(maturity_date, DATE('{last_business_date}'), DAY) / 365.25 >= 1 AND DATE_DIFF(maturity_date, DATE('{last_business_date}'), DAY) / 365.25 < 3 THEN '1 to 3'
    WHEN DATE_DIFF(maturity_date, DATE('{last_business_date}'), DAY) / 365.25 >= 3 AND DATE_DIFF(maturity_date, DATE('{last_business_date}'), DAY) / 365.25 < 7 THEN '3 to 7'
    WHEN DATE_DIFF(maturity_date, DATE('{last_business_date}'), DAY) / 365.25 >= 7 AND DATE_DIFF(maturity_date, DATE('{last_business_date}'), DAY) / 365.25 < 12 THEN '7 to 12'
    WHEN DATE_DIFF(maturity_date, DATE('{last_business_date}'), DAY) / 365.25 >= 12 AND DATE_DIFF(maturity_date, DATE('{last_business_date}'), DAY) / 365.25 < 20 THEN '12 to 20'
    WHEN DATE_DIFF(maturity_date, DATE('{last_business_date}'), DAY) / 365.25 >= 20 THEN '20+'
    ELSE NULL
  END AS years_to_maturity_bucket,
  -- Group assignment
  CASE
    WHEN sp_long IN ('AAA', 'AA') THEN 'AAA/AA'
    WHEN sp_long IN ('A', 'BBB') THEN 'BBB/A'
    ELSE NULL
  END AS rating_group
FROM
  `reference_data_v2.reference_data_flat`
WHERE
  sp_long IN ('AAA', 'AA', 'A', 'BBB')
  AND ref_valid_to_date > CURRENT_TIMESTAMP()
  AND maturity_date > DATE('{last_business_date}')
"""

def sql_to_df(query):
    client = bigquery.Client()
    df = client.query(query).to_dataframe()
    return df

def ficc_prices(batch):
    url = 'https://api.ficc.ai/api/batchpricing'
    resp = requests.post(url, data={
        'username': EMAIL,
        'password': PASSWORD,
        'cusipList': [x['cusip'] for x in batch],
        'quantityList': [x['quantity'] for x in batch],
        'tradeTypeList': [x['side'] for x in batch]
    })
    resp.raise_for_status()
    return resp.json()

def main():
    df = sql_to_df(query)
    print(f"df.shape: {df.shape}")
    df.to_pickle('df.pkl')
    quantities = [25, 100, 250, 500, 750, 1000, 1250]
    sides = ['S','P']

    print(df['years_to_maturity_bucket'].value_counts(dropna=False))

    expected_rating_groups = ['AAA/AA', 'BBB/A']
    expected_years_buckets = ['1 to 3', '3 to 7', '7 to 12', '12 to 20', '20+']

    spreads_table = defaultdict(dict)

    for rating_group in expected_rating_groups:
        for years_bucket in expected_years_buckets:
            cusips = df[(df['rating_group'] == rating_group) & 
                        (df['years_to_maturity_bucket'] == years_bucket)]['cusip'].tolist()
            if not cusips:
                for q in quantities:
                    spreads_table[(rating_group, years_bucket)][q] = None
                continue

            N = 100
            if N < len(cusips):
                cusips = random.sample(cusips, N)
            else:
                cusips = cusips

            batch = []
            for cusip in cusips:
                for q in quantities:
                    for side in sides:
                        batch.append({'cusip': cusip, 'quantity': q, 'side': side})

            #print(f"Requesting {len(batch)} scenarios for {rating_group}, {years_bucket}")
            results = ficc_prices(batch)
            
            # Parse the JSON string if it's a string
            if isinstance(results, str):
                results = json.loads(results)
            
            # Convert the results to a DataFrame
            df_results = pd.DataFrame(results)
            
            # Convert notional to bond count
            df_results['quantity_bonds'] = df_results['quantity'] // 1000

            # Filter out rows with error messages
            df_results = df_results[df_results['error_message'].isna()]

            # Ensure pairs (CUSIP, quantity) have both sides (Bid and Offer)
            bid_offers = df_results.pivot_table(
                index=['cusip', 'quantity_bonds'],
                columns='trade_type',
                values='price',
                aggfunc='first'
            ).dropna(subset=['Bid Side', 'Offered Side'])

            # Compute spread per scenario explicitly
            bid_offers['Spread'] = bid_offers['Offered Side'] - bid_offers['Bid Side']

            # Now average these spreads per quantity_bonds
            average_spreads = bid_offers.groupby('quantity_bonds')['Spread'].mean()

            for q in quantities:
                spread_value = average_spreads.get(q, None)
                spreads_table[(rating_group, years_bucket)][q] = spread_value

    # Prepare for DataFrame
    output = []
    for (rating_group, years_bucket), spread_dict in spreads_table.items():
        for q, spread in spread_dict.items():
            output.append({
                "Rating Group": rating_group,
                "Years Bucket": years_bucket,
                "Quantity (bonds)": q,
                "Spread": spread
            })

    df_spreads = pd.DataFrame(output)
    
    # DEBUG: Check for duplicates before pivoting
    dupes = df_spreads.duplicated(subset=["Rating Group", "Years Bucket", "Quantity (bonds)"])
    print("Any duplicates? ", dupes.any())
    if dupes.any():
        print(df_spreads[dupes])

    print(f"date: {last_business_date}")
    for rating in ['AAA/AA', 'BBB/A']:
        display_df = df_spreads[df_spreads['Rating Group'] == rating].pivot(
            index='Years Bucket',
            columns='Quantity (bonds)',
            values='Spread'
        ).reindex([
            '1 to 3', '3 to 7', '7 to 12', '12 to 20', '20+'
        ])

        print(f"\n{rating}\n")
        print(display_df.to_markdown(floatfmt=".3f"))



if __name__ == '__main__':
    main()