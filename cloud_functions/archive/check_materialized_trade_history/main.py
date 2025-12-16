import pandas as pd
import pytz
import datetime


def main(request):
    tz = pytz.timezone('US/Pacific')
    now = datetime.datetime.now(tz)  # .strftime('%Y-%m-%d %H-%M')
    PROJECT_ID = "eng-reactor-287421"

    query = 'SELECT MAX(trade_date) as trade_date FROM `eng-reactor-287421.auxiliary_views.materialized_trade_history` '
    df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')
    print(f"Time now: {now}; Max trade_date: {df.iloc[0][0].strftime('%Y-%m-%d')}")
    return 'SUCCESS'
