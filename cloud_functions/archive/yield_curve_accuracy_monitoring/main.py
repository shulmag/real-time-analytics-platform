'''
 '''
 
import pandas as pd
from google.cloud import secretmanager
from pandas.tseries.holiday import USFederalHolidayCalendar
import requests
import json
import redis
import pickle5 as pickle
import numpy as np
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from ficc_ycl import yield_main
from google.cloud import bigquery


REDIS_CLIENT = redis.Redis(host='10.146.62.92', port=6379, db=0)
bday_us = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())

def send_error_email(subject,error_message):
    receiver_email = "ficcteam@ficc.ai"
    #receiver_email = "eng@ficc.ai"
    sender_email = "notifications@ficc.ai"
    
    recipients = [receiver_email] 
    emaillist = [elem.strip().split(',') for elem in recipients]
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender_email
    html = """\
    <html>
      <head></head>
      <body>
        {0}
      </body>
    </html>
    """.format(error_message)

    part1 = MIMEText(html, 'html')
    msg.attach(part1)

    smtp_server = "smtp.gmail.com"
    port = 587

    with smtplib.SMTP(smtp_server,port) as server:
        try:
            server.starttls()
            server.login(sender_email, 'ztwbwrzdqsucetbg')
            server.sendmail(sender_email, receiver_email, msg.as_string())
        except Exception as e:
            print(e)
        finally:
            server.quit() 

def get_yield_value(maturity,target_date):    
    request_json = {"maturity":maturity,"target_date":target_date}
    result = yield_main(request_json)
    return result['result']

def get_mmd_data():
    bq_client = bigquery.Client()
    query = '''
            SELECT date, maturity, AAA 
            FROM `eng-reactor-287421.yield_curves.mmd_approximation` 
            order by date desc, maturity asc limit 5
            '''

    query_job = bq_client.query(query)
    df = query_job.result().to_dataframe()
    return df

def main(args):
    flag = False
    most_recent_business_day = (datetime.now() - bday_us).strftime('%Y-%m-%d')
    index_data = get_mmd_data()
    index_data['AAA'] = index_data['AAA'] * 100
    email_df = pd.DataFrame(data=[],columns=['MMD AAA YCL','Date reported','Maturity','Ficc YCL','Delta'])
    for maturity in [1,5,10,15,30]:
        target_date = index_data[index_data.maturity == maturity]['date'].values[0]
        calculated_ytw = np.round(get_yield_value(maturity, most_recent_business_day),2)
        index_yield = index_data[index_data.maturity == maturity]['AAA'].astype(float).values[0]
        delta_yield = np.round(abs(calculated_ytw - index_yield),2)
        if delta_yield > 10:
            temp_df = pd.DataFrame(data=[[index_yield, target_date, maturity, calculated_ytw, delta_yield]],
                                  columns=['MMD AAA YCL','Date reported','Maturity','Ficc YCL','Delta'])
            email_df = pd.concat([email_df,temp_df])
            flag = True
    
    email_df.reset_index(inplace=True, drop=True)
    if flag == True:
        send_error_email(f'WARNING! yield curve delta more than 10 bps', email_df.to_html(index=False)) 
        return "ERROR"
    else:
        return "Success"