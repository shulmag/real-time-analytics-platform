from google.cloud import bigquery
import smtplib, ssl
from email.mime.text import MIMEText

'''
This function is taken from Gil's script
https://github.com/Ficc-ai/ficc/blob/dev/lib/send_email.py
'''
def send_error_email(subject,error_message):
    sender_email = "error@ficc.ai"
    password = "yctAkBarTS71"
    receiver_email = "myles@ficc.ai"
    
    msg = MIMEText(error_message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    smtp_server = "smtp.gmail.com"
    port = 587
    sender_email = "error@ficc.ai"

    with smtplib.SMTP(smtp_server,port) as server:
        try:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        except Exception as e:
            # Print any error messages to stdout
            print(e)
        finally:
            server.quit() 

def main(): 
    bq_client = bigquery.Client()
        
    DATA_QUERY = """ SELECT
                    status, count(status) as count
                 FROM
                   `eng-reactor-287421.reference_data.ice_files_loading_processing`
                group by status
                    """
    df=bq_client.query(DATA_QUERY).result().to_dataframe()

    df.index=df.status
    if df.loc[0]['count'] != df.loc[2]['count']:
         send_error_email('Error','There was an error the data loaded is not equal to the data being processed')
    if 1 in list(df.index):
        send_error_email('Error','There was an error processing the data')
