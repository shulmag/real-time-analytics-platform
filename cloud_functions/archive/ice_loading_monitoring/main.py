'''
Description: Checks the logs for ICE loading in BigQuery and the table is `eng-reactor-287421.reference_data.ice_files_loading_processing` 
and checks if a file is present, and we were unable to process the XML. This is crucial beause ICE changes the XML schema 
periodically, which causes our code to fail. The ways that it could fail are: (1) ICE adds a new field and causes our 
XML parser to fail, or (2) ICE changes the data type for a certain value and it fails during typecasting.
TODO: create a more robust XML parser and/or figure out ultimate source of the data types to prevent failures (since the 
documentation provided by ICE does not give the data types).
'''
import smtplib
from email.mime.text import MIMEText

from google.cloud import bigquery, secretmanager


def access_secret_version(secret_id, project_id='eng-reactor-287421', version_id='latest'):
    client = secretmanager.SecretManagerServiceClient()    # create the Secret Manager client
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'    # build the resource name of the secret version
    response = client.access_secret_version(request={'name': name})    # access the secret version
    payload = response.payload.data.decode('UTF-8')
    return payload


def send_error_email(subject, error_message):
    sender_email = access_secret_version('notifications_username')
    password = access_secret_version('notifications_password')
    receiver_emails = ['eng@ficc.ai', 'gil@ficc.ai', 'eng@ficc.ai']
    receiver_email = ', '.join(receiver_emails)

    msg = MIMEText(error_message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    smtp_server = 'smtp.gmail.com'
    port = 587
    sender_email = 'notifications@ficc.ai'

    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        except Exception as e:
            print(e)
        finally:
            server.quit()


def main(args):
    BQ_CLIENT = bigquery.Client()

    num_days_in_past_to_check = 5
    DATA_QUERY = f'''SELECT zip_file_name, status FROM `eng-reactor-287421.reference_data.ice_files_loading_processing` WHERE status = 1 AND DATE_DIFF(current_date, date, day) < {num_days_in_past_to_check} AND date > "2022-07-21" ORDER BY date DESC'''

    print(f'Getting dataframe from making the following BigQuery call:\n{DATA_QUERY}')
    df = BQ_CLIENT.query(DATA_QUERY).result().to_dataframe()
    if len(df) > 0:
        error_message = f'In the last {num_days_in_past_to_check} days we have failed to process the following ICE files:'
        for i in df['zip_file_name']:
            error_message += '\n' + str(i)
        send_error_email('[Warning] ICE File Loading error from cloud function `ice_loading_monitoring`', error_message)
        print(error_message)
    else:
        print(f'No failures when processing ICE files in the last {num_days_in_past_to_check} days')
    return 'Success'
