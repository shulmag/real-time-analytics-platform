'''
'''
import paramiko
from datetime import datetime
from pytz import timezone

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re

from google.cloud import secretmanager


EASTERN = timezone('US/Eastern')


class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]


def today_is_a_holiday() -> bool:
    '''Determine whether today is a US national holiday.'''
    now = datetime.now(EASTERN)
    today = pd.Timestamp(now).tz_localize(None).normalize()    # `.tz_localize(None)` is to remove the time zone; `.normalize()` is used to remove the time component from the timestamp
    current_year = now.year
    holidays_in_last_year_and_next_year = set(USHolidayCalendarWithGoodFriday().holidays(start=f'{current_year - 1}-01-01',end=f'{current_year + 1}-12-31'))
    if today in holidays_in_last_year_and_next_year:
        print(f'Today, {today}, is a national holiday, and so we will not perform large batch pricing, and so there will not be any files in the SFTP')
        return True
    return False


def access_secret_version(secret_id: str, project_id: str = 'eng-reactor-287421', version_id='latest'):
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
    response = secretmanager.SecretManagerServiceClient().access_secret_version(request={'name': name})
    payload = response.payload.data.decode('UTF-8')
    return payload


def send_email(subject, message):
    sender_email = access_secret_version('notifications_username')
    recipients = ['ficc-eng@ficc.ai']
    password = access_secret_version('notifications_password')

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = ', '.join(recipients)

    body = MIMEText(message, 'html')
    msg.attach(body)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, recipients, msg.as_string())


def check_sftp_files(context):
    if today_is_a_holiday(): return 'SUCCESS'    # since we do not perform large batch pricing on a national holiday, we do not need to check if there are files in the SFTP server

    # SFTP connection details
    sftp_host = 'sftp.ficc.ai'
    sftp_username = access_secret_version('investortools_sftp_username')
    sftp_password = access_secret_version('investortools_sftp_password')
    directory = '/uploads'

    # connect to SFTP
    transport = paramiko.Transport((sftp_host, 22))
    transport.connect(username=sftp_username, password=sftp_password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    now = datetime.now(EASTERN)    # get current time in ET

    files = sftp.listdir_attr(directory)    # get list of files and their modification times
    
    # filter files for today and extract their upload times
    today_files = []
    for f in files:
        match = re.search(r'priced_(\d{4}-\d{2}-\d{2})--(\d{2}-\d{2}-\d{2})_', f.filename)
        if match:
            file_date = datetime.strptime(match.group(1), '%Y-%m-%d').date()
            file_time = datetime.strptime(match.group(2), '%H-%M-%S').time()
            if file_date == now.date():
                file_datetime = EASTERN.localize(datetime.combine(file_date, file_time))    # create a timezone-aware datetime object
                today_files.append(file_datetime)

    # sort today's files by time
    today_files.sort()
    file_count = len(today_files)

    # determine expected files and criticality
    expected_files = (now.hour - 8 + 1) // 2    # 9:00 -> 1, 10:00 -> 1, 11:00 -> 2, 12:00 -> 2, 13:00 -> 3, 14:00 -> 3, 15:00 -> 4, 16:00 -> 4, 17:00 -> 5
    expected_files = min(max(expected_files, 0), 5)    # number of expected files should be between 0 and 5 (the number of times per day that we run the `large_batch_pricing` cloud function to price the entire universe for Investortools; cloud scheduler job: `price_entire_universe_for_investor_tools`)
    success = file_count == expected_files
    is_critical = file_count < (expected_files / 2)    # consider it a critical situation if less than half the number of expected files are present

    # close SFTP connection
    sftp.close()
    transport.close()

    # prepare email
    subject = f"{file_count} / {expected_files} files uploaded successfully to the SFTP for Investortools as of {now.strftime('%Y-%m-%d %H:%M:%S')} ET"
    if is_critical:
        subject_prefix = 'CRITICAL'
    elif success:
        subject_prefix = 'SUCCESS'
    else:
        subject_prefix = 'WARNING'
    subject = f'[{subject_prefix}] {subject}'

    file_times = [f.strftime('%Y-%m-%d %H:%M:%S %Z') for f in today_files]
    message = f'''
    <p>Number of files uploaded today: {file_count}</p>
    <p>Expected number of files: {expected_files}</p>
    <p>All files uploaded today:</p>
    <ul>
    {''.join(f"<li>{file_time}</li>" for file_time in file_times)}
    </ul>
    '''
    if subject_prefix != 'SUCCESS': send_email(subject, message)
    
    print(f'Checked SFTP. Found {file_count} files out of {expected_files} expected. Files posted at: {file_times}')
    return f'Found {file_count} files out of {expected_files} expected.'


if __name__ == '__main__':    # for local testing
    print(check_sftp_files(None))
