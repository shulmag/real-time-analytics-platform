'''
'''
import smtplib
from email.mime.text import MIMEText


def send_error_email(subject, error_message):
    sender_email = 'error@ficc.ai'
    password = 'yctAkBarTS71'
    receiver_emails = ['eng@ficc.ai', 'eng@ficc.ai', 'gil@ficc.ai']
    receiver_email = ', '.join(receiver_emails)

    message = MIMEText(error_message)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = receiver_email

    smtp_server = 'smtp.gmail.com'
    port = 587
    sender_email = 'error@ficc.ai'

    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        except Exception as e:
            print(e)    # print any error messages to stdout
        finally:
            server.quit()
