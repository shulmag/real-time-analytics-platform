import smtplib
from email.mime.text import MIMEText


def send_error_email(subject, error_message):
    sender_email = 'error@ficc.ai'
    password = 'yctAkBarTS71'
    receiver_email = 'engineering@ficc.ai'
    
    msg = MIMEText(error_message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    smtp_server = 'smtp.gmail.com'
    port = 587
    sender_email = 'error@ficc.ai'

    with smtplib.SMTP(smtp_server,port) as server:
        try:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        except Exception as e:
            print(e)
        finally:
            server.quit()
