'''
'''
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

from google.cloud import secretmanager

from auxiliary_functions import KEY_MATURITIES


def access_secret_version(secret_id: str, project_id: str = 'eng-reactor-287421', version_id='latest'):
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
    response = secretmanager.SecretManagerServiceClient().access_secret_version(request={'name': name})
    payload = response.payload.data.decode('UTF-8')
    return payload


def send_error_email_image(subject, message, images, cutoff):
    recipients = ['ficcteam@ficc.ai']    # 'eng@ficc.ai'
    sender_email = access_secret_version('notifications_username')
    sender_password = access_secret_version('notifications_password')
    
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender_email
    html = f'''\
    <html>
      <head><b>Ficc Real-time Yield Curve Updates for Maturities: {', '.join(KEY_MATURITIES.astype(str))} </b> <br> Yield curve values are effective as of {cutoff}</head>
      <body>
        {message}
      </body>
    </html>
    '''

    part1 = MIMEText(html, 'html')
    msg.attach(part1)

    for idx, img in enumerate(images):
        image = MIMEImage(img.read())
        image.add_header('Content-ID', f'<image{idx+1}>')
        msg.attach(image)

    smtp_server = 'smtp.gmail.com'
    port = 587

    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, msg.as_string())
        except Exception as e:
            print(e)
        finally:
            server.quit() 
