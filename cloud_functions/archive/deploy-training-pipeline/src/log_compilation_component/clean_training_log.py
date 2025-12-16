'''
 # @ Create date: 2024-01-26
 # @ Modified date: 2024-03-29
 '''

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime


def send_email(sender_email: str, message: str, recipients: list) -> None:
    '''Send email with `message` to `recipients` from `sender_email`.'''
    smtp_server = 'smtp.gmail.com'
    port = 587

    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()
            sender_password = 'ztwbwrzdqsucetbg'
            server.login(sender_email, sender_password)
            for receiver in recipients:
                server.sendmail(sender_email, receiver, message.as_string())
        except Exception as e:
            print(e)
        finally:
            server.quit()

def remove_lines_with_character(character_to_remove, file_path, new_file_path=None):
    with open(file_path, 'r') as file:    # read the file
        lines = file.readlines()
    filtered_lines = [line for line in lines if character_to_remove not in line]    # filter out lines containing the specified character
    if new_file_path is None: new_file_path = file_path
    with open(new_file_path, 'w') as file:    # write the filtered lines back to the file
        file.writelines(filtered_lines)


def send_training_log(attachment_path, recipients:list, model:str, TESTING:bool=False):
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'

    print(f'Sending email to {recipients}')
    sender_email = 'notifications@ficc.ai'
    
    msg = MIMEMultipart()
    if TESTING:
        msg['Subject'] = f'(TESTING) Logs for pipeline run {datetime.now().strftime("%Y-%m-%d")}'
    else: 
        msg['Subject'] = f'Logs for pipeline run {datetime.now().strftime("%Y-%m-%d")}'

    msg['From'] = sender_email

    with open(attachment_path, 'rb') as attachment:    # attach the file
        part = MIMEApplication(attachment.read(), Name=attachment_path)
        part['Content-Disposition'] = f'''attachment; filename="{attachment_path}"'''    # using triple quotes to do 3 layers of nested quotes
        msg.attach(part)
    send_email(sender_email, msg, recipients)
