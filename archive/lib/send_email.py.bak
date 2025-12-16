def send_error_email(subject,error_message):
    import smtplib, ssl

    sender_email = "error@ficc.ai"
    password = "yctAkBarTS71"
    receiver_email = "engineering@ficc.ai"

    from email.mime.text import MIMEText
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
            return(str(e))
        finally:
            server.quit() 