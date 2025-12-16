import os
import smtplib
import logging
from email.mime.text import MIMEText
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import secretmanager

# Set Google credentials environment variable for local testing
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/gil/git/ficc/creds.json"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ficc.ai Campus Access API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to access secrets from Google Secret Manager
def access_secret_version(secret_id: str, project_id: str = 'eng-reactor-287421', version_id='latest'):
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
    response = secretmanager.SecretManagerServiceClient().access_secret_version(request={'name': name})
    payload = response.payload.data.decode('UTF-8')
    return payload

def send_email(subject, message):
    # Multiple recipients as a list
    receiver_emails = ['myles@ficc.ai', 'gil@ficc.ai','jon@ficc.ai']
    
    # For display in the email header (as comma-separated string)
    receiver_email_str = ', '.join(receiver_emails)
    
    sender_email = access_secret_version('notifications_username')
    sender_password = access_secret_version('notifications_password')
    
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email_str
    
    smtp_server = 'smtp.gmail.com'
    port = 587
    
    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()
            server.login(sender_email, sender_password)
            # Pass the list of recipients to sendmail
            server.sendmail(sender_email, receiver_emails, msg.as_string())
            logger.info(f"Email sent successfully to {receiver_email_str}")
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            print(e)
        finally:
            server.quit()

@app.get("/")
async def root():
    return {"message": "ficc.ai Campus Access API is running"}

@app.post("/api/apply")
async def submit_application(data: dict = Body(...)):
    print(data)
    try:
        # Extract form data
        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        
        # Log the application
        logger.info(f"Received application from: {name} ({email})")
        
        # Create email body
        email_body = f"""
New Campus Access Application

Name: {name}
Email: {email}

This student has requested access to the ficc.ai User Interface through the Campus Access program.
Please review and provide credentials within 24-48 hours.
        """
        
        # Send email notification
        send_email(
            subject=f"New Campus Access Application: {name}",
            message=email_body
        )
        
        # Return success response
        return {
            "status": "success",
            "message": "Your application has been submitted successfully. We will contact you shortly with your access credentials."
        }
    
    except Exception as e:
        # Log errors
        logger.error(f"Error processing application: {e}")
        return {
            "status": "error",
            "message": "An error occurred while processing your application. Please try again later."
        }

if __name__ == "__main__":
    import uvicorn
    # Run the server when executed directly
    uvicorn.run(app, port=8000)