'''
Firebase authentication module for ficc analytics.
Handles token verification and user authentication.
'''

import os
import firebase_admin
from firebase_admin import auth, credentials
import json

# Initialize Firebase Admin SDK with credentials
try:
    # Path to service account key file
    cred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fbAdminConfig.json')
    
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        print(f"[INFO] Firebase initialized with credentials from {cred_path}")
    else:
        print(f"[WARNING] Firebase credentials file not found at {cred_path}")
        print(f"[WARNING] Authentication will not work without valid credentials")
except Exception as e:
    print(f"[ERROR] Failed to initialize Firebase: {str(e)}")
    print(f"[ERROR] Authentication will not work - fix Firebase credentials")

def verify_firebase_token(id_token):
    """
    Verify the provided Firebase ID token and return the decoded token.
    Validates that the token is properly signed and not expired.
    
    Args:
        id_token (str): The Firebase ID token to verify
        
    Returns:
        dict: The decoded token if verification succeeds, containing user information
              including 'email', 'uid', and other claims
        
    Raises:
        ValueError: If verification fails due to invalid, expired, or malformed token
    """
    try:
        # Verify the ID token
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print(f"[ERROR] Token verification failed: {str(e)}")
        raise ValueError(f"Invalid token: {str(e)}")