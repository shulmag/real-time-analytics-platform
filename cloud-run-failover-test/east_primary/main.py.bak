import os
import json
from datetime import datetime
import uuid
from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

def get_region():
    try:
        response = requests.get(
            'http://metadata.google.internal/computeMetadata/v1/instance/region',
            headers={'Metadata-Flavor': 'Google'}
        )
        return response.text.split('/')[-1]
    except Exception:
        return 'Unknown'

@app.route('/', methods=['GET'])
def beta_ficc_ai_function():
    region = get_region()
    
    # Each region fails on its own fail=true request
    if request.args.get('fail', '').lower() == 'true':
        return jsonify({"error": f"Simulated server failure in {region}"}), 500
    
    try:
        response_data = {
            "region": region,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "id": str(uuid.uuid4())
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))