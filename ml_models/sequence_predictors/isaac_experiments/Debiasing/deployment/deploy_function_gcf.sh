gcloud functions deploy debiasing-test \
--gen2 \
--entry-point=main  \
--source "/home/jupyter/ficc/ml_models/sequence_predictors/isaac_experiments/Production Testing/deployment" \
--runtime=python38 \
--memory="8GB" \
--region='us-central1'  
# --timeout=150  \
# --max-instances 1  \
# --min-instances 1 \

# --trigger-topic=msrb_intraday_real_time_trade_files_update
# --vpc-connector=VPC_CONNECTOR
#--trigger-bucket="msrb_intraday_real_time_trade_files" \
#--trigger-location=us 

# #!/bin/bash

# # Set the variables
# FUNCTION_NAME="debiasing-test"
# BUCKET_NAME="msrb_intraday_real_time_trade_files"
# PROJECT_ID="eng-reactor-287421"
# REGION="us-central1"
# ENTRY_POINT="main"
# RUNTIME="python38"
# MEMORY="8GB"
# TIMEOUT="300"

# # Deploy the Cloud Function
# gcloud functions deploy ${FUNCTION_NAME} \
#   --gen2 \
#   --project=${PROJECT_ID} \
#   --region=${REGION} \
#   --entry-point=${ENTRY_POINT} \
#   --runtime=${RUNTIME} \
#   --memory=${MEMORY} \
#   --trigger-bucket=${BUCKET_NAME} \
#   --timeout=${TIMEOUT} \
#   --max-instances=1 \
#   --min-instances=1 \
#   --source="/home/jupyter/ficc/ml_models/sequence_predictors/isaac_experiments/Production Testing/deployment" \
#   --trigger-location=${REGION}

# # Set up the Eventarc trigger
# gcloud eventarc triggers create ${FUNCTION_NAME}-trigger \
#   --project=${PROJECT_ID} \
#   --location=${REGION} \
#   --destination-run-service=${FUNCTION_NAME} \
#   --destination-run-region=${REGION} \
#   --event-filters="type=google.cloud.storage.object.v1.finalized" \
#   --event-filters="resource=${BUCKET_NAME}"
