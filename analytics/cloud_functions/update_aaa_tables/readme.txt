#Deployment

gcloud functions deploy update_aaa_tables \
  --runtime python312 \
  --region us-central1 \
  --source . \
  --entry-point main \
  --trigger-http \
  --memory 1024MB \
  --timeout 540s \
  --vpc-connector yield-curve-connector