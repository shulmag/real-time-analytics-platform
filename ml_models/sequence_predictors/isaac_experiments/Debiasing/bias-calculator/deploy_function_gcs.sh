#!/bin/bash

DEPLOYMENT_DIRECTORY="/home/jupyter/ficc/ml_models/sequence_predictors/isaac_experiments/Debiasing/bias-calculator"
ZIP_FILENAME="bias-calculator.zip"
DESTINATION_BUCKET="gcf-v2-sources-964018767272-us-central1/debiasing-test"

echo "Compressing function located in $DEPLOYMENT_DIRECTORY to $ZIP_FILENAME"
cd $DEPLOYMENT_DIRECTORY
echo "Compressing files" 

zip -r -q $ZIP_FILENAME .

echo "Files compressed to $DEPLOYMENT_DIRECTORY/$ZIP_FILENAME"
echo "Uploading to gs://$DESTINATION_BUCKET"

gcloud storage cp $ZIP_FILENAME gs://$DESTINATION_BUCKET

echo "Upload complete. Deleting zip file"
rm $ZIP_FILENAME