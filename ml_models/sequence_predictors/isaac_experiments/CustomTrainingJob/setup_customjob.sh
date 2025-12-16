export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export REPO_NAME=custom-train-job
export IMAGE_NAME=custom-train-job-test
export IMAGE_TAG=latest
export IMAGE_URI=us-east4-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

echo "Custom train job container will be uploaded to ${IMAGE_URI}"

echo "Building container"
docker build -f Dockerfile -t ${IMAGE_URI} ./

if [ $? -ne 0 ]; then
  echo "Building container failed with exit code $?"
  exit 1
fi

gcloud auth configure-docker us-east4-docker.pkg.dev 

echo "Uploading container"
docker push ${IMAGE_URI}

if [ $? -ne 0 ]; then
  echo "Pushing container to artefact registry failed with exit code $?"
  exit 1
fi

echo "Upload complete. Function exiting."