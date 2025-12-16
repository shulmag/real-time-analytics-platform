# Vertex AI SDK Demo

The code here demonstrates how to train and experiment with models on Vertex AI's python SDK. Vertex AI simplifies the process of training and experimenting, especially for tasks like hyperparameter tuning, by providing tools for result tracking and resource scaling to enable parallelization.

The primary goal here is to backtest the performance of a neural network model over a specified range of historical dates. Manually training such a model using Compute Engine is not practical, as it is better suited for smaller and more complex experiments.

## Getting Started

To get started, follow these steps:

1. Run the `setup_customjob.sh` script to create and push the custom container image required by Vertex AI. Remember to modify the project name, repo and image name in `setup_customjob.sh` if required.

```bash
./setup_customjob.sh
```

If output has to be logged, use the following to log it to `output.txt`.

```bash
./setup_customjob.sh 2>&1 | tee output.txt
```

2. Run the `runSDK.py` script to set up a vertex AI job. Remember to open the file and configure the hyperparameters before doing so. This can also be done in `SDK Demo.ipynb`.

```bash
python runSDK.py
```

3. Experiment runs are shown on the Vertex AI Hyperparameter Tuning Job page on GCP Console, [here](https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs?authuser=1&project=eng-reactor-287421).


## Debugging
Vertex AI performs experiment runs using pre-built or custom container images. In this instance, we are using a custom container image with the code for our training application. The `setup_customjob.sh` script automates the process of building the docker image and uploading it to GCP's artifact registry, where it can be retrieved by Vertex AI for custom jobs. However, debugging the training application is often easier done locally, which can be done by running the following:
```bash
docker build -f Dockerfile -t ${IMAGE_NAME} ./
docker run --gpus=all ${IMAGE_NAME} **kwargs
```

For help with arguments the train.py script takes, run:
```bash
python train.py --help
```

For instance: 
```bash
python train.py \
--target_date=2023-05-26  \
--train_months=1 \
--NUM_EPOCHS=1 \
--VALIDATION_SPLIT=0.1 \
--bucket=custom-train-job-test  \
--file=data_latest_01-09_no_exclusions.pkl \
--bucket=custom-train-job-test \
--BATCH_SIZE=10000 
--LEARNING_RATE=0.0007
```

