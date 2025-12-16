import google.cloud.aiplatform as aip
from google.cloud.aiplatform import hyperparameter_tuning as hpt


LOCATION = "us-central1"
STAGING_BUCKET = "gs://ficc-historical-results"
PROJECT_ID ='eng-reactor-287421'
SERVICE_ACCOUNT = "964018767272-compute@developer.gserviceaccount.com"
JOB_NAME = "historical_results" 
CONTAINER_URI = "us-east4-docker.pkg.dev/eng-reactor-287421/custom-train-job/ficc-historical-models:latest"

disk_spec = {
    "boot_disk_type": "pd-ssd"  ,
    "boot_disk_size_gb": 100
}

machine_spec = {
    "machine_type": "n1-highmem-8",
    "accelerator_type": "NVIDIA_TESLA_T4",
    "accelerator_count": 1
}


containerSpec = {
    "image_uri": CONTAINER_URI,
    "args": [
        "--train_months=6",
        "--NUM_EPOCHS=150",
        "--VALIDATION_SPLIT=0.1",
        "--bucket=custom-train-job-test",
        "--file=small_data.pkl",
        "--BATCH_SIZE=1000",
        "--LEARNING_RATE=0.0001"
    ]
}

worker_pool_spec = [
    {
        "replica_count": 1,
        "machine_spec": machine_spec,
        "disk_spec": disk_spec,
        "container_spec": containerSpec
    }
]


target_dates = ['2023-08-01',
                 '2023-08-02',
                 '2023-08-03',
                 '2023-08-04',
                 '2023-08-07',
                '2023-05-29']

if __name__=="__main__":
    aip.init(project=PROJECT_ID,
             staging_bucket=STAGING_BUCKET,
             location=LOCATION)
    
    job = aip.CustomJob(display_name=JOB_NAME, 
                worker_pool_specs=worker_pool_spec)
    
    hpt_dates = hpt.CategoricalParameterSpec(target_dates)

    hpt_job = aip.HyperparameterTuningJob(
        display_name=JOB_NAME,
        custom_job=job,
        metric_spec={
            "mae": "minimize",
        },
        parameter_spec={
            "target_date": hpt_dates,
        },
        max_trial_count=N,
        parallel_trial_count=N,
    )


    hpt_job.run(service_account = SERVICE_ACCOUNT)