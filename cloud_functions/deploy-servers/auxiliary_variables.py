'''
'''
import threading    # using threading instead of multiprocessing since threading has shared memory (good for I/O-bound tasks) and multiprocessing uses multiple CPUs and each process has its own memory (good for CPU-bound tasks)

from pytz import timezone

from google.cloud import run_v2
from google.cloud.devtools import cloudbuild_v1


EASTERN = timezone('US/Eastern')

# Global variables
PROJECT_ID = "eng-reactor-287421"
ARTIFACT_REGISTRY = "cloud-run-source-deploy"
REGION = "us-central1"
IMAGE_URI_LOCATION = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REGISTRY}"
VPC_CONNECTOR_LOCATION = f"projects/{PROJECT_ID}/locations/{REGION}/connectors"

CPU_ROUTER = 2
MEMORY_ROUTER = "4Gi"
CPU_HIGH_MEMORY = 8
MEMORY_HIGH_MEMORY = "32Gi"
CONCURRENCY = 16
TIMEOUT = 3600
ENVIRONMENT = "gen1"
MIN_INSTANCES_ROUTER = 1
MAX_INSTANCES = 1000

# Define services and their Dockerfiles
SERVICES = ["router", "high_memory"]
DOCKERFILES = ["Dockerfile.router", "Dockerfile.high_memory"]

# Track background processes
BUILD_IMAGE_THREADS = []
DEPLOY_THREADS = []

# Initialize Google Cloud clients
BUILD_CLIENT = cloudbuild_v1.CloudBuildClient()
RUN_CLIENT = run_v2.ServicesClient()

# Use a threading event to stop all processes in case of failure
failure_event = threading.Event()
