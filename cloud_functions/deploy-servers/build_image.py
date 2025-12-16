'''
'''
import os
import threading    # using threading instead of multiprocessing since threading has shared memory (good for I/O-bound tasks) and multiprocessing uses multiple CPUs and each process has its own memory (good for CPU-bound tasks)
import tarfile

from google.cloud import storage
from google.cloud.devtools import cloudbuild_v1

from auxiliary_variables import PROJECT_ID, ARTIFACT_REGISTRY, IMAGE_URI_LOCATION, BUILD_CLIENT, SERVICES, DOCKERFILES, BUILD_IMAGE_THREADS, failure_event
from auxiliary_functions import handle_failure, run_shell_command


def switch_directory():
    target_dir = os.path.abspath("../../app_engine/demo/server")
    try:
        os.chdir(target_dir)
        print(f"Changed directory to {os.getcwd()}")
    except FileNotFoundError as e:
        handle_failure(f"Could not change to server directory: {target_dir}. {type(e)}: {e}")


def build_image_with_gcloud(service: str, dockerfile: str) -> None:
    '''Build an image from `dockerfile`.'''
    image_uri = f"{IMAGE_URI_LOCATION}/{service}"
    print(f"Building {service} and uploading to {image_uri} using {dockerfile}...")

    build_image_command = [
        "gcloud", "builds", "submit",
        "--config=cloudbuild.yaml",
        f"--substitutions=_IMAGE_URI={image_uri},_DOCKERFILE={dockerfile}"
    ]

    if run_shell_command(build_image_command):
        print(f"Build successful for {service}")
    else:
        print(f"Build failed for {service}")


def upload_source_to_gcs(bucket_name: str, object_name: str, source_dir: str):
    '''Tar and upload your source directory to GCS.'''
    tar_path = "/tmp/source.tar.gz"

    # Create tar.gz archive
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(source_dir, arcname=".")

    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(tar_path)

    print(f"Uploaded source to gs://{bucket_name}/{object_name}")


def build_image(service: str, dockerfile: str) -> None:
    '''Build an image from `dockerfile`. When using the cloud build API, we must upload the files to a storage first.
    NOTE: this function does not work as is. The build fails with the error: `Build failed for base. <class 'google.api_core.exceptions.Unknown'>: None Build failed; check build logs for details 2: Build failed; check build logs for details`, 
    but the difficulty is that the image is not being sent to Google Cloud Build and so we cannot inspect the logs.'''
    image_uri = f"{ARTIFACT_REGISTRY}/{service}"
    print(f"Building {service} and uploading to {image_uri} using {dockerfile}...")

    # Set up storage reference
    bucket_name = "cloudbuild_api"
    object_name = f"{service}/source.tar.gz"
    source_dir = "."

    upload_source_to_gcs(bucket_name, object_name, source_dir)
    
    build = cloudbuild_v1.Build(steps=[cloudbuild_v1.BuildStep(name="gcr.io/cloud-builders/docker", 
                                                               args=["build", "-t", image_uri, "-f", dockerfile, "."]), 
                                       cloudbuild_v1.BuildStep(name="gcr.io/cloud-builders/docker", 
                                                               args=["push", image_uri])],
                                images=[image_uri],
                                source=cloudbuild_v1.Source(storage_source=cloudbuild_v1.StorageSource(bucket=bucket_name, 
                                                                                                       object_=object_name)))

    try:
        operation = BUILD_CLIENT.create_build(project_id=PROJECT_ID, build=build)
        print(f"Submitted build for {service}. Waiting for result...")
        operation.result()    # Wait for build to complete
        print(f"Build successful for {service}")
    except Exception as e:
        handle_failure(f"Build failed for {service}. {type(e)}: {e}")


BUILD_IMAGE_FUNC = build_image_with_gcloud


def build_all_images():
    switch_directory()
    print("Starting base image build...")
    BUILD_IMAGE_FUNC("base", "Dockerfile.base")

    print("Starting parallel image builds...")
    for service, dockerfile in zip(SERVICES, DOCKERFILES):
        thread = threading.Thread(target=BUILD_IMAGE_FUNC, args=(service, dockerfile))
        thread.start()
        BUILD_IMAGE_THREADS.append(thread)

    # Wait for all builds to finish
    for thread in BUILD_IMAGE_THREADS:
        thread.join()

    if failure_event.is_set():
        print("Build process terminated due to failure.")
        exit(1)
