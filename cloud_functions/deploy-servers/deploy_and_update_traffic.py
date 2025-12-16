'''
'''
import threading    # using threading instead of multiprocessing since threading has shared memory (good for I/O-bound tasks) and multiprocessing uses multiple CPUs and each process has its own memory (good for CPU-bound tasks)
from datetime import datetime

from google.cloud import run_v2
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import policy_pb2
from google.api_core.exceptions import NotFound
from google.protobuf.duration_pb2 import Duration

from auxiliary_variables import EASTERN, \
                                PROJECT_ID, \
                                RUN_CLIENT, \
                                VPC_CONNECTOR_LOCATION, \
                                REGION, \
                                MAX_INSTANCES, \
                                CONCURRENCY, \
                                IMAGE_URI_LOCATION, \
                                MIN_INSTANCES_ROUTER, \
                                CPU_ROUTER, \
                                MEMORY_ROUTER, \
                                CPU_HIGH_MEMORY, \
                                MEMORY_HIGH_MEMORY, \
                                TIMEOUT, \
                                DEPLOY_THREADS, \
                                ENVIRONMENT, \
                                failure_event
from auxiliary_functions import handle_failure, run_shell_command


def deploy_and_update_traffic_with_gcloud(service_name, vpc_connector, image_uri, min_instances, cpu, memory, route_to_different_server):
    '''Deploy and update traffic to `service_name`.'''
    print(f"Deploying {service_name}...")

    deploy_service_command = [
        "gcloud", "run", "deploy", service_name,
        "--image", image_uri,
        "--allow-unauthenticated",
        "--region", REGION,
        "--vpc-connector", vpc_connector,
        "--cpu", str(cpu),
        "--memory", memory,
        "--concurrency", str(CONCURRENCY),
        "--timeout", str(TIMEOUT),
        "--min-instances", str(min_instances),
        "--max-instances", str(MAX_INSTANCES),
        "--set-env-vars", f"ROUTE_TO_DIFFERENT_SERVER={route_to_different_server}",
        "--execution-environment", ENVIRONMENT
    ]

    if run_shell_command(deploy_service_command):
        print(f"Deployment successful for {service_name}, updating traffic...")
        update_traffic_command = [
            "gcloud", "run", "services", "update-traffic", service_name,
            "--to-latest",
            "--region", REGION
        ]
        if run_shell_command(update_traffic_command):
            print(f"Traffic updated for {service_name}")
        else:
            print(f"Traffic update failed for {service_name}")
    else:
        print(f"Deployment failed for {service_name}")


def allow_unauthenticated_access(project_id: str, region: str, service_name: str):
    '''Grants unauthenticated access to a Cloud Run service by assigning `roles/run.invoker` to `allUsers`. 
    This is equivalent to deploying with `gcloud run deploy ... --allow-unauthenticated`.'''
    resource_path = f"projects/{project_id}/locations/{region}/services/{service_name}"

    # Get the existing IAM policy
    policy = RUN_CLIENT.get_iam_policy(request={"resource": resource_path})

    # Define the policy binding for public access
    binding = policy_pb2.Binding(role="roles/run.invoker", members=["allUsers"])    # Allows public access

    # Append binding if not already present
    if binding not in policy.bindings:
        policy.bindings.append(binding)

        # Set the updated IAM policy
        set_policy_request = iam_policy_pb2.SetIamPolicyRequest(resource=resource_path, policy=policy)
        RUN_CLIENT.set_iam_policy(request=set_policy_request)
        print(f"Service `{service_name}` is now publicly accessible!")
    else:
        print(f"Service `{service_name}` is already public.")


def deploy_service_and_update_traffic(service_name, vpc_connector, image_uri, min_instances, cpu, memory, route_to_different_server):
    '''Deploy and update traffic for `service_name`.'''
    if failure_event.is_set(): return    # immediately terminate if previous events failed
    print(f"Deploying {service_name}...")

    parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    service = run_v2.Service(labels={'service_cloud_run': service_name}, 
                             template=run_v2.RevisionTemplate(containers=[{"image": image_uri,
                                                                           "env": [{"name": "ROUTE_TO_DIFFERENT_SERVER", "value": str(route_to_different_server)}, 
                                                                                   {"name": "DEPLOYMENT_VERSION", "value": datetime.now(EASTERN).strftime('%Y-%m-%d_%H-%M-%S')}],    # forces a new deployment since an environment variable is changing
                                                                           "resources": {"limits": {"cpu": str(cpu), 
                                                                                                    "memory": memory}, 
                                                                                         "cpu_idle": True, 
                                                                                         "startup_cpu_boost": True}}],
                                                              scaling=run_v2.RevisionScaling(max_instance_count=MAX_INSTANCES,
                                                                                             min_instance_count=min_instances),
                                                              max_instance_request_concurrency=CONCURRENCY, 
                                                              execution_environment=1, 
                                                              timeout=Duration(seconds=TIMEOUT),
                                                              vpc_access=run_v2.VpcAccess(connector=vpc_connector)))
    service_path = f"{parent}/services/{service_name}"
    try:
        # Step 1: Check if service already exists
        existing_service = RUN_CLIENT.get_service(name=service_path)
        print(f"Service {service_name} exists. Updating it with {image_uri} now...")
        # Step 2: Update the service
        existing_service.template = service.template
        update_request = run_v2.UpdateServiceRequest(service=existing_service, 
                                                     update_mask={"paths": ["template"]})
        operation = RUN_CLIENT.update_service(request=update_request)
        operation.result()    # Wait for update to complete
        print(f"Successfully updated {service_name}.")
    except NotFound as e:
        print(f"{type(e)}: {e}. Service {service_name} does not exist. Creating it with {image_uri} now...")

        # Step 3: Create the service if it doesn’t exist
        create_request = run_v2.CreateServiceRequest(parent=parent, service=service, service_id=service_name)
        operation = RUN_CLIENT.create_service(request=create_request)
        operation.result()    # Wait for deployment to complete
        print(f"Successfully created {service_name}")
    except Exception as e:
        handle_failure(f"Deployment failed for {service_name}. {type(e)}: {e}")

    allow_unauthenticated_access(PROJECT_ID, REGION, service_name)

    try:
        # Step 4: Update Traffic to Route 100% to the Latest Revision
        print(f"Updating traffic for {service_name} to latest revision...")
        existing_service = RUN_CLIENT.get_service(name=service_path)

        existing_service.traffic = [run_v2.TrafficTarget(percent=100, type_=run_v2.TrafficTargetAllocationType.TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST)]
        update_request = run_v2.UpdateServiceRequest(service=existing_service,
                                                     update_mask={"paths": ["traffic"]})
        traffic_update_op = RUN_CLIENT.update_service(request=update_request)
        traffic_update_op.result()    # Wait for traffic update to complete
        print(f"Traffic successfully updated for {service_name}")
    except Exception as e:
        handle_failure(f"Traffic update failed for {service_name}. {type(e)}: {e}")


DEPLOY_SERVICE_FUNC = deploy_service_and_update_traffic


def deploy_router_service(service_name, vpc_connector):
    thread = threading.Thread(target=DEPLOY_SERVICE_FUNC,
                              args=(service_name, vpc_connector, f"{IMAGE_URI_LOCATION}/router", MIN_INSTANCES_ROUTER, CPU_ROUTER, MEMORY_ROUTER, "True"))
    thread.start()
    DEPLOY_THREADS.append(thread)


def deploy_high_memory_service(service_name, vpc_connector):
    thread = threading.Thread(target=DEPLOY_SERVICE_FUNC,
                              args=(service_name, vpc_connector, f"{IMAGE_URI_LOCATION}/high_memory", 0, CPU_HIGH_MEMORY, MEMORY_HIGH_MEMORY, "False"))
    thread.start()
    DEPLOY_THREADS.append(thread)


def deploy_all_services(ignore_infrequently_used_services: bool = False):
    print("Starting deployment and traffic update...")

    # Deploy services in parallel
    deploy_router_service("server", f"{VPC_CONNECTOR_LOCATION}/server-connector")
    deploy_high_memory_service("server-batch-pricing-and-compliance", f"{VPC_CONNECTOR_LOCATION}/server-connector")
    if not ignore_infrequently_used_services: deploy_high_memory_service("server-investortools", f"{VPC_CONNECTOR_LOCATION}/server-connector-2")

    if not ignore_infrequently_used_services:
        for i in range(3):
            deploy_high_memory_service(f"server-vanguard-{i}", f"{VPC_CONNECTOR_LOCATION}/server-connector-vanguard")

    # Wait for all deployments to finish
    for thread in DEPLOY_THREADS:
        thread.join()

    if failure_event.is_set():
        print("Deployment process terminated due to failure.")
        exit(1)    # Ensures no false success message

    print("Servers successfully deployed with traffic switched to latest!")
