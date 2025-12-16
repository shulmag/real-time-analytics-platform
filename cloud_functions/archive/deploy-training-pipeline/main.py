from kfp import dsl, compiler
import google.cloud.aiplatform as aip
import datetime
import functions_framework
from src.data_processing_component.component import data_processing_component
from src.model_training_component.component import train_model_component
from src.log_compilation_component.component import log_compilation_component
from src.archiving_component.component import archiving_component
from src.model_deployment_component.component import model_deployment_component
from functools import partial

project_id = 'eng-reactor-287421'
pipeline_root_path  = 'gs://ficc-pipelines-test'

def set_compute_resources(obj, cpu_limit = None, memory_limit = None, gpu=None, gpu_count=None):
    
    if cpu_limit:
        obj.set_cpu_limit(cpu_limit)
    
    if memory_limit:
        obj.set_memory_limit(memory_limit)
    
    if gpu: 
        obj.add_node_selector_constraint(gpu)
        
    if gpu and gpu_count:
        obj. set_gpu_limit(gpu_count)
        
    if gpu and not gpu_count:
        obj.set_gpu_limit(1)


# def set_production_model_training_resource(obj):
#     set_compute_resources(obj, 
#                           cpu_limit = "32", 
#                           memory_limit = "200G",
#                           gpu="NVIDIA_TESLA_T4", 
#                           gpu_count=2)

# def set_production_data_processing_resource(obj):
#     set_compute_resources(obj, 
#                           cpu_limit = "64", 
#                           memory_limit = "200G")

def run_job(enable_caching, TESTING = True):
    curr_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    if TESTING: name = f"test-automated-training-job-{curr_date}"
    else: name = f"automated-training-job-{curr_date}"

    # Before initializing, make sure to set the GOOGLE_APPLICATION_CREDENTIALS
    # environment variable to the path of your service account.
    aip.init(
        project=project_id,
        location='us-central1',
    )

    # Prepare the pipeline job
    job = aip.PipelineJob(
        display_name=name,
        job_id=name,
        template_path="automated-training.yaml",
        pipeline_root=pipeline_root_path,
        parameter_values={
            'project_id': project_id
        },
        enable_caching=enable_caching,
        failure_policy='slow' #failure_policy=slow ensures that the pipeline runs even after failures (e.g. if deployment-component fails, logs still compile and send)
    )

    job.submit()

@dsl.pipeline(name="automated-training", 
              pipeline_root=pipeline_root_path)
def pipeline(project_id: str, 
             TESTING: bool = False,
             model: str = 'yield_spread',
             parameters: dict = {'NUM_EPOCHS':100, 'BATCH_SIZE':10000, 'LEARNING_RATE': 0.007},
             email_recipients: list = ['eng@ficc.ai', 'gil@ficc.ai', 'eng@ficc.ai', 'eng@ficc.ai']
             ) -> bool:
    
    data_op = data_processing_component(TESTING=TESTING,
                                        model = model, 
                                        file_name='processed_data.pkl',
                                        bucket_name='ficc-pipelines-test'
                                        )
    set_compute_resources(data_op, 
                          cpu_limit = "64", 
                          memory_limit = "400G")
    
    train_op = train_model_component(dataset = data_op.output,
                                     model = model, 
                                     TESTING = TESTING,
                                     parameters = parameters,
                                     email_recipients=email_recipients)
    
    set_compute_resources(train_op, 
                          cpu_limit = "32", 
                          memory_limit = "200G",
                          gpu="NVIDIA_TESLA_T4", 
                          gpu_count=2) # 2 GPUs because highmem32 CPU does not support just 1 T4; this is a known issue to be resolved by reducing dataframe size
    
    deploy_op = model_deployment_component(model_artifact=train_op.output,
                                           region = 'us-central1',
                                           endpoint_name = '2038540737385070592',
                                           TESTING=TESTING)
    set_compute_resources(deploy_op, 
                          cpu_limit = "1")
    
    logging_op = log_compilation_component(model_artifact=train_op.output,
                                           data_artifact=data_op.output, 
                                           deployment_artifact=deploy_op.output,
                                           email_recipients=email_recipients,
                                           TESTING=TESTING)
    set_compute_resources(logging_op, 
                          cpu_limit = "1")

    archive_op = archiving_component(data_artifact = data_op.output,
                                        model_artifact = train_op.output,
                                        destination_bucket_name='ficc-pipelines-test', # remove `test` suffix once officially moved to production
                                        ) 
    set_compute_resources(archive_op, 
                          cpu_limit = "1", 
                          memory_limit = "1G")
    
    
    return archive_op.output

@functions_framework.http
def main(request):

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="automated-training.yaml", 
        )

    run_job(enable_caching=False,
            TESTING=False)
    
    return 'SUCCESS'