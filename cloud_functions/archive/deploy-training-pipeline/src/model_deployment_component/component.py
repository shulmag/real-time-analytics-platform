'''
 # @ Create date: 2024-02-16
 # @ Modified date: 2024-06-03
 '''
from kfp import dsl
from kfp.dsl import Model, Dataset, Artifact

project_id = 'eng-reactor-287421'

@dsl.component(base_image=f'gcr.io/{project_id}/data-processing-base:test', \
               target_image=f'gcr.io/{project_id}/model-deployment-component:v1', 
               packages_to_install=['google-cloud-aiplatform==1.28.1',
                                    'protobuf==3.20.3']) #we downgrade protobuf for compatibility with ai-platform
def model_deployment_component(model_artifact: Model,
                            #    validation_artifact: Artifact,
                               region: str,
                               endpoint_name: str,
                               TESTING:bool) -> Artifact:
    '''Kubeflow component that trains yield spread model 

    Args:
        var (type): 

    Returns:    
        var (type): 
        
    '''
    import os
    import sys
    import gcsfs
    from datetime import datetime
    from ficc.utils.auxiliary_functions import function_timer
    from logger import Logger
    from clean_training_log import send_training_log
    from google.cloud import aiplatform

    project_id = 'eng-reactor-287421'

    aiplatform.init(
        project=project_id,
        location=region,
    )

    @function_timer
    def upload_model_to_vertex_ai(model_path, model_display_name):
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_path,
            serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-11:latest',
        )

        model.wait()

        print(f'Model display name: {model.display_name}')
        print(f'Model resource name:  {model.resource_name}')
        return model

    @function_timer
    def deploy_model_to_vertex_ai(model, endpoint_name, UNDEPLOY_OLD_MODELS):
        # deploy new model
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
        model.deploy(
            endpoint=endpoint,
            traffic_percentage=100,
            accelerator_type = 'NVIDIA_TESLA_T4',
            accelerator_count=1
        )

        if UNDEPLOY_OLD_MODELS:
            # clean up endpoint by removing the old models if there are more than 1 deployed
            deployed_models = sorted(endpoint.list_models(), key=lambda x: x.create_time)
            if len(deployed_models) > 1:     # keep only the most recent model
                old_models = deployed_models[:-1]
                print(f'More than 1 model deployed, removing all but the most recent model. Removing models with names: {[old_model.display_name for old_model in old_models]} and IDs: {[old_model.id for old_model in old_models]}')
                for old_model in old_models:
                    endpoint.undeploy(deployed_model_id=old_model.id)

    ##### SETUP LOGGING #####
    log_file = f'model-deployment-component-log_{datetime.now().strftime("%Y-%m-%d")}.log'
    original_stdout = sys.stdout
    logger = Logger(log_file, original_stdout)
    sys.stdout = logger
    ##### SETUP LOGGING #####

    
    DEPLOY = model_artifact.metadata['deploy_decision']
    SUCCESS = False #boolean passed to archiving component so that it knows to archive the model or not

    if TESTING:
        DEPLOY = True
        SUCCESS = True
        UNDEPLOY_OLD_MODELS = False 
        model_display_name = f'test-model-{datetime.now().strftime("%Y-%m-%d")}'
        print('TESTING = True; model will be force deployed but old models will not be removed from endpoint')
    else:
        UNDEPLOY_OLD_MODELS = True
        model_display_name = f'model-{datetime.now().strftime("%Y-%m-%d")}'
    
    if DEPLOY==True:
        try:
            model_path = model_artifact.metadata['model_path']
            model = upload_model_to_vertex_ai(model_path, model_display_name)
            deployed_model_response = deploy_model_to_vertex_ai(model, endpoint_name, UNDEPLOY_OLD_MODELS)
            print("SUCCESS - Model uploaded and deployed successfully.")
            SUCCESS = True
        except Exception as e: 
            print(f"FAILURE - Unexpected behaviour, model met requirements for deployment but failed to deploy with error message {e}.")
            raise e
    else: 
        print("FAILURE - Model did not meet requirements for deployment. Component not failing because this is expected behaviour.")
        

    artifact = Artifact(uri = dsl.get_uri())
    log_path = os.path.join(artifact.uri, log_file)
    metadata = {'log_path':log_path,
                'SUCCESS':SUCCESS}
    
    for k in metadata: 
        artifact.metadata[k] = metadata[k]

    print('Information about the artifact')
    print('Name:', artifact.name)
    print('URI:', artifact.uri)
    print('Path:', artifact.path)
    print('Metadata:', artifact.metadata)

    # Close logging 
    logger.close()
    sys.stdout = original_stdout

    fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
    with fs.open(log_path, 'wb') as gcs_file:
        with open(log_file, 'rb') as local_file:
            gcs_file.write(local_file.read())

    if SUCCESS==False:
        print('Model failed to deploy. If TESTING=False and the deploy_decision=False, this is the intended behaviour.')
    
    return artifact