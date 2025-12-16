'''
 # @ Create date: 2024-02-16
 # @ Modified date: 2024-02-16
 '''
  
from kfp.dsl import Model, Dataset, Artifact
from kfp import dsl

project_id = 'eng-reactor-287421'

@dsl.component(base_image=f'gcr.io/{project_id}/data-processing-base:test', \
               target_image=f'gcr.io/{project_id}/log-compilation-component:v1')
def log_compilation_component(data_artifact: Dataset, 
                              model_artifact: Model,
                              deployment_artifact: Artifact,
                              email_recipients: list = ['eng@ficc.ai', 'eng@ficc.ai', 'eng@ficc.ai', 'eng@ficc.ai'],
                              TESTING: bool = False) -> bool:
    '''Kubeflow component that consolidates and emails all pipeline logs 

    Args:
        data_artifact (Dataset): Dataset object containing path and metadata for successfully processed data file
        model_artifact (Model): Model object containing path and metadata for successfully trained model
        email_recipient (list): List of recipients for log email
        

    Returns:    
        bool: Success or failure 
        
    '''
    import sys
    import gcsfs
    from datetime import datetime
    from logger import Logger
    from clean_training_log import remove_lines_with_character, send_training_log

    ##### SETUP LOGGING #####
    log_file = f'log-compilation-component-log_{datetime.now().strftime("%Y-%m-%d")}.log'
    original_stdout = sys.stdout
    logger = Logger(log_file, original_stdout)
    sys.stdout = logger
    ##### SETUP LOGGING #####

    compiled_log_file = f'compiled-pipeline-log_{datetime.now().strftime("%Y-%m-%d")}.log'
    fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')

    logs = (('data-processing-component', data_artifact.metadata['log_path']),
            ('model-training-component', model_artifact.metadata['log_path']),
            ('model-deployment-component', deployment_artifact.metadata['log_path']),
            )


    with open(compiled_log_file, 'ab') as f:
        for component_name, path in logs:
            # path = component.metadata['log_path']
            print(f'Adding logs from {component_name}, located at {path}')
            header = '\n\n'+'#'*25 + f" {component_name.upper()} logs " + '#'*25 + '\n\n'
            f.write(header.encode('utf-8'))
            with fs.open(path, 'rb') as gcs_file:
                f.write(gcs_file.read())

    # Close logging 
    logger.close()
    sys.stdout = original_stdout

    with open(compiled_log_file, 'ab') as f:
        print(f'Adding final logs from log-compilation-component')
        header = '\n\n'+'#'*25 + f" LOG-COMPILATION COMPONENT logs " + '#'*25 + '\n\n'
        f.write(header.encode('utf-8'))
        with open(log_file, 'rb') as lf:
            f.write(lf.read())

    send_training_log(compiled_log_file, 
                      email_recipients, 
                      'yield_spread',
                      TESTING)

    return True