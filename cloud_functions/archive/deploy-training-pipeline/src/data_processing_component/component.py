'''
 # @ Create date: 2024-01
 # @ Modified date: 2024-06-16
 '''
  
from kfp import dsl
from kfp.dsl import Model, Dataset

project_id = 'eng-reactor-287421'

@dsl.component(base_image=f'gcr.io/{project_id}/data-processing-base:test', \
               target_image=f'gcr.io/{project_id}/data-processing-component:test')
def data_processing_component(TESTING:bool, 
                              model:str,
                              file_name:str = None,
                              bucket_name:str = None,
                              output_file_name:str = None
                              ) -> Dataset:
    '''Kubeflow component that loads the existing data file, queries, processes and saves new data and outputs a Dataset artifact.

    Args:
        TESTING (bool): Boolean indictor for testing or production; in testing, queries always pull 1000 new trades
        model (str): 'yield_spread' or 'dollar_price'
        file_name (str): name of processed data pickle file to load from Cloud Storage; defaults to 'processed_data_test.pkl' based on auxiliary_variables
        bucket_name (str): name of Cloud Storage bucket from with to load processed data pickle file; defaults to 'ficc-pipelines-test' based on auxiliary_variables
        output_file_name (str): name of updated data file to save in pipeleine uri; defaults to same name as loaded file if not specified

    Returns:    
        data_artifact (kfp.dsl.Dataset): Dataset object containing path and metadata for the new processed data file
        
    '''
    import os 
    import sys
    import pandas as pd
    import gcsfs
    from datetime import datetime
    from pandas.tseries.offsets import BDay
    from ficc.utils.auxiliary_functions import function_timer
    from automated_training_auxiliary_functions import save_update_data_results_to_pickle_files, \
                                                            padded_print, \
                                                            get_storage_client, \
                                                            get_bq_client, \
                                                            remove_lines_with_character
    from automated_training_auxiliary_variables import EASTERN, YEAR_MONTH_DAY, COLUMNS_FOR_TRAINING_COMPONENT
    import automated_training_auxiliary_functions as ATAF
    from logger import Logger
    
    def get_data_query_TESTING(last_trade_date, features, conditions):
        features_as_string = ', '.join(features)
        last_trade_date = (pd.to_datetime(last_trade_date) - BDay(1)).strftime('%Y-%m-%d') #ensures that trades are always returned 
        conditions = conditions + [f'trade_date > "{last_trade_date}"']
        conditions_as_string = ' AND '.join(conditions)
        return f'''SELECT {features_as_string}
                FROM `eng-reactor-287421.auxiliary_views.materialized_trade_history`
                WHERE {conditions_as_string}
                ORDER BY trade_datetime desc
                LIMIT 1000'''
    
    ##### SETUP LOGGING #####
    current_datetime = datetime.now(EASTERN).strftime(YEAR_MONTH_DAY)
    log_file = f'data-processing-component-log_{current_datetime}.log'
    original_stdout = sys.stdout # save the original sys.stdout in a temporary variable to restore it before ending script; will raise an error if not done
    logger = Logger(log_file, original_stdout)
    sys.stdout = logger
    padded_print(f' STARTING DATA-PROCESSING-COMPONENT ({current_datetime}) ')
    
    ##### SET VARIABLES ######

    # GCP API access in Vertex AI components do not need creds if we specify the correct project_id
    # We do not want creds to be in the container for security reasons!
    STORAGE_CLIENT = get_storage_client()
    BQ_CLIENT = get_bq_client()
    STORAGE_CLIENT.project = project_id
    BQ_CLIENT.project = project_id
    
    if bucket_name: 
        ATAF.BUCKET_NAME = bucket_name
    if file_name: 
        ATAF.CUMULATIVE_DATA_PICKLE_FILENAME_YIELD_SPREAD = file_name
    if not output_file_name:
        #if name for output file is not given, then reuse the original name
        output_file_name = ATAF.CUMULATIVE_DATA_PICKLE_FILENAME_YIELD_SPREAD

    if TESTING:
        '''
        If the pipeline is deployed in testing mode (TESTING=True), then it will do the following: 
            - get_data_query() will be replaced by get_data_query_TESTING(), which guarantees that trades will be returned (for testing purposes)
        Else:
            - it will use get_data_query(), which will return no new trades if processed_data.pkl has already been updated
        '''
        ATAF.get_data_query = get_data_query_TESTING
        padded_print(' Data-processing-component deployed in testing mode; data query will always return 1000 trades ')
    else:
        padded_print(' Data-processing-component deployed in production mode; component will fail if there are no new trades ')

    ##### MAIN FUNCTION ######
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    data, last_trade_date, num_features_for_each_trade_in_history, _ = save_update_data_results_to_pickle_files(model)

    ##### ARTIFACT MANAGEMENT #####
    data_artifact = Dataset(uri = dsl.get_uri())
    main_data_path = os.path.join(data_artifact.uri, output_file_name) # path for main data file
    training_data_path = os.path.join(data_artifact.uri, 'training_' + output_file_name) # path for data file with just the columns needed for training
    log_path = os.path.join(data_artifact.uri, log_file)

    padded_print(f' SAVING MAIN DATA FILE TO {main_data_path} ', 50)
    data.to_pickle(main_data_path)
    padded_print(f' DATA SAVED to {main_data_path} ', 50)

    padded_print(f' SAVING SUBSET OF MAIN DATA FILE FOR TRAINING TO {training_data_path} ', 50)
    data[COLUMNS_FOR_TRAINING_COMPONENT].to_pickle(training_data_path)
    padded_print(f' DATA SAVED to {training_data_path} ', 50)

    padded_print(' COMPONENT SUMMARY: ', 25)
    
    metadata = {'N': len(data), 
                'last_trade_date':last_trade_date, 
                'num_features_for_each_trade_in_history': num_features_for_each_trade_in_history,
                'file_path': main_data_path,
                'training_data_path': training_data_path,
                'log_path': log_path}

    for k in metadata: 
        data_artifact.metadata[k] = metadata[k]

    print('Information about the artifact')
    print('Name:', data_artifact.name)
    print('URI:', data_artifact.uri)
    print('Path:', data_artifact.path)
    print('Metadata:', data_artifact.metadata)

    ##### CLOSE LOGGING #####
    padded_print(f' Saving logs to {log_path} ', 25)
    padded_print(' EXITING DATA-PROCESSING-COMPONENT ')
    logger.close()
    sys.stdout = original_stdout

    ##### WRITE LOGS TO CLOUD STORAGE #####
    remove_lines_with_character('', log_file)

    if log_path.startswith('gs://'):
        fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
        with fs.open(log_path, 'wb') as gcs_file:
            with open(log_file, 'rb') as local_file:
                gcs_file.write(local_file.read())
    
    return data_artifact