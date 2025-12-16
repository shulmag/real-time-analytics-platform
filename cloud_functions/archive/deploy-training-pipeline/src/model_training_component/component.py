'''
 # @ Create date: 2024-01
 # @ Modified date: 2024-05-20
 '''

from kfp import dsl
from kfp.dsl import Model, Dataset

project_id = 'eng-reactor-287421'

@dsl.component(base_image=f'gcr.io/{project_id}/training-base:test', \
               target_image=f'gcr.io/{project_id}/train_model_component:test')
def train_model_component(dataset:Dataset, 
                          model:str,
                          TESTING: bool,
                          parameters: dict,
                          email_recipients:list,
                          current_date:str = None,
                          bucket_name:str = 'ficc-pipelines-test') -> Model:
    '''Kubeflow component that trains yield spread model 

    Args:
        dataset (kfp.dsl.Dataset): Dataset object containing path to training data file, which should be a pickled pandas dataframe 

    Returns:
        model (kfp.dsl.Model): Model object containing path to successfully trained model
    '''

    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'

    if TESTING: 
        print('===== Pipeline deployed in testing mode =====')
    import os
    import sys 
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import gcsfs
    import traceback
    import pickle5 as pickle
    from automated_training_auxiliary_variables import EASTERN, YEAR_MONTH_DAY
    from automated_training_auxiliary_functions import setup_gpus, train_model, get_trade_date_where_data_exists_on_this_date, decrement_business_days, get_model_results, send_no_new_model_email, send_results_email_multiple_tables
    import automated_training_auxiliary_functions as ATAF
    from ficc.utils.auxiliary_functions import function_timer
    from logger import Logger
    from clean_training_log import remove_lines_with_character, send_training_log

    '''
    If the pipeline is deployed in testing mode (TESTING=True), then it will do the following: 
        - predictions will not be uploaded to bigquery using upload_predictions() 
            - NOTE: THIS IS CURRENTLY REMOVED FOR TESTING PURPOSES
        - data will always be returned for the model to train in train_model()
    '''
    
    if bucket_name: 
        ATAF.BUCKET_NAME = bucket_name

    ATAF.TESTING = TESTING
    if TESTING: 
        ATAF.SAVE_MODEL_AND_DATA = False
    EMAIL_RECIPIENTS = email_recipients
    setup_gpus()

    ##### SETUP LOGGING #####
    log_file = f'model-training-component-log_{datetime.now().strftime("%Y-%m-%d")}.log'
    original_stdout = sys.stdout
    logger = Logger(log_file, original_stdout)
    sys.stdout = logger

    ##### PARSE OPTIONAL ARGUMENTS #####
    def parse_parameter(parameter_name:str, parameters:dict):
        try: 
            setattr(ATAF, parameter_name, parameters[parameter_name])
            print(f'Custom value for {parameter_name} specified: {parameters[parameter_name]}')
        except Exception as e: 
            print(f'Failed to parse {parameter_name}')

    for parameter in parameters: 
        parse_parameter(parameter, parameters)

    for parameter in ['NUM_EPOCHS', 'BATCH_SIZE', 'LEARNING_RATE', 'DROPOUT']:
        print(f'Running component with: {parameter} = {getattr(ATAF, parameter)}')

    ##### LOAD DATA FILE AND METADATA FROM DATA PROCESSING COMPONENT #####
    DEPLOY_DECISION = False
    file_path = dataset.metadata['training_data_path']
    # fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
    # with fs.open(file_path, 'rb') as f:
    #     data = pickle.load(f)
    print(f'======== LOADING DATA FROM {file_path} ========')
    data = pd.read_pickle(file_path)
    last_trade_date = dataset.metadata['last_trade_date']
    num_features_for_each_trade_in_history = dataset.metadata['num_features_for_each_trade_in_history']
    print(f'======== DATASET LOADED FROM {file_path} ========')
    
    ##### FUNCTIONS #####
    def apply_exclusions(data: pd.DataFrame, dataset_name: str = None):
        from_dataset_name = f' from {dataset_name}' if dataset_name is not None else ''
        data_before_exclusions = data[:]
        
        previous_size = len(data)
        data = data[(data.days_to_call == 0) | (data.days_to_call > np.log10(400))]
        current_size = len(data)
        if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having 0 < days_to_call <= 400')
        
        previous_size = current_size
        data = data[(data.days_to_refund == 0) | (data.days_to_refund > np.log10(400))]
        current_size = len(data)
        if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having 0 < days_to_refund <= 400')
        
        previous_size = current_size
        data = data[(data.days_to_maturity == 0) | (data.days_to_maturity > np.log10(400))]
        current_size = len(data)
        if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having 0 < days_to_maturity <= 400')
        
        previous_size = current_size
        data = data[data.days_to_maturity < np.log10(30000)]
        current_size = len(data)
        if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having days_to_maturity >= 30000')
        
        return data, data_before_exclusions

    ##### TRAIN MODEL #####

    current_datetime = datetime.now(EASTERN)
    #This block of code ensures that the model always has data to train on 
    if current_date is None:
        # For when we want to train the model on the current date
        current_date = current_datetime.date().strftime(YEAR_MONTH_DAY)
        previous_business_date = get_trade_date_where_data_exists_on_this_date(decrement_business_days(current_date, 1), data)    # ensures that the business day has trades on it
    else:
        # For when we want to retroactively train a model on a given date 
        print(f'Using the argument when calling the script as the current date: {current_date}')
        previous_business_date = get_trade_date_where_data_exists_on_this_date(decrement_business_days(current_date, 1), data)    # ensures that the business day has trades on it
        last_trade_date = get_trade_date_where_data_exists_on_this_date(decrement_business_days(previous_business_date, 1), data)    # ensures that the business day has trades on it

    current_date_model, test_data_date, previous_business_date_model, previous_business_date_model_date, encoders, mae, mae_df_list, email_intro_text = train_model(data = data, 
                                                                                                                                                                    last_trade_date = last_trade_date, 
                                                                                                                                                                    model= model, 
                                                                                                                                                                    num_features_for_each_trade_in_history = num_features_for_each_trade_in_history, 
                                                                                                                                                                    date_for_previous_model = previous_business_date, 
                                                                                                                                                                    exclusions_function = apply_exclusions)
    if mae_df_list is not None:
        current_date_data_current_date_model_result_df, current_date_data_previous_business_date_model_result_df = mae_df_list
        try:
            business_date_before_test_data_date = get_trade_date_where_data_exists_on_this_date(decrement_business_days(test_data_date, 1), data)    # ensures that the business day has trades on it
            assert previous_business_date_model is not None, f'Raising an AssertionError since previous_business_date_model is `None`, which will run the cleanup logic in the `except` clause'
            business_date_before_test_data_date_data_previous_business_date_model_result_df = get_model_results(data, business_date_before_test_data_date, model, previous_business_date_model, encoders, apply_exclusions)
        except Exception as e:
            print(f'Unable to create the third dataframe used in the model evaluation email due to {type(e)}: {e}')
            print('Stack trace:')
            print(traceback.format_exc())
            business_date_before_test_data_date = None
            business_date_before_test_data_date_data_previous_business_date_model_result_df = None

    if current_date_model is None:
        if not TESTING:
            send_no_new_model_email(last_trade_date, EMAIL_RECIPIENTS, model)
        raise RuntimeError(f'No new data was found for {model} training, so the procedure is terminating gracefully and without issue. Raising an error so that pipeline correctly fails.')
    else:
        try:
            mae_df_list = [current_date_data_current_date_model_result_df, current_date_data_previous_business_date_model_result_df, business_date_before_test_data_date_data_previous_business_date_model_result_df]
            
            current_date_data_current_date_model_mae = current_date_data_current_date_model_result_df.loc['Investment Grade', 'Mean Absolute Error'] 
            current_date_data_previous_business_date_model_mae = current_date_data_previous_business_date_model_result_df.loc['Investment Grade', 'Mean Absolute Error']
            
            if current_date_data_current_date_model_mae <= current_date_data_previous_business_date_model_mae:                                
                DEPLOY_DECISION = True

            print(f'Deploy decision: {DEPLOY_DECISION}\nPrevious MAE: {current_date_data_previous_business_date_model_mae:.3f}, Current MAE: {current_date_data_current_date_model_mae:.3f}')    

            description_list = [f'The below table shows the accuracy of the newly trained {model} model for the trades that occurred on {test_data_date}.', 
                                f'The below table shows the accuracy of the {model} model trained on {previous_business_date_model_date} which was the one deployed on {previous_business_date_model_date} for the trades that occurred on {test_data_date}. If there are three tables in this email, then this one evaluates on the same test dataset as the first table but with a different (previous business day) model. If the accuracy on this table is better than the first table, this may imply that the older model is more accurate. Note, however, that the model has not been (and, cannot be) evaluated yet on the trades that will occur today.', 
                                f'The below table shows the accuracy of the {model} model trained on {previous_business_date_model_date} which was the one deployed on {previous_business_date_model_date} for the trades that occurred on {business_date_before_test_data_date}. If there are three tables in this email, then this one evaluates the same model as the second table but on a different (previous business day) test dataset. If the accuracy on this table is better than the second table, this may mean that the trades in the test set used for the first two tables are more challenging (harder to predict) than the trades from the test set used for this table.']
            mae_df_list, description_list = list(zip(*[(mae_df, description) for (mae_df, description) in zip(mae_df_list, description_list) if mae_df is not None]))    # only keep the (`mae_df`, `description`) pair if the `mae_df` is not None, and then put them into separate lists
            send_results_email_multiple_tables(mae_df_list, description_list, current_date, EMAIL_RECIPIENTS, model, email_intro_text)
        except Exception as e:
            print(f'{type(e)}:', e)

    ##### SAVE MODEL TO ARTIFACT #####
    model_artifact = Model(uri=dsl.get_uri())
    model_path = os.path.join(model_artifact.uri, 'model')
    log_path = os.path.join(model_artifact.uri, log_file)
    
    print(f'======== SAVING MODEL TO {model_path} ========')
    current_date_model.save(model_path)
    print(f'======== MODEL SAVED to {model_path}. EXITING MODEL TRAINING COMPONENT ========')

    metadata = {'mae':mae,
                'current_date_data_current_date_model_result_df': current_date_data_current_date_model_result_df.to_json(orient='index'),
                'current_date_data_previous_business_date_model_result_df': current_date_data_previous_business_date_model_result_df.to_json(orient='index'),
                'model_path': model_path,
                'train_start': data.trade_date.min().strftime('%Y-%m-%d'), 
                'train_end': last_trade_date,
                'test_date': test_data_date,
                'parameters': {'BATCH_SIZE':ATAF.BATCH_SIZE,
                            'NUM_EPOCHS':ATAF.NUM_EPOCHS,
                            'DROPOUT':ATAF.DROPOUT},
                'log_path':log_path,
                'deploy_decision': DEPLOY_DECISION
                }

    for k in metadata: 
        model_artifact.metadata[k] = metadata[k]

    print(f'======== COMPONENT SUMMARY: ========')
    print('Information about the artifact')
    print('Name:', model_artifact.name)
    print('URI:', model_artifact.uri)
    print('Path:', model_artifact.path)
    print('Metadata:', model_artifact.metadata)

    # Close logging 
    
    print(f'===== Saving logs to {log_path} ======')
    logger.close()
    sys.stdout = original_stdout

    remove_lines_with_character('', log_file)
    
    if log_path.startswith('gs://'):
        fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
        with fs.open(log_path, 'wb') as gcs_file:
            with open(log_file, 'rb') as local_file:
                gcs_file.write(local_file.read())
            
    return model_artifact
