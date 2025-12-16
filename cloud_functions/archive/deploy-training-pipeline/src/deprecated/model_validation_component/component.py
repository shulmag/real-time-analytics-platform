'''
 # @ Create date: 2024-02
 # @ Modified date: 2024-02-06
 '''
  
from kfp import dsl
from kfp.dsl import Model, Dataset, Artifact

project_id = 'eng-reactor-287421'

@dsl.component(base_image=f'gcr.io/{project_id}/data-processing-base:test', \
               target_image=f'gcr.io/{project_id}/model-validation-component:v1')
def model_validation_component(TESTING:bool,
                               model_artifact:Model,
                               window_size:int,
                               sd_multiple:float=1.0) -> Artifact:

    '''Kubeflow component that checks whether or not to deploy a model

    Args:
        X (kfp.dsl.Y): 

    Returns:    
        artifact (kfp.dsl.Artifact): artifact object  
        
    '''
    import sys
    import numpy as np
    import pandas as pd
    import gcsfs
    from ficc.utils.auxiliary_functions import function_timer
    from ficc.utils.auxiliary_variables import PREDICTORS, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES
    from datetime import datetime
    from pandas.tseries.offsets import BDay
    from automated_training_auxiliary_functions import TESTING, \
                                                    EMAIL_RECIPIENTS, \
                                                    get_storage_client, \
                                                    get_bq_client, \
                                                    sqltodf

    import automated_training_auxiliary_functions as ATAF
    import os 
    from logger import Logger
    
    if TESTING: 
        print('===== Pipeline deployed in testing mode =====')

    ##### SETUP LOG FILE #####
    log_file = f'model-validation-component-log_{datetime.now().strftime("%Y-%m-%d")}.log' 
    original_stdout = sys.stdout
    logger = Logger(log_file, original_stdout)
    sys.stdout = logger
    ##### SETUP LOG FILE #####

    ATAF.TESTING = TESTING
    ATAF.get_creds = lambda: None
    ##########
    BQ_CLIENT = get_bq_client()

    #need to specify project_id if there are no creds provided 
    #we do not want creds to be in the container for security reasons!
    BQ_CLIENT.project = project_id

    HISTORICAL_PREDICTION_TABLE = 'eng-reactor-287421.historic_predictions.historical_predictions'

    @function_timer
    def get_historical_mae():
        QUERY = f'''
        WITH
            training_trades AS (
            SELECT
                rtrs_control_number,
                CASE
                WHEN recent[SAFE_OFFSET(0)].calc_date IS NULL THEN TRUE
                ELSE
                FALSE
            END
                empty_history,
                cusip,
                yield,
                trade_date,
                settlement_date,
                maturity_date,
                next_call_date,
                refund_date,
                COALESCE(DATE_DIFF(maturity_date, settlement_date, DAY), 0) AS days_to_maturity,
                COALESCE(DATE_DIFF(next_call_date, settlement_date, DAY),0) AS days_to_call,
                COALESCE(DATE_DIFF(refund_date, settlement_date, DAY),0) AS days_to_refund,
            FROM
                `eng-reactor-287421.auxiliary_views.materialized_trade_history`
            WHERE
                yield IS NOT NULL
                AND yield > 0
                AND par_traded >= 10000
                AND coupon_type IN (8,
                4,
                10,
                17)
                AND capital_type <> 10
                AND default_exists <> TRUE
                AND most_recent_default_event IS NULL
                AND default_indicator IS FALSE
                AND msrb_valid_to_date > current_date -- condition TO remove cancelled trades
                AND settlement_date IS NOT NULL),
            training_trades_exclude_short_term AS (
            SELECT
                *
            FROM
                training_trades
            WHERE
                ((training_trades.days_to_maturity>400)
                OR (training_trades.days_to_maturity=0))
                AND ((training_trades.days_to_refund>400)
                OR (training_trades.days_to_refund=0))
                AND ((training_trades.days_to_call>400)
                OR (training_trades.days_to_call=0))
                AND training_trades.days_to_maturity<30000
                AND NOT empty_history )
            SELECT
                DISTINCT *,
                STDDEV(MAE) OVER (ORDER BY UNIX_DATE(trade_date) RANGE BETWEEN {window_size-1} PRECEDING AND CURRENT ROW) AS SD
            FROM (
            SELECT
                trade_date,
                AVG(ABS(new_ys-new_ys_prediction)) OVER (ORDER BY UNIX_DATE(trade_date) RANGE BETWEEN {window_size-1} PRECEDING
                AND CURRENT ROW) AS MAE,
                COUNT(*) OVER (ORDER BY UNIX_DATE(trade_date) RANGE BETWEEN {window_size-1} PRECEDING
                AND CURRENT ROW) AS N
            FROM
                `historic_predictions.historical_predictions`
            WHERE
                rtrs_control_number IN (
                SELECT
                rtrs_control_number
                FROM
                training_trades_exclude_short_term)
            AND DATE_DIFF(CURRENT_DATE(), trade_date , DAY) <= {2*window_size}
            )
            ORDER BY
            trade_date DESC  
            '''
        return sqltodf(bq_client=BQ_CLIENT, sql=QUERY)

    mae_df = get_historical_mae()
    print(mae_df)

    sd = mae_df['SD'].iloc[0]
    mae_threshold = mae_df['MAE'].iloc[0] + sd_multiple*sd
    deploy_decision = False

    model_mae = model_artifact.metadata['mae']

    if (model_mae <= mae_threshold):
        deploy_decision = True 
        print(f'Model MAE with exclusions is {model_mae}, threshold is {mae_threshold} based on a rolling window size of {window_size} days. Model is ready to be deployed.')
    else:
        print(f'Model MAE with exclusions is {model_mae}, threshold is {mae_threshold} based on a rolling window size of {window_size} days. Model is not ready to be deployed.')

    # Close logging 
    logger.close()
    sys.stdout = original_stdout


    artifact = Artifact(uri = dsl.get_uri())
    log_path = os.path.join(artifact.uri, log_file)
    metadata = {'deploy_decision': deploy_decision, 
                    'window_size': window_size,
                    'mae_threshold':mae_threshold,
                    'log_path':log_path}
    
    for k in metadata: 
        artifact.metadata[k] = metadata[k]

    print('Information about the artifact')
    print('Name:', artifact.name)
    print('URI:', artifact.uri)
    print('Path:', artifact.path)
    print('Metadata:', artifact.metadata)

    # Close logging 
    fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
    with fs.open(log_path, 'wb') as gcs_file:
        with open(log_file, 'rb') as local_file:
            gcs_file.write(local_file.read())
    print(f'===== Logs saved to {log_path} ======')
    
    return artifact