'''
Description: This script deletes old models from the Vertex AI model registry that are not deployed to any endpoints.
'''
from zoneinfo import ZoneInfo    # similar to `pytz` from older versions (< 3.9) of Python 
from datetime import datetime, timedelta

from google.cloud import aiplatform


YEAR_MONTH_DAY_HOUR_MIN_SEC = "%Y-%m-%d %H:%M:%S"

REGION = "us-east4"
PROJECT_ID = 'eng-reactor-287421'
DAYS_OLD = 60    # keep models newer than this many days


TESTING = False
if TESTING:
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/user/ficc/ficc/mitas_creds.json'


def convert_to_eastern_time(dt: datetime) -> datetime:
    '''Assumes `dt` is in UTC.'''
    return dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))


def delete_old_models(request):
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Get all models in region
    models = aiplatform.Model.list()

    # Get all endpoints with deployed models
    endpoints = aiplatform.Endpoint.list()
    deployed_model_ids = set()
    for endpoint in endpoints:
        for deployed in endpoint.list_models():
            deployed_model_ids.add(deployed.model.split("/")[-1])

    cutoff_datetime = datetime.now() - timedelta(days=DAYS_OLD)
    cutoff_datetime_eastern = convert_to_eastern_time(cutoff_datetime)    # used only for print

    num_models_deleted = 0
    prefix = "TESTING (so no actual deletion): " if TESTING else ""
    print(f'{prefix}Begin deletion of undeployed models in Vertex AI Model Registry in {REGION} that are older than {cutoff_datetime_eastern.strftime(YEAR_MONTH_DAY_HOUR_MIN_SEC)} eastern ({DAYS_OLD} days before the current datetime)')
    for model in models:
        model_id = model.resource_name.split("/")[-1]
        model_datetime = model.create_time.replace(tzinfo=None)

        if model_datetime < cutoff_datetime and model_id not in deployed_model_ids:
            model_datetime_eastern = convert_to_eastern_time(model_datetime)    # used only for print
            print(f"{prefix}Deleting {model.display_name} (ID: {model_id}) created on {model_datetime_eastern.strftime(YEAR_MONTH_DAY_HOUR_MIN_SEC)} eastern")
            if not TESTING: model.delete()
            num_models_deleted += 1
        else:
            recent_text = f'recent (within the last {DAYS_OLD} days)'
            if model_id in deployed_model_ids and model_datetime >= cutoff_datetime:
                suffix = f'{recent_text} and deployed (in either case, not deleting)'
            elif model_id in deployed_model_ids:
                suffix = 'deployed'
            else:
                suffix = recent_text
            print(f"Skipping {model.display_name} (ID: {model_id}) because it is {suffix}")

    return f"Succesfully deleted {num_models_deleted} models", 200
