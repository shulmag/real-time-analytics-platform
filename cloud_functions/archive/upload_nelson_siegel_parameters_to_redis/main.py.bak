import pandas as pd
import numpy as np
import redis
import pickle5 as pickle
from google.cloud import bigquery


PROJECT_ID = "eng-reactor-287421"
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/jupyter/ficc/ml_models/sequence_predictors/ahmad_creds.json"
bq_client = bigquery.Client()

def sqltodf(sql, bq_client):
  bqr = bq_client.query(sql).result()
  return bqr.to_dataframe()


def grab_nelson_siegel_parameter():
  nelson_params_daily = sqltodf("select * from `eng-reactor-287421.yield_curves_v2.nelson_siegel_coef_daily` order by date desc", bq_client)
  nelson_params_daily.set_index("date", drop=True, inplace=True)
  nelson_params_daily = nelson_params_daily[~nelson_params_daily.index.duplicated(keep='first')]

  # nelson_params = sqltodf("select * from `eng-reactor-287421.yield_curves_v2.nelson_siegel_coef_minute` order by date desc", bq_client)
  nelson_params = sqltodf("Select * from `eng-reactor-287421.yield_curves_v2.nelson_siegel_coef_minute` where date < '2022-06-27' or date > '2022-06-28' order by date asc;", bq_client)
  nelson_params.set_index("date", drop=True, inplace=True)
  nelson_params = nelson_params[~nelson_params.index.duplicated(keep='first')]

  nelson_params_temp = sqltodf("select * from `eng-reactor-287421.yield_curves_v2.nelson_siegel_coef_minute_temp` order by date desc", bq_client)
  nelson_params_temp.set_index("date", drop=True, inplace=True)
  nelson_params_temp = nelson_params_temp[~nelson_params_temp.index.duplicated(keep='first')]

  nelson_params.loc['2022-06-13'] = nelson_params_temp.loc['2022-06-13']
  nelson_params.loc['2022-09-26'] = nelson_params_temp.loc['2022-09-26']
  nelson_params = pd.concat([nelson_params, nelson_params_temp.loc['2022-06-27']], ignore_index=False)

  nelson_params =nelson_params.sort_index(ascending=False)

  return nelson_params, nelson_params_daily

def grab_standard_scalar_parameter():
  scalar_params = sqltodf("select * from`eng-reactor-287421.yield_curves_v2.standardscaler_parameters_daily` order by date desc", bq_client)
  scalar_params.set_index("date", drop=True, inplace=True)
  scalar_params = scalar_params[~scalar_params.index.duplicated(keep='first')]

  return scalar_params

def grab_shape_parameter():
  shape_parameter  = sqltodf("SELECT *  FROM `eng-reactor-287421.yield_curves_v2.shape_parameters` order by Date desc", bq_client)
  shape_parameter.set_index("Date", drop=True, inplace=True)
  shape_parameter = shape_parameter[~shape_parameter.index.duplicated(keep='first')]

  return shape_parameter

def main(args):
  redis_client = redis.Redis(host='10.227.69.60', port=6379, db=0)
  
  nelson_params, nelson_params_daily = grab_nelson_siegel_parameter()
  scalar_params = grab_standard_scalar_parameter()
  shape_parameter = grab_shape_parameter()

  redis_client.set('ns_daily',pickle.dumps(nelson_params_daily))
  redis_client.set('ns_minute',pickle.dumps(nelson_params))
  redis_client.set('scalar_params', pickle.dumps(scalar_params))
  redis_client.set('shape_parameter', pickle.dumps(shape_parameter))

  return "SUCCESS"

