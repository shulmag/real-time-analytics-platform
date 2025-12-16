'''
'''
from google.cloud import bigquery


def get_bq_client():
    '''Initialize the credentials and the bigquery client.'''
    return bigquery.Client()


def sqltodf(sql, bq_client):
    '''This function comes directly from `modules.ficc.utils.auxiliary_functions`.'''
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()