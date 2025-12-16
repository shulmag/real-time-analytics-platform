# -*- coding: utf-8 -*-
# @Date:   2021-11-18 15:12:36

import ficc_globals as globals
from google.cloud import bigquery
import pandas as pd

def sqltodf(sql,bq_client):
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()

def yield_curve_params(client):
    #The following fetches Nelson-Siegel coefficient and standard scalar parameters from BigQuery and sends them to a dataframe. 
    globals.nelson_params = sqltodf("select * from `eng-reactor-287421.yield_curves.nelson_siegel_coef_daily` order by date desc",client)
    globals.scalar_params = sqltodf("select * from`eng-reactor-287421.yield_curves.standardscaler_parameters_daily` order by date desc",client)


    #The below sets the index of both dataframes to date column and converts the data type to datetime. 
    globals.nelson_params.set_index("date",drop=True,inplace=True)
    globals.scalar_params.set_index("date",drop=True,inplace=True)
    globals.scalar_params.index = pd.to_datetime(globals.scalar_params.index)
    globals.nelson_params.index = pd.to_datetime(globals.nelson_params.index)
    
