# 2025-09-15
# Last Edited by Developer
# 2025-09-16

It runs only on valid U.S. business days, computes the most actively traded issues and total trade volumes, and updates the results in the following bq tables:
eng-reactor-287421.analytics_data_source.minute_top_issues
eng-reactor-287421.analytics_data_source.minute_trade_count.


# Deployment
The Cloud Run Function may be deployed by the following command:

gcloud functions deploy update-trade-count-top-issues\
   --gen2\
   --region=us-central1\
   --runtime=python311\
   --source=.\
   --entry-point=main\
   --trigger-http\
   --timeout=540\
   --min-instances=0\
   --max-instances=1