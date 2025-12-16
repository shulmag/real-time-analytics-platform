import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="Cusip Global Service Importer-2fdcdfc4edba.json"

import paramiko
import os
import os.path
import tempfile 
import json
from time import sleep


from datetime import datetime,timedelta
from pytz import timezone
from google.cloud import bigquery
from send_email import send_error_email
from google.cloud import secretmanager
from google.api_core.exceptions import BadRequest

# Construct a BigQuery client object.
bq_client = bigquery.Client()
# ET time zone: 
eastern = timezone('US/Eastern')

def access_secret_version(project_id, secret_id, version_id):
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()
    # Build the resource name of the secret version.
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    # Access the secret version.
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    return(payload)


def get_table_name_from_pip_file_name(file_name):
    return(file_name.split('.PIP')[0].replace('-','_'))

def get_file_name_from_table_name(table_name):
    return(table_name.replace('_','-') + '.PIP')

#returns list of new files since last upload:
def changemon(existing_tables,sftp):
    ls_prev = set(existing_tables)
    files = sftp.listdir()
    file_list = [get_table_name_from_pip_file_name(file_name) for file_name in files]
    ls = set(file_list)
    add = ls-ls_prev
    if add:
        return(add)
        
#this function load a single file into BQ: 
def load_e_pip_into_bq_table(file_name,tmp_file):

    #Create schema for E file
    cusip_schema = [bigquery.SchemaField("issuer_num","string"),	bigquery.SchemaField("issue_num","string"),	bigquery.SchemaField("issue_check","string"),	bigquery.SchemaField("issue_description","string"),	bigquery.SchemaField("issue_additional_info","string"),	bigquery.SchemaField("issue_status","string"),	bigquery.SchemaField("issue_type_code","string"),	bigquery.SchemaField("dated_date","date"),	bigquery.SchemaField("maturity_date","date"),	bigquery.SchemaField("partial_maturity","string"),	bigquery.SchemaField("coupon__rate","numeric"),	bigquery.SchemaField("currency_code","string"),	bigquery.SchemaField("security_type_description","string"),	bigquery.SchemaField("fisn","string"),	bigquery.SchemaField("issue_group","string"),	bigquery.SchemaField("isin","string"),	bigquery.SchemaField("where_traded","string"),	bigquery.SchemaField("ticker_symbol","string"),	bigquery.SchemaField("us_cfi_code","string"),	bigquery.SchemaField("iso_cfi_code","string"),	bigquery.SchemaField("issue_entry_date","date"),	bigquery.SchemaField("alternative_minimum_tax","string"),	bigquery.SchemaField("bank_qualified","string"),	bigquery.SchemaField("callable","string"),	bigquery.SchemaField("first_coupon_date","date"),	bigquery.SchemaField("initial_public_offering","string"),	bigquery.SchemaField("payment_frequency_code","string"),	bigquery.SchemaField("closing_date","date"),	bigquery.SchemaField("dtc_eligible","string"),	bigquery.SchemaField("pre_refunded","string"),	bigquery.SchemaField("refundable","string"),	bigquery.SchemaField("remarketed","string"),	bigquery.SchemaField("sinking_fund","string"),	bigquery.SchemaField("taxable","string"),	bigquery.SchemaField("bond_form","string"),	bigquery.SchemaField("enhancements","string"),	bigquery.SchemaField("fund_distribution_policy","string"),	bigquery.SchemaField("fund_investment_policy","string"),	bigquery.SchemaField("fund_type","string"),	bigquery.SchemaField("guarantee","string"),	bigquery.SchemaField("income_type","string"),	bigquery.SchemaField("insured_by","string"),	bigquery.SchemaField("ownership_restrictions","string"),	bigquery.SchemaField("payment_status","string"),	bigquery.SchemaField("preferred_type","string"),	bigquery.SchemaField("putable","string"),	bigquery.SchemaField("rate_type","string"),	bigquery.SchemaField("redemption","string"),	bigquery.SchemaField("source_document","string"),	bigquery.SchemaField("sponsoring","string"),	bigquery.SchemaField("voting_rights","string"),	bigquery.SchemaField("warrant_assets","string"),	bigquery.SchemaField("warrant_status","string"),	bigquery.SchemaField("warrant_type","string"),	bigquery.SchemaField("underwriter","string"),	bigquery.SchemaField("auditor","string"),	bigquery.SchemaField("paying_agent","string"),	bigquery.SchemaField("tender_agent","string"),	bigquery.SchemaField("transfer_agent","string"),	bigquery.SchemaField("bond_counsel","string"),	bigquery.SchemaField("financial_advisor","string"),	bigquery.SchemaField("municipal_sale_date","date"),	bigquery.SchemaField("sale_type","string"),	bigquery.SchemaField("offering_amount","numeric"),	bigquery.SchemaField("offering_amount_code","string"),	bigquery.SchemaField("issue_transaction","string"),	bigquery.SchemaField("issue_last_update_date","date"),	bigquery.SchemaField("obligor_name","string"),	bigquery.SchemaField("obligor_cusip_issuer_number","string"),	bigquery.SchemaField("co_obligor_name","string"),	bigquery.SchemaField("co_obligor_cusip_issuer_number","string"),	bigquery.SchemaField("government_stimulus_program","string"),	bigquery.SchemaField("reserved_6","string"),	bigquery.SchemaField("reserved_7","string"),	bigquery.SchemaField("reserved_8","string"),	bigquery.SchemaField("reserved_9","string"),	bigquery.SchemaField("reserved_10","string")]
    #Set table_id to the ID of the table to create.
    table_id = "eng-reactor-287421.cusip_global_services." + file_name[:-4].replace('-','_')
    
    job_config = bigquery.LoadJobConfig(
        schema=cusip_schema,
        field_delimiter="|",
        maxBadRecords=5,
        allowJaggedRows=True,
        ignore_unknown_values=True
    )
    tmp_file.seek(0)
    load_job = bq_client.load_table_from_file(tmp_file, table_id, job_config=job_config)# Make an API request.
    from google.api_core.exceptions import BadRequest

    try:
        result = load_job.result()  # Waits for the job to complete.
    except BadRequest as ex:
        send_error_email('failed to load cusip issues daily updates',ex.message)
            
def update_cusip_master(daily_table):    
    # simple non parameterized query
    client = bigquery.Client()
    query = """
        MERGE cusip_global_services.cusip_issues_master AS target
        USING """ + daily_table + """ AS source
        ON source.issuer_num = target.issuer_num AND source.issue_num = target.issue_num AND source.issue_check = target.issue_check
        WHEN MATCHED THEN
          UPDATE SET issuer_num=source.issuer_num,	issue_num=source.issue_num,	issue_check=source.issue_check,	issue_description=source.issue_description,	issue_additional_info=source.issue_additional_info,	issue_status=source.issue_status,	issue_type_code=source.issue_type_code,	dated_date=source.dated_date,	maturity_date=source.maturity_date,	partial_maturity=source.partial_maturity,	coupon__rate=source.coupon__rate,	currency_code=source.currency_code,	security_type_description=source.security_type_description,	fisn=source.fisn,	issue_group=source.issue_group,	isin=source.isin,	where_traded=source.where_traded,	ticker_symbol=source.ticker_symbol,	us_cfi_code=source.us_cfi_code,	iso_cfi_code=source.iso_cfi_code,	issue_entry_date=source.issue_entry_date,	alternative_minimum_tax=source.alternative_minimum_tax,	bank_qualified=source.bank_qualified,	callable=source.callable,	first_coupon_date=source.first_coupon_date,	initial_public_offering=source.initial_public_offering,	payment_frequency_code=source.payment_frequency_code,	closing_date=source.closing_date,	dtc_eligible=source.dtc_eligible,	pre_refunded=source.pre_refunded,	refundable=source.refundable,	remarketed=source.remarketed,	sinking_fund=source.sinking_fund,	taxable=source.taxable,	bond_form=source.bond_form,	enhancements=source.enhancements,	fund_distribution_policy=source.fund_distribution_policy,	fund_investment_policy=source.fund_investment_policy,	fund_type=source.fund_type,	guarantee=source.guarantee,	income_type=source.income_type,	insured_by=source.insured_by,	ownership_restrictions=source.ownership_restrictions,	payment_status=source.payment_status,	preferred_type=source.preferred_type,	putable=source.putable,	rate_type=source.rate_type,	redemption=source.redemption,	source_document=source.source_document,	sponsoring=source.sponsoring,	voting_rights=source.voting_rights,	warrant_assets=source.warrant_assets,	warrant_status=source.warrant_status,	warrant_type=source.warrant_type,	underwriter=source.underwriter,	auditor=source.auditor,	paying_agent=source.paying_agent,	tender_agent=source.tender_agent,	transfer_agent=source.transfer_agent,	bond_counsel=source.bond_counsel,	financial_advisor=source.financial_advisor,	municipal_sale_date=source.municipal_sale_date,	sale_type=source.sale_type,	offering_amount=source.offering_amount,	offering_amount_code=source.offering_amount_code,	issue_transaction=source.issue_transaction,	issue_last_update_date=source.issue_last_update_date,	obligor_name=source.obligor_name,	obligor_cusip_issuer_number=source.obligor_cusip_issuer_number,	co_obligor_name=source.co_obligor_name,	co_obligor_cusip_issuer_number=source.co_obligor_cusip_issuer_number,	government_stimulus_program=source.government_stimulus_program,	reserved_6=source.reserved_6,	reserved_7=source.reserved_7,	reserved_8=source.reserved_8,	reserved_9=source.reserved_9,	reserved_10=source.reserved_10
        WHEN NOT MATCHED BY TARGET THEN
          INSERT (issuer_num,	issue_num,	issue_check,	issue_description,	issue_additional_info,	issue_status,	issue_type_code,	dated_date,	maturity_date,	partial_maturity,	coupon__rate,	currency_code,	security_type_description,	fisn,	issue_group,	isin,	where_traded,	ticker_symbol,	us_cfi_code,	iso_cfi_code,	issue_entry_date,	alternative_minimum_tax,	bank_qualified,	callable,	first_coupon_date,	initial_public_offering,	payment_frequency_code,	closing_date,	dtc_eligible,	pre_refunded,	refundable,	remarketed,	sinking_fund,	taxable,	bond_form,	enhancements,	fund_distribution_policy,	fund_investment_policy,	fund_type,	guarantee,	income_type,	insured_by,	ownership_restrictions,	payment_status,	preferred_type,	putable,	rate_type,	redemption,	source_document,	sponsoring,	voting_rights,	warrant_assets,	warrant_status,	warrant_type,	underwriter,	auditor,	paying_agent,	tender_agent,	transfer_agent,	bond_counsel,	financial_advisor,	municipal_sale_date,	sale_type,	offering_amount,	offering_amount_code,	issue_transaction,	issue_last_update_date,	obligor_name,	obligor_cusip_issuer_number,	co_obligor_name,	co_obligor_cusip_issuer_number,	government_stimulus_program,	reserved_6,	reserved_7,	reserved_8,	reserved_9,	reserved_10) VALUES (source.issuer_num,	source.issue_num,	source.issue_check,	source.issue_description,	source.issue_additional_info,	source.issue_status,	source.issue_type_code,	source.dated_date,	source.maturity_date,	source.partial_maturity,	source.coupon__rate,	source.currency_code,	source.security_type_description,	source.fisn,	source.issue_group,	source.isin,	source.where_traded,	source.ticker_symbol,	source.us_cfi_code,	source.iso_cfi_code,	source.issue_entry_date,	source.alternative_minimum_tax,	source.bank_qualified,	source.callable,	source.first_coupon_date,	source.initial_public_offering,	source.payment_frequency_code,	source.closing_date,	source.dtc_eligible,	source.pre_refunded,	source.refundable,	source.remarketed,	source.sinking_fund,	source.taxable,	source.bond_form,	source.enhancements,	source.fund_distribution_policy,	source.fund_investment_policy,	source.fund_type,	source.guarantee,	source.income_type,	source.insured_by,	source.ownership_restrictions,	source.payment_status,	source.preferred_type,	source.putable,	source.rate_type,	source.redemption,	source.source_document,	source.sponsoring,	source.voting_rights,	source.warrant_assets,	source.warrant_status,	source.warrant_type,	source.underwriter,	source.auditor,	source.paying_agent,	source.tender_agent,	source.transfer_agent,	source.bond_counsel,	source.financial_advisor,	source.municipal_sale_date,	source.sale_type,	source.offering_amount,	source.offering_amount_code,	source.issue_transaction,	source.issue_last_update_date,	source.obligor_name,	source.obligor_cusip_issuer_number,	source.co_obligor_name,	source.co_obligor_cusip_issuer_number,	source.government_stimulus_program,	source.reserved_6,	source.reserved_7,	source.reserved_8,	source.reserved_9,	source.reserved_10)
        ;
    """
    query_res = client.query(query)  # Make an API request.
    try:
        result = query_res.result()  # Waits for the job to complete.
    except BadRequest as ex:
        send_error_email('failed to merge cusip issues daily updates',ex.message)

def get_issues_file_by_date(date,sftp):
    file_name = 'ACUD{}E.PIP'.format(date)
    return(file_name)

def most_recent_business_day():
    today = datetime.now(eastern)
    offset = max(1, (today.weekday() + 6) % 7 - 3)
    most_recent = today - timedelta(offset)
    return(most_recent)

def main(mmdd):
    host = 'edx.standardandpoors.com'
    port = 22
    user = 'Ficc6333'
    pwd = '4dquWX$2'

    transport = paramiko.Transport((host, port))
    transport.connect(username = user, password = pwd)
    sftp = paramiko.SFTPClient.from_transport(transport)

    #verify we are in the correct CUSIP directory: 
    if sftp.getcwd() != '/Inbox':sftp.chdir('.\Inbox') 

    #tables = set([t.table_id for t in bq_client.list_tables('eng-reactor-287421.cusip_global_services')])
    date = most_recent_business_day().strftime('%m-%d')
    file_name = get_issues_file_by_date(date,sftp)
    
    extension = os.path.splitext(file_name)[1]
    if(file_name.find('E.PIP') != -1 and extension == '.PIP' and file_name != 'ACUMASTER-ISSUE.PIP'): 
        with sftp.open(file_name) as pip:
            cusips_text = pip.read().decode('utf-8')
            clean = cusips_text.split('999999', 1)[0]    
            temp = tempfile.NamedTemporaryFile() 
            temp.write(clean.encode())   
            load_e_pip_into_bq_table(file_name,temp)
            update_cusip_master('cusip_global_services.' + get_table_name_from_pip_file_name(file_name))
            bq_client.delete_table('eng-reactor-287421.cusip_global_services.' + get_table_name_from_pip_file_name(file_name))
    sftp.close()
    transport.close()
        
    return 'success'

    

''' DELETE ALL TABLES:
from datetime import datetime, timedelta
start_date = datetime.strptime('10-03', '%m-%d').date()
end_date = datetime.strptime('10-14', '%m-%d').date()
while start_date <= end_date:
    date = start_date.strftime('%m-%d')
    start_date = start_date+timedelta(days=1)
    file_name = 'ACUD{}E.PIP'.format(date)
    try:
        main(date)
        print(date)
        #bq_client.delete_table('eng-reactor-287421.cusip_global_services.' + get_table_name_from_pip_file_name(file_name))
    except Exception as z:
        pass
'''

