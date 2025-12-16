import os
from io import StringIO, BytesIO
import pandas as pd 
import numpy as np
from lxml import etree
from google.cloud import bigquery
import pytz
from datetime import datetime
from google.api_core.exceptions import BadRequest



#Setting up the global variables to be used.
my_timezone = pytz.timezone('America/New_York')


# Main change will hapen here as the file should be read either from the FTP or cloud storage
path = "../../gsm_init_muni_APFICC_GSMF10I.1.1_1.20201203T1300-05.xml"


# Project, data and table configuration
PROJECT = 'eng-reactor-287421'
dataset = 'FULL_ICE'
table = 'reference_data'

"""Schema information in the form of list, each list below is used for changing the type of the column,
once the dataframe is created. """
final_columns = ['instrument_id',
 'data_timestamp',
 'apex_asset_type',
 'instrument_type',
 'primary_name',
 'delivery_date',
 'issue_price',
 'settlement_date',
 'issue_date',
 'outstanding_indicator',
 'primary_currency_code',
 'child_issue_ind',
 'federal_tax_status',
 'sector',
 'sector_class',
 'category',
 'sub_category',
 'Moodys_Long_Rating',
 'Moodys_Long_Rating_effective_date',
 'Moodys_Credit_Watch_Long_Rating',
 'Moodys_Credit_Watch_Long_Rating_effective_date',
 'Moodys_Issue_Long_Rating',
 'Moodys_Issue_Long_Rating_effective_date',
 'country_code',
 'maturity_amount',
 'denom_increment_amount',
 'min_denom_amount',
 'organization_master_id',
 'issuerissuer',
 'issuer_incorporated_state_code',
 'issuer_primary_name',
 'issuer_primary_name_abbreviated',
 'issuer_organization_status',
 'issuer_organization_type',
 'issuer_org_country_code',
 'id_bb_sec_num',
 'market_sector',
 'security_typ',
 'ticker',
 'unique_id',
 'accrual_date',
 'bond_form',
 'bond_insurance',
 'coupon_type',
 'current_coupon_rate',
 'daycount_basis_type',
 'debt_type',
 'default_indicator',
 'depository_type',
 'first_coupon_date',
 'interest_payment_frequency',
 'issue_amount',
 'last_period_accrues_from_date',
 'maturity_date',
 'next_coupon_payment_date',
 'orig_principal_amount',
 'original_yield',
 'outstanding_amount',
 'sale_type',
 'settlement_type',
 'principal_factor',
 'principal_factorT1',
 'principal_factorT2',
 'principal_factorT3',
 'principal_factorT4',
 'bank_qualified',
 'capital_type',
 'DTCC_status',
 'first_execution_date',
 'formal_award_date',
 'issue_key',
 'issue_text',
 'maturity_description_code',
 'muni_security_type',
 'project_name',
 'sale_date',
 'series_name',
 'use_of_proceeds',
 'purpose_class',
 'purpose_sub_class',
 'extraordinary_make_whole_call',
 'extraordinary_redemption',
 'make_whole_call',
 'mandatory_sink_amount_type',
 'amount_outstanding_change_date',
 'amount_outstanding',
 'amount_outstanding_decimal',
 'CUSIP',
 'ISIN',
 'Bloomberg_Global_Id',
 'Standard__Poors_Long_Rating',
 'Standard__Poors_Long_Rating_effective_date',
 'Standard__Poors_Credit_Watch_Long_Outlook_Rating',
 'Standard__Poors_Credit_Watch_Long_Outlook_Rating_effective_date',
 'state_tax_status',
 'conduit_obligor_name',
 'primary_exchange',
 'eval_quotation_basis',
 'Moodys_Enhanced_Long_Rating',
 'Moodys_Enhanced_Long_Rating_effective_date',
 'Standard__Poors_School_ICR_Long_Rating',
 'Standard__Poors_School_ICR_Long_Rating_effective_date',
 'previous_coupon_payment_date',
 'other_enhancement_type',
 'primary_market',
 'SEDOL',
 'SEDOL_status',
 'IDII',
 'IDII_status',
 'country_of_quotation',
 'mic',
 'currency_code',
 'Moodys_Short_Rating',
 'Moodys_Short_Rating_effective_date',
 'Moodys_Credit_Watch_Short_Rating',
 'Moodys_Credit_Watch_Short_Rating_effective_date',
 'Moodys_Issue_Short_Rating',
 'Moodys_Issue_Short_Rating_effective_date',
 'Standard__Poors_Short_Rating',
 'Standard__Poors_Short_Rating_effective_date',
 'Fitch_Ratings_Long_Rating',
 'Fitch_Ratings_Long_Rating_effective_date',
 'Fitch_Ratings_Short_Rating',
 'Fitch_Ratings_Short_Rating_effective_date',
 'Fitch_Ratings_Issue_Long_Rating',
 'Fitch_Ratings_Issue_Long_Rating_effective_date',
 'Fitch_Ratings_Issue_Short_Rating',
 'Fitch_Ratings_Issue_Short_Rating_effective_date',
 'call_notice',
 'call_timing',
 'call_timing_in_part',
 'mandatory_redemption_code',
 'next_call_date',
 'next_call_price',
 'optional_redemption_code',
 'par_call_date',
 'par_call_price',
 'maximum_call_notice_period',
 'additional_project_txt',
 'other_accrual_date',
 'Standard__Poors_Credit_Watch_Long_Rating',
 'Standard__Poors_Credit_Watch_Long_Rating_effective_date',
 'secured',
 'asset_claim_code',
 'sink_fund_redemption_method',
 'next_sink_date',
 'sink_frequency',
 'Standard__Poors_Issue_Long_Rating',
 'Standard__Poors_Issue_Long_Rating_effective_date',
 'Moodys_Insured_Long_Rating',
 'Moodys_Insured_Long_Rating_effective_date',
 'Fitch_Ratings_Credit_Watch_Long_Rating',
 'Fitch_Ratings_Credit_Watch_Long_Rating_effective_date',
 'make_whole_benchmark',
 'make_whole_call_end_date',
 'make_whole_call_spread',
 'backed_underlying_security_id',
 'called_redemption_type',
 'orig_instrument_enhancement_type',
 'refunding_dated_date',
 'refunding_issue_key',
 'orig_cusip_status',
 'refund_date',
 'refund_price',
 'muni_issue_type',
 'tax_credit_percent',
 'tax_credit_frequency',
 'mtg_insurance',
 'orig_avg_life_date',
 'use_of_proceeds_supplementary',
 'pac_bond_indicator',
 'state_code',
 'sec_regulation',
 'Moodys_Enhanced_Short_Rating',
 'Moodys_Enhanced_Short_Rating_effective_date',
 'next_put_date',
 'put_end_date',
 'put_feature_price',
 'put_frequency',
 'put_start_date',
 'put_type',
 'current_coupon_reset_date',
 'previous_coupon_effective_date',
 'previous_coupon_rate',
 'previous_rate_reset_frequency',
 'first_variable_reset_date',
 'initial_variable_rate',
 'interest_payment_day',
 'next_reset_date',
 'reset_frequency',
 'variable_rate_ceiling',
 'tender_agent',
 'tender_price',
 'tender_type',
 'loc_bank',
 'loc_expiration_date',
 'loc_percent',
 'loc_type',
 'SIC_primary',
 'NAICS_primary',
 'call_cav',
 'min_notification_days',
 'reset_day',
 'sink_defeased',
 'other_enhancement_company',
 'int_holiday_treatment_method',
 'reset_holiday_treatment_method',
 'cusip_change',
 'hybrid_type',
 'additional_state_code',
 'max_notification_days',
 'clearing_agents']

str_column = ["additional_state_code",]
del_column = ['Moodys_Enhanced_Short_Rating_effective_date','data_timestamp',
 'Moodys_Credit_Watch_Short_Rating',
 'Standard__Poors_School_ICR_Long_Rating',
 'Moodys_Enhanced_Long_Rating_effective_date',
 'Standard__Poors_Short_Rating',
 'Fitch_Ratings_Credit_Watch_Long_Rating',
 'Fitch_Ratings_Issue_Long_Rating_effective_date',
 'Moodys_Long_Rating',
 'Moodys_Enhanced_Long_Rating',
 'Moodys_Issue_Long_Rating_effective_date',
 'Standard__Poors_Short_Rating_effective_date',
 'Standard__Poors_Credit_Watch_Long_Rating_effective_date',
 'Moodys_Insured_Long_Rating',
 'Moodys_Issue_Short_Rating',
 'Standard__Poors_Issue_Long_Rating',
 'Fitch_Ratings_Issue_Short_Rating_effective_date',
 'Moodys_Credit_Watch_Long_Rating_effective_date',
 'Moodys_Enhanced_Short_Rating',
 'Fitch_Ratings_Short_Rating',
 'Standard__Poors_Credit_Watch_Long_Outlook_Rating_effective_date',
 'Moodys_Credit_Watch_Short_Rating_effective_date',
 'Fitch_Ratings_Short_Rating_effective_date',
 'Fitch_Ratings_Long_Rating_effective_date',
 'Moodys_Short_Rating',
 'Standard__Poors_Long_Rating',
 'Fitch_Ratings_Long_Rating',
 'Fitch_Ratings_Issue_Short_Rating',
 'Moodys_Credit_Watch_Long_Rating',
 'Standard__Poors_Credit_Watch_Long_Rating',
 'Fitch_Ratings_Issue_Long_Rating',
 'Moodys_Long_Rating_effective_date',
 'Standard__Poors_School_ICR_Long_Rating_effective_date',
 'Moodys_Short_Rating_effective_date',
 'Moodys_Issue_Long_Rating',
 'Moodys_Insured_Long_Rating_effective_date',
 'Standard__Poors_Long_Rating_effective_date',
 'Fitch_Ratings_Credit_Watch_Long_Rating_effective_date',
 'Standard__Poors_Issue_Long_Rating_effective_date',
 'Standard__Poors_Credit_Watch_Long_Outlook_Rating',
 'Moodys_Issue_Short_Rating_effective_date',]
str_float_columns = question_columns = ["SEDOL","linked_markets","clearing_agents","sector_class","category",
    "max_notification_days","NAICS_primary","cusip_change","SIC_primary","issuerissuer","SEDOL_status",
    "sub_category","hybrid_type", "IDII_status","sector",]

float_columns = ["maturity_amount","denom_increment_amount","min_denom_amount","issue_price","orig_principal_amount",
                "amount_outstanding", "amount_outstanding_decimal",]
date_columns = ["next_coupon_payment_date","delivery_date","settlement_date","issue_date","accrual_date","first_coupon_date", "previous_coupon_payment_date", "first_execution_date", 
                "last_period_accrues_from_date","maturity_date","formal_award_date", "sale_date", "next_call_date", "par_call_date", "next_sink_date",
                "amount_outstanding_change_date", "other_accrual_date", "refund_date","refunding_dated_date",
                "orig_avg_life_date","current_coupon_reset_date","previous_coupon_effective_date","next_put_date",
                "put_end_date","put_start_date","first_variable_reset_date","make_whole_call_end_date", 
                "next_reset_date","loc_expiration_date","odd_first_coupon_date","next_tender_date", 
                "tender_start_date","designated_termination_date","conditional_call_date","next_auction_reset_date",]


""" Removing elements in the del_column list from final_columns, these are primarily columns realated to credit ratings. """
final_columns = list(set(final_columns) - set(del_column))

bqclient = bigquery.Client(project=PROJECT,)


#helper function to load the final dataframe to bigquery
def load_data(bq,data,project,dataset,table):
    bq = bq
    table_id = '{}.{}.{}'.format(project,dataset,table)
    job_config = bigquery.LoadJobConfig(schema =[])
    job = bq.load_table_from_dataframe(data, table_id,job_config=job_config)

    try:
        job.result() 
        return 'success'  
    except BadRequest as ex:
        print(ex)

""" Helper function to read the xml elements for the main resference table."""
def classification(elem,new_dict): 
    """Function to parse details regarding the classification tag"""
    list_of_elem = elem.xpath("//instrument/global_information/organization_information/classifications")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            new_dict.update({element.attrib["classification_ranking"] + "_" + element.tag:element.text})


def ratings(elem,new_dict):
    """Function to parse details regarding the retaing of the issuer and the security"""
    list_of_elem = elem.xpath("//instrument/global_information/ratings")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
                tag = "_".join([element.attrib["agency"].replace("'",""),element.attrib["type"]])
                new_dict.update({tag.replace(" ","_"):element.text})
                new_dict.update({tag+"_"+"effective_date":element.attrib["effective_date"]})
    
def instrument_xref(elem,new_dict):
    """Function to parse details regarding various id like bloomberg_id, cusip and other identifiers for a cusip."""
    for element in elem.xpath("//instrument/master_information/instrument_xref/xref"):

        if element.attrib["type"] == "CUSIP":
            new_dict.update({element.attrib["type"]:element.text})
        elif element.attrib["type"] == "ISIN":
            new_dict.update({element.attrib["type"]:element.text})
        elif element.attrib["type"] == "Bloomberg Global Id":
            new_dict.update({element.attrib["type"]:element.text})
            if "id_bb_sec_num" in element.attrib:
                new_dict.update({"id_bb_sec_num":element.attrib["id_bb_sec_num"]})
            new_dict.update({"market_sector":element.attrib["market_sector"]})
            if "security_typ" in element.attrib:
                new_dict.update({"security_typ":element.attrib["security_typ"]})
            if "security_typ2" in element.attrib:
                new_dict.update({"security_typ2":element.attrib["security_typ2"]})
            if "ticker" in element.attrib:
                new_dict.update({"ticker":element.attrib["ticker"]})
            if "unique_id" in element.attrib:
                new_dict.update({"bloomberg_unique_id":element.attrib["unique_id"]})
            
def instrument_master(elem,new_dict):
    """Function to parse details from the instrument_master tag. Example - primary name, sector, federal tax status etc."""
    list_of_elem = elem.xpath("//instrument/master_information/instrument_master")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            new_dict.update({element.tag:element.text})
        
def country_information(elem,new_dict):
    """Function to parse details from the country_information tag. Example - country_code"""
    list_of_elem = elem.xpath("//instrument/global_information/country_information/instrument_country_information")
    if len(list_of_elem)>0:
         for element in list_of_elem[0].getchildren():
                new_dict.update({element.tag:element.text})

def instrument_details(elem,new_dict):
    """Function to parse details from the instrument_detail tag. Example - maturity_date,maturity_amount"""
    list_of_elem = elem.xpath("//instrument/global_information/instrument_details/maturity_details")
    if len(list_of_elem)>0:
         for element in list_of_elem[0].getchildren():
                new_dict.update({element.tag:element.text})
                
def denomination_amount(elem,new_dict):
    """Function to parse details from the instrument_details tag. Example - min_increment_amt,min_denomication_amount"""
    list_of_elem = elem.xpath("//instrument/global_information/instrument_details/denomination_amounts")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            new_dict.update({element.tag:element.text})
            
def debts_put_details(elem,new_dict):
    """Function to parse details from the instrument_details tag. Example - next_put_date,put_price etc."""
    
    list_of_elem = elem.xpath("//instrument/debt/put_details")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            new_dict.update({element.tag:element.text})
            
def escrow_details(elem,new_dict):
    """Function to parse details from the muni_escrow tag. Example - escrow_obligation agent etc."""
    list_of_elem = elem.xpath("//instrument/debt/muni_escrow")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            new_dict.update({element.tag:element.text})


def organization_master(elem,new_dict):
    list_of_elem = elem.xpath("//instrument/master_information/organization_master")
    if len(list_of_elem)>0:
        new_dict.update({"organization_master_id":list_of_elem[0].attrib["id"]})
        for element in list_of_elem[0].getchildren():
            if element.tag != "organization_xref":
                new_dict.update({"issuer"+"_"+element.tag:element.text})
            else:
                for childs in element.getchildren():
                    new_dict.update({childs.attrib["entity_level"] +"_"+ childs.attrib["type"]:childs.text})
                    
def debts_fixed_income(elem,new_dict):
    list_of_elem = elem.xpath("//instrument/debt/fixed_income")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            new_dict.update({element.tag:element.text})

        
def debts_muni_details(elem,new_dict):
    list_of_elem = elem.xpath("//instrument/debt/muni_details")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            if element.tag != "linkage_child":
                new_dict.update({element.tag:element.text})
                
def debts_call_details(elem,new_dict):
    list_of_elem = elem.xpath("//instrument/debt/call_details")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            if element.tag != "call_schedule":
                new_dict.update({element.tag:element.text})
                
def debts_floating_rate_instruments(elem,new_dict):
    list_of_elem = elem.xpath("//instrument/debt/floating_rate_instruments")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            new_dict.update({element.tag:element.text})
            
def debts_tender_details(elem,new_dict):
    list_of_elem = elem.xpath("//instrument/debt/tender_details")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            new_dict.update({element.tag:element.text})
            
def debts_loc_details(elem,new_dict):
    list_of_elem = elem.xpath("//instrument/debt/loc_details")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            new_dict.update({element.tag:element.text})

        
def debts_sink_details(elem,new_dict):
    list_of_elem = elem.xpath("//instrument/debt/sink_details")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():
            new_dict.update({element.tag:element.text})
    
def market_master(elem,new_dict):
    list_of_elem = elem.xpath("//instrument/master_information/market_master/market")
    if len(list_of_elem)>0:
        if "primary" in list_of_elem[0].attrib:
            new_dict.update({"primary_market":"true"})
        else:
            new_dict.update({"primary_market":"false"})
            
        new_dict.update({"market_id":list_of_elem[0].attrib["id"]})
        for element in list_of_elem[0].getchildren():
            if element.tag != "xref":
                new_dict.update({element.tag:element.text})
            else:
                new_dict.update({element.attrib["type"]:element.text})
                
def debts_amount_outstanding_history(elem,new_dict):
    list_of_elem = elem.xpath("//instrument/debt/amount_outstanding_history/change")
    if len(list_of_elem)>0:
        new_dict.update({"amount_outstanding_change_date":list_of_elem[0].attrib["date"]})
        for elements in list_of_elem[0].getchildren():
            new_dict.update({elements.tag:elements.text})




def create_df(new_elem,final_list):
    """ This functions creates a empty dictionary and calls the above helper function to parse the xml and the tag-value
    as a key-element pair for each instrument. Once done it appends this dictionary to the final_list."""
    new_dict = {}
    
    instrument_xref(new_elem,new_dict)
    instrument_master(new_elem,new_dict)
    classification(new_elem,new_dict)
    ratings(new_elem,new_dict)
    country_information(new_elem,new_dict)
    instrument_details(new_elem,new_dict)
    denomination_amount(new_elem,new_dict)
    debts_put_details(new_elem,new_dict)
    escrow_details(new_elem,new_dict)
    organization_master(new_elem,new_dict)
    debts_fixed_income(new_elem,new_dict)
    debts_muni_details(new_elem,new_dict)
    debts_call_details(new_elem,new_dict)
    debts_floating_rate_instruments(new_elem,new_dict)
    debts_tender_details(new_elem,new_dict)
    debts_loc_details(new_elem,new_dict)
    debts_sink_details(new_elem,new_dict)
    market_master(new_elem,new_dict)
    debts_amount_outstanding_history(new_elem,new_dict)
    
    final_list.append(new_dict)
    

            
def load_dataframe(bqclient,doc_df,PROJECT,dataset,table,count):
    """Function that Calls the load_data function to upload the dataframe to bigquery"""
    try:
        test = load_data(bqclient,doc_df,PROJECT,dataset,table)
        print(test)
    except Exception as e:
        print(e)


def upload_dataframe(final_list,count,data_timestamp):
    """Function to prepare the datafraome for ingestion into bigquery table"""
    doc_df = pd.DataFrame(final_list)
    new_columns = []
    #Modify the column names to remove special characters, "-" and spaces.
    for columns in doc_df.columns:
        columns = columns.replace("&","")
        columns = columns.replace(" ","_")
        columns = columns.replace("-","_")
        new_columns.append(columns)
    doc_df.columns = new_columns
    # Appends the left over columns i.e columns which are not present in the current batch but are part of the schema.
    for columns in list(set(final_columns) - set(new_columns)):

        doc_df[columns] = None
    # Create a new column to make it point in time, this shows when the data was acutally loaded into bigquery
    doc_df["upload_date"] = my_timezone.localize(datetime.now()).date() 
    
    doc_df.replace(np.nan,None,inplace = True)
    # Type coversion of different columns based on the lists of columns defined on the top.
    for columns in float_columns:
        doc_df[columns] = doc_df[float_columns].astype("float")
        doc_df[columns] = pd.Series(doc_df[columns]).round(4)
    for columns in str_float_columns:
        doc_df[columns] = doc_df[columns].astype('float')
    for columns in str_column:
        doc_df[columns] = doc_df[columns].astype('str')
        

    # Type casting date columns to DATE sql datatype
    for columns in doc_df.columns:
        if columns in date_columns:
            doc_df[columns] = pd.to_datetime(doc_df[columns],format = "%Y-%m-%d",exact = True,errors = 'coerce').dt.date
    # Adding the data_timestamp, this the time when the data was generated by ICE.
    doc_df["data_timestamp"] = pd.to_datetime(data_timestamp[0])
    doc_df["data_timestamp"] = doc_df["data_timestamp"].dt.date
    if "Ticker" in doc_df.columns:
        del doc_df["Ticker"]
    
    
    load_dataframe(bqclient,doc_df,PROJECT,dataset,table,count)


count = 0
total = 0
"""Loops over the context to find the exact number of instruments in the xml file. This is used when creating batch size
    and uploading the final residual amount at the end"""
context = etree.iterparse(path,events=("start",),tag = "instrument")

for j,k in context:
    count += 1
    while k.getprevious() is not None:    
        del k.getparent()[0]
else:
    total = count

data_timestamp = []
"""Creating an iterator to get the timestamp of the data."""
head_context = etree.iterparse(path,events=("start",),tag = "header")
for event,elem in head_context:
    new_elem = etree.parse(BytesIO(etree.tostring(elem)))
    list_of_elem = elem.xpath("//header")
    if len(list_of_elem)>0:
        for element in list_of_elem[0].getchildren():

            if element.tag == "timestamp":
                data_timestamp.append(element.text)
    break

def fast_iter(context,data_timestamp, *args, **kwargs):
    """Main iterator function to loop over the elements """
  
    final_list = []
    count = 0
    for _,elem in context:
        process_element(elem,final_list)
        count += 1
        # This makes sure that the context tree is not expanding, We delete the non-referred nodes in the xml tree to save space.
        while elem.getprevious() is not None:
            
            del elem.getparent()[0]
        # Create a batch size of 10,000 and then uploads this data to bigquery
        if count%10000 == 0:
            upload_dataframe(final_list,count,data_timestamp)
            final_list = []
        # Upload the left over instruments once the batch iteration is over, executed at the end of this loop.
        if count == total:
            upload_dataframe(final_list,count,data_timestamp)
            final_list = []
            
        


def process_element(elem,final_list):    
    new_elem = etree.parse(BytesIO(etree.tostring(elem)))

    create_df(new_elem,final_list)


# Creates as iterator to loop over
context = etree.iterparse(path,events=("start",),tag = "instrument")
fast_iter(context,data_timestamp)



        






        


    
    
    

    
    
    
           
    
    
    
                
        
    
    