#!/usr/bin/env python
# coding: utf-8
# # Needed libraries
import sys
#!{sys.executable} -m pip install selenium
#!{sys.executable} -m pip install fake-useragent
# Import libraries

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
import re,time,os,codecs,csv
import tqdm
import math
start_time = time.time()
# Create Browser [TBD Headless]

#make browser
ua=UserAgent()
dcap = dict(DesiredCapabilities.CHROME)
print(dcap)
dcap["CHROME.page.settings.userAgent"] = (ua.random)
service_args=['--ssl-protocol=any','--ignore-ssl-errors=true']

driver = webdriver.Chrome(ChromeDriverManager().install())
#driver = webdriver.Chrome('/Users/Gil/Notebooks/chromedriver',desired_capabilities=dcap,service_args=service_args)

link= 'https://emma.msrb.org/Home/Index' #'https://emma.msrb.org/TradeData/Search'
timeout = 10
driver.implicitly_wait(2) # seconds
#visit the link
driver.get(link)
#WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.ID, 'acceptcookies')))
time.sleep(1)
cookie = driver.find_element_by_class_name('acceptcookies')
cookie.click()

#time.sleep(1)

#cusip_list =['263590EZ0','575824AE7']


import pandas as pd
df = pd.read_csv('/Users/Gil/git/ficc/emma/cusip_list.csv')
df = df[:32283]
cusip_list= df['cusip']

loaded = []
try:
    df_loaded = pd.read_csv('/Users/Gil/git/ficc/emma/called_cusip.csv')
    loaded = df_loaded['cusip']
    not_loaded = set(cusip_list) - set(loaded)
except Exception as e:
    not_loaded = cusip_list

#not_loaded = [c for c in cusip_list if c not in loaded]: 

#Processing for every Month
if len(loaded) == 0:
    with open('/Users/Gil/git/ficc/emma/called_cusip.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('cusip','bond_called_date'))

for cusip in tqdm.tqdm(not_loaded):
    if cusip != cusip:
        continue
    
    quickSearchText = driver.find_element_by_id('quickSearchText')
    quickSearchText.clear()
    quickSearchText.send_keys(cusip) 
    
    quickSearchButton = driver.find_element_by_id('quickSearchButton')
    quickSearchButton.click()
    #time.sleep(2)
    try:
        yesButton = driver.find_element_by_id('ctl00_mainContentArea_disclaimerContent_yesButton')
        yesButton.click()
        securityDetailsContent = driver.find_element_by_id('securityDetailsContent')
        securityDetailsContent.click()
    except NoSuchElementException:
        pass 

    try:
        driver.find_element_by_link_text('Disclosure Documents').click()
        view = driver.find_element_by_link_text('View').click()
    except NoSuchElementException:
        continue

    try:
        driver.find_element_by_link_text('x').click()
    except NoSuchElementException:
        pass 

    try:
        called = driver.find_element_by_id("ruleMandatedDiv").text
        bond_called_date = called #called.split('Bond Call as of ')[1]
        with open('/Users/Gil/git/ficc/emma/called_cusip.csv', mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow((cusip,bond_called_date))
    except NoSuchElementException:
        continue 

print('DONE!')