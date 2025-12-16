'''
Create date: 2024-06-25
Modified date: 2024-06-25
Description: Combine two CSVs where the corrections from the second CSV are reflected in the first one. More details in: https://ficcai.atlassian.net/browse/FA-2098.
'''
import csv


DATETIME = '2024_06_13_09_00_00'
ORIGINALLY_PRICED_CUSIPS_FILE_PATH = f'priced_{DATETIME}.csv'
REPRICED_CUSIPS_FILE_PATH = f'priced_{DATETIME}_rerun_failed_cusips.csv'
OUTPUT_FILE_PATH = f'priced_{DATETIME}_corrected.csv'


repriced_cusips = set()
cusip_quantity_tradetype_to_repriced_yield_and_price = dict()
with open(REPRICED_CUSIPS_FILE_PATH, 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for row in csvreader:
        cusip, quantity, trade_type = row[0], row[1], row[2]
        repriced_cusips.add(cusip)
        cusip_quantity_tradetype_to_repriced_yield_and_price[(cusip, quantity, trade_type)] = row


combined_csv_lines = []
with open(ORIGINALLY_PRICED_CUSIPS_FILE_PATH, 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        cusip = row[0]
        if cusip in repriced_cusips:
            quantity, trade_type = row[1], row[2]
            combined_csv_lines.append(cusip_quantity_tradetype_to_repriced_yield_and_price[(cusip, quantity, trade_type)])
        else:
            combined_csv_lines.append(row)


with open(OUTPUT_FILE_PATH, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(combined_csv_lines)
