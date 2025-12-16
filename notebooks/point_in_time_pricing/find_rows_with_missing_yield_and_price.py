'''
Create date: 2024-06-25
Modified date: 2024-06-25
Description: Find CUSIPs that are missing a yield and price in a priced CSV. More details in: https://ficcai.atlassian.net/browse/FA-2098.
'''
import csv


FILE_PATH = 'priced_2024_06_14_15_00_00.csv'    # final CSV output from this code is the same whether we use this priced_2024_06_14_15_00_00.csv or priced_2024_06_13_09_00_00.csv since the input file when originally priced was the same
OUTPUT_FILE_PATH = 'failed_cusips.csv'


cusips = []
with open(FILE_PATH, 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for row in csvreader:
        if row[3] == '' and row[4] == '':    # yield and price are missing
            cusips.append(row[0])    # first item in `row` corresponds to the CUSIP


cusips_already_seen = set()
unique_cusips = []
for cusip in cusips:
    if cusip not in cusips_already_seen:
        unique_cusips.append([cusip])    # need to put the cusip in a list so that it writes properly to the CSV
        cusips_already_seen.add(cusip)


with open(OUTPUT_FILE_PATH, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(unique_cusips)
