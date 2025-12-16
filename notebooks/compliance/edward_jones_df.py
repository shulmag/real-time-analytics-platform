'''
Description: This script runs the compliance module, specifically for Edward Jones, and on hypothetical trades. This data is passed 
             in as a pickle file before running `price_cusips_list(...)` which actually calls the compliance module with this data.
             This script was used in the following task: https://ficcai.atlassian.net/browse/FA-2702.

             **NOTE**: For the task, using a 16 CPU high memory machine (n1-highmem-16: 16 vCPUs, 104 GB Memory), the entire process took one hour. 
             **NOTE**: Before running the script, create a virtual environment with `app_engine/demo/server/requirements.txt`.
             **NOTE**: This script needs to be run on a VM so that the yield curve redis can be accessed which is necessary for `process_data(...)`. The error that will be raised otherwise is a `TimeoutError`.
             **NOTE**: To not use the yield spread with similar trades model for point-in-time pricing, the following variable must be changed: `app_engine/demo/server/modules/auxiliary_variables.py::USE_SIMILAR_TRADES_MODEL_FOR_YIELD_SPREAD_PREDICTIONS_FOR_POINT_IN_TIME_PRICING`.
             **NOTE**: To see the output of this script in an `output.txt` file use the command: $ python -u edward_jones_df.py >> output.txt. 
             **NOTE**: To run the procedure in the background, use the command: $ nohup python -u edward_jones_df.py >> output.txt 2>&1 &. This will return a process number such as [1] 66581, which can be used to kill the process.
             Breakdown:
             1. `nohup`: This allows the script to continue running even after you log out or close the terminal.
             2. python -u <file_name>.py: This part is executing your Python script in unbuffered mode (meaning that output is written immediately). If you are using Python 3, you might want to specify python3 instead of just python, depending on your environment.
             3. >> output.txt 2>&1:
                 * >> output.txt appends the standard output (stdout) of the script to output.txt instead of overwriting it.
                 * 2>&1 redirects standard error (stderr) to the same file as standard output, so both stdout and stderr go into output.txt.
             4. &: This runs the command in the background.
'''
import os
import sys

import pandas as pd

from price_compliance_df import price_compliance_df
from edward_jones_actual_trades_query import OUTPUT_FILE_NAME, prepare_priced_df_for_edward_jones


server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'app_engine', 'demo', 'server'))    # get the directory containing the 'app_engine/demo/server' package
sys.path.append(server_dir)    # add the directory to sys.path
print('NOTE: this file must be run from the `notebooks/compliance/` directory')


if __name__ == '__main__':
    df = pd.read_pickle('ej_2025-02-10.pkl')
    print(f'Number of items in `df`: {len(df)}')
    print(f'{len(df.columns)} columns: {sorted(df.columns)}')
    priced_df = price_compliance_df(df, 
                                    df_contains_trade_history_and_reference_data=False, 
                                    keep_additional_columns_from_original_df=True)
    priced_df = prepare_priced_df_for_edward_jones(priced_df)

    print(priced_df.to_markdown())
    priced_df.to_csv(OUTPUT_FILE_NAME, index=False)
