# Scripts for Cloud Functions

This directory contains scripts to update, deploy, or rerun cloud functions. 

## `rerun_yield_curve_functions.py`

Description: This script can be run to re-populate tables related to the yield curve if a failure occurs. This occurs if the cloud function `update_all_sp_indices_and_maturities` fails. If this occurs, first rerun that cloud function and ensure the data has been updated for the date in question. Then proceed to the instructions below:

1. Start a VM, as the VPC connector for Redis cannot be accessed from a local machine. 
2. Ensure that the path to your credentials file is in `rerun_yield_curve_functions.py`.
3. Run the script with the `TESTING` flag set to `True` to see that the run will be successful, filling in the `target_date` to the desired date as follows ```python rerun_yield_curve_functions.py --testing True --target_date 2025-02-14``` where `target_date` is set to `2025-02-14` as an example. NOTE: the desired behavior in testing mode is that the first two cloud functions (`train_daily_etf_model` and `train_daily_yield_curve`) run successfully and the last, `compute_shape_parameter` fails, as this cloud function is dependent on the first two uploading data to their corresponding tables. If the first two do not run, this indicates that `update_all_sp_indices_and_maturities` did not run successfully or that the `target_date` entered is incorrect. 
4. Once you have confirmed that the testing run was successful, rerun the command with the `TESTING` flag set to `False`: ```python rerun_yield_curve_functions.py --testing False --target_date 2025-02-14```. If all three cloud functions run successfully, the yield curve tables and the yield curve redis instance should be updated.

See [this notion page](https://www.notion.so/Yield-Curve-0e9d3fb1a49a4789826083361257a962?pvs=4) for more details on how the yield curve functions work together.
