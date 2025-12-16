'''
Description: This script reruns the yield curve cloud functions `train_daily_etf_model`, `train_daily_yield_curve`, and `compute_shape_parameter` locally. This is to be used when a failure has occurred. 
'''
import sys
import os
import argparse


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/user/ficc/creds.json'

cloud_function_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if cloud_function_directory not in sys.path: sys.path.append(cloud_function_directory)


from train_daily_etf_model import main as train_daily_etf_model_main
from train_daily_yield_curve import main as train_daily_yield_curve_main
from compute_shape_parameter import main as compute_shape_parameter_main


def rerun_yield_curve_functions(*, testing=True, target_date=None) -> None:
    cloud_functions = [train_daily_etf_model_main, train_daily_yield_curve_main, compute_shape_parameter_main]

    for module in cloud_functions:
        setattr(module, "TESTING", testing)

    # Use the `set_target_date` function to update TARGET_DATE in each Cloud Function, which otherwise defaults to datetime.now()
    for module in cloud_functions:
        module.set_target_date(target_date)

    print("\nThe following cloud functions will be called with the following variables:\n")
    for module in cloud_functions:
        print(f"{module.__name__}")
        print(f"   - TESTING: {getattr(module, 'TESTING')}")
        print(f"   - TARGET_DATE: {getattr(module, 'TARGET_DATE')}")
        print("")  

    try:
        for module in cloud_functions:
            print(f"Calling {module.__name__}...")
            module.main(None)
            print(f"Finished {module.__name__}")
    
        print(f"All {len(cloud_functions)} functions executed successfully.")
    
    except Exception as e:
        print(f"{type(e)} occurred while running {module.__name__}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-run yield curve cloud functions after failure.")
    parser.add_argument("--testing", type=bool, default=True, help="Run in testing mode (default: True).")
    parser.add_argument("--target_date", type=str, required=True, help="Target date (YYYY-MM-DD).")

    args = parser.parse_args()
    rerun_yield_curve_functions(testing=args.testing, target_date=args.target_date)
