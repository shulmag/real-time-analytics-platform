# Instructions
1. In auxiliary_variables.py, change `USERNAME` and `PASSWORD` to match your credentials.
2. In auxiliary_variables.py, change `CSV_FILEPATH` to be the file path of the CSV. The CSV must have the following format: (1) every row is a unique CUSIP, or (2) every row has a CUSIP, quantity, and trade type (comma separated).
3. In auxiliary_variables.py, change `UNIQUE_QUANTITIES` and `UNIQUE_TRADE_TYPES` to match the quantities and trade types, respectively, for each CUSIP to be priced. These variables are unused if the CSV in `CSV_FILEPATH` has a quantity and trade type associated with each CUSIP, i.e., in format option (2).
4. Create an environment using the requirements.txt file.
5. Run api_call.py in your terminal using
```>>> python api_call.py```
If writing the output to an external file, such as `output.txt`, use the following command to write the output immediately:
```>>>python -u api_call.py >> output.txt```

# Notes
- The `try...except` mechanism with the exponential backoff retry is implemented in asynchronous_api_calls.py.
- The output CSV will be stored in a file named `priced_<datetime>.csv`. This is defined in `api_call.py`.
- The code has been tested with Python 3.12.3, but it should work for any version of Python >= 3.10.
- To ensure the script uses data available at the time it starts running (and not data received after it begins), pass `current_time_string` as the final argument to the `call_price_batches(...)` function in `api_call.py::main(...)`. `current_time_string` is defined to be the time the script began running.
