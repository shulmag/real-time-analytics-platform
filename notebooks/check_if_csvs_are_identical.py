import pandas as pd


def are_csvs_identical(file1, file2):
    try:
        # Read the two CSV files into DataFrames
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        identical_columns = ['cusip', 'quantity', 'trade_type']

        # Check if the shape of both DataFrames is the same
        if df1.shape != df2.shape:
            print(f'Shapes are not identical. df1.shape: {df1.shape}, df2.shape: {df2.shape}')
            return False

        # Find differences
        differences = df1.drop(columns=identical_columns).ne(df2.drop(columns=identical_columns)) & ~(df1.drop(columns=identical_columns).isna() & df2.drop(columns=identical_columns).isna())
        differing_rows = differences.any(axis=1)  # Identify rows with any differences

        # If there are differences, export them to a CSV
        if differing_rows.any():
            print("The following rows differ between the two CSV files:")
            
            # Extract the differing rows, along with the identical columns
            diff_df1 = df1[differing_rows]
            diff_df2 = df2[differing_rows]

            # Highlight the differences
            diff_highlight = diff_df1.drop(columns=identical_columns).copy()
            for col in diff_highlight.columns:
                diff_highlight[col] = diff_df1[col].astype(str) + " | " + diff_df2[col].astype(str)
                diff_highlight[col] = diff_highlight[col].where(differences[col], '')

            # Add the identical columns back to the final DataFrame
            for col in identical_columns:
                diff_highlight[col] = diff_df1[col]

            # Reorder columns to place the identical columns at their original positions
            final_columns_order = identical_columns + [col for col in diff_highlight.columns if col not in identical_columns]
            diff_df = diff_highlight[final_columns_order]
            print(diff_df)
        else:
            print("The CSV files are identical.")

        # Export the differences to a CSV file
        diff_df.to_csv(output_file, index=True)
        print(f"Differences have been exported to {output_file}.")

    except Exception as e:
        print(f"An error occurred: {e}")
        return False


file1 = '/Users/user/Downloads/priced_2024_07_05_16_00_00_100_redis.csv'
file2 = '/Users/user/Downloads/priced_2024_07_05_16_00_00_100_bigquery.csv'
output_file = '/Users/user/Downloads/differences.csv'

if are_csvs_identical(file1, file2):
    print("The CSV files are identical.")
else:
    print("The CSV files are not identical.")
