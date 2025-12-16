import csv
import random


def select_random_lines_from_csv(input_file, output_file, has_header: bool = False, num_lines: int = 100):
    with open(input_file, 'r') as csvfile:
        reader = list(csv.reader(csvfile))
        if has_header:
            header = reader[0]  # Assuming the first row is a header
            data = reader[1:]  # Exclude header from data
        else:
            data = reader

        # Check if the number of lines requested is greater than available lines
        if num_lines > len(data):
            print(f'Requested {num_lines} lines, but the file only contains {len(data)} lines.')
            num_lines = len(data)

        # Randomly select the lines
        selected_lines = random.sample(data, num_lines)    # `random.sample(...)` does not use replacement

    # Write the selected lines to a new CSV
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        if has_header: writer.writerow(header)  # Write header to the output file
        writer.writerows(selected_lines)

    print(f'Selected {num_lines} random lines from {input_file} and saved to {output_file}')


# Example usage
input_csv = '/Users/user/Downloads/ficc-week-4.csv'
output_csv = 'output.csv'
select_random_lines_from_csv(input_csv, output_csv)
