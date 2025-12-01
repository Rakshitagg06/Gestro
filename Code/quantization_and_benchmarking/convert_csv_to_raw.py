import pandas as pd
import numpy as np
import os

def convert_csv_to_raw_with_relative_paths(csv_path, output_dir):
    """
    Reads a CSV file containing gesture data, converts each row into a
    separate binary .raw file, and creates an input_list.txt with
    correct relative paths for the SNPE tools.

    Args:
        csv_path (str): The path to the input CSV file.
        output_dir (str): The directory to save the .raw files and input_list.txt.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at '{csv_path}'")
        return

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, header=None)

    # The first column is the label, the rest are features
    features = df.iloc[:, 1:].values.astype(np.float32)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving .raw files to: {output_dir}/")

    # The path for the list file
    list_file_path = os.path.join(output_dir, 'input_list.txt')

    with open(list_file_path, 'w') as outfile:
        for i, row in enumerate(features):
            raw_filename = f"sample_{i}.raw"
            raw_filepath_for_saving = os.path.join(output_dir, raw_filename)
            
            # Save the landmark data as a raw binary file
            row.tofile(raw_filepath_for_saving)
            
            # This is the crucial part: create the correct RELATIVE path for the list file
            # e.g., "quantization_data/sample_0.raw"
            path_for_list_file = os.path.join(output_dir, raw_filename)
            
            # Use forward slashes for cross-platform compatibility with the tools
            correct_relative_path = path_for_list_file.replace('\\', '/')
            outfile.write(correct_relative_path + '\n')

    print(f"✅ Successfully created .raw files.")
    print(f"✅ input_list.txt with correct relative paths saved to: {list_file_path}")


if __name__ == "__main__":
    csv_file = input("Enter the path to your .csv data file: ").strip()
    output_directory = input("Enter the name for the output directory: ").strip()
    
    convert_csv_to_raw_with_relative_paths(csv_file, output_directory)

