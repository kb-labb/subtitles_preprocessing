import pandas as pd
import argparse


def read_parquet_files(txt_file):
    with open(txt_file, "r") as f:
        parquet_files = [line.strip() for line in f.readlines() if line.strip()]
    return parquet_files


def combine_parquet_files_from_txt(txt_file, output_file_prefix):
    parquet_files = read_parquet_files(txt_file)

    total_duration = 0
    dataframes = []
    n = 0

    for file in parquet_files:
        df = pd.read_parquet(file)
        
        for index, row in df.iterrows():
            dataframes.append(row.to_frame().T)
            row["duration"] = row["end_time"] - row["start_time"] # skip if duration is already in the file
            total_duration += row["duration"]

            # Check if the total duration exceeds 25 hours (25 hours = 90000000 ms), save if yes
            if total_duration > 90000000:
                n += 1
                combined_df = pd.concat(dataframes, ignore_index=True)
                output_file = f"{output_file_prefix}_{n}.parquet"
                combined_df.to_parquet(output_file, index=False)
                print(f"Combined parquet file saved to: {output_file}")

                total_duration = 0
                dataframes = []

    # Save any remaining rows that haven't been saved yet
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        output_file = f"{output_file_prefix}_{n + 1}.parquet"
        combined_df.to_parquet(output_file, index=False)
        print(f"Combined parquet file saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple parquet files from a list in a text file."
    )

    parser.add_argument(
        "--input_txt", type=str, help="Path to the text file containing the list of parquet files."
    )
    parser.add_argument(
        "--output_file_prefix", type=str, help="Prefix for the combined output parquet file, for example '/home/jussik/svt' will create files like '/home/jussik/svt_1.parquet', '/home/jussik/svt_2.parquet', etc."
    )

    args = parser.parse_args()
    combine_parquet_files_from_txt(args.input_txt, args.output_file_prefix)


if __name__ == "__main__":
    main()
