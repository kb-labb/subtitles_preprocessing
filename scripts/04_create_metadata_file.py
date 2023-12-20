import pandas as pd
import os
from os import makedirs
import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--data",
        type=str,
        default="chunks",
        help="Directory with chunks.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metadata",
        help="Directory where the concatenated metadata file should be saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()
    chunks_dir = args.data
    output_dir = args.output
    if not os.path.exists(output_dir):
        makedirs(output_dir)
    concatenated_metadata = pd.DataFrame()

    for root, _, files in os.walk(chunks_dir):
        for file in files:
            if file == "metadata.csv":
                metadata_file = os.path.join(root, file)
                metadata = pd.read_csv(metadata_file)
                concatenated_metadata = pd.concat([concatenated_metadata, metadata])

    concatenated_metadata.to_csv(f"{output_dir}/metadata_full.csv", index=False)


if __name__ == "__main__":
    main()
