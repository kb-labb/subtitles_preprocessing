import pandas as pd
import os
import argparse
import time

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)

parser.add_argument(
    "--data",
    type=str,
    help="Directory with chunks.",
)
parser.add_argument(
    "--output",
    type=str,
    help="Directory where the concatenated metadata file should be saved.")

args = vars(parser.parse_args())    
 
chunks_dir = args["data"]
output_dir = args["output"]
concatenated_metadata = pd.DataFrame()

for root, dirs, files in os.walk(chunks_dir):
    for file in files:
        if file == 'metadata.csv':
            metadata_file = os.path.join(root, file)
            metadata = pd.read_csv(metadata_file)
            concatenated_metadata = pd.concat([concatenated_metadata, metadata])
            
concatenated_metadata.to_csv(f'{output_dir}/metadata_full.csv', index=False)
  
    
