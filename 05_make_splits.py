import pandas as pd
import numpy as np
import os
from os import makedirs
from tqdm import tqdm
import argparse
import time
import re
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)


parser.add_argument(
    "--metadata",
    type=str,
    default='metadata',
    help="Path to metadafile." 
)
parser.add_argument(
    "--output",
    type=str,
    default='split',
    help="Directory where .csv train, test, validation files are saved.")

parser.add_argument('--ratios', nargs='+', type=float, default=[0.8, 0.1, 0.1],
                    help='Ratios for train, test, and validation sets.')

args = vars(parser.parse_args())
    

def make_splits(metadata_dir, output_dir, ratios):
    """ Create and save .csv files with train/dev/test splits. """

    metadata = pd.read_csv(os.path.join(metadata_dir, 'metadata_full.csv')) #file with chunks' filenames and transcriptions 
    train_size, test_size, val_size = ratios[0], ratios[1], ratios[2]
    
    metadata['programid'] = metadata['file_name'].apply(lambda x: ("_").join(re.findall(r"[\w']+", x)[:-2]))

    metadata = metadata.sample(frac=1, random_state=42) 

    total = len(metadata)
    train = round(train_size * total)
    test = round(test_size * total)
    val = round(val_size * total)

    train_df = metadata[:train]
    test_df = metadata[train:train + test]
    validation_df = metadata[train + test:]

    train_df.to_csv(f'{output_dir}/train.csv', index = False)
    test_df.to_csv(f'{output_dir}/test.csv', index = False)
    validation_df.to_csv(f'{output_dir}/validation.csv', index = False)

metadata_dir = args["metadata"]
output_dir = args["output"]
ratios = args["ratios"] 
if not os.path.exists(output_dir):
    makedirs(output_dir)
make_splits(metadata_dir, output_dir, ratios)
    