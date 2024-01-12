import os
import pandas as pd
import argparse
from datasets import load_dataset, Audio, Dataset, DatasetDict

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--input",
    type=str,
    default='split',
    help="Path to train, test and validation splits." 
)
parser.add_argument(
    "--chunks",
    type=str,
    default='chunks',
    help="Directory where the chunks are stored")

parser.add_argument(
    "--output",
    type=str,
    default='HF',
    help="Directory where HF dataset is saved to")

args = vars(parser.parse_args())    
# Load CSVs
split_dir = args["input"]
df_train = pd.read_csv(os.path.join(split_dir,"train.csv"))
df_validation = pd.read_csv(os.path.join(split_dir,"validation.csv"))
df_test = pd.read_csv(os.path.join(split_dir,"test.csv"))

# Filter out small files
def filter_files_by_size(df):
    base_path = "/home/leoves/projects/nov08/subtitles_preprocessing/chunks/"
    valid_files = []
    valid_transcriptions = []
    for idx, row in df.iterrows():
        file_path = os.path.join(base_path, row["file_name"])
        if os.path.getsize(file_path) > 225:
            valid_files.append(file_path)
            valid_transcriptions.append(row["transcription"])
    return valid_files, valid_transcriptions

train_files, train_transcriptions = filter_files_by_size(df_train)
validation_files, validation_transcriptions = filter_files_by_size(df_validation)
test_files, test_transcriptions = filter_files_by_size(df_test)

# Create datasets
train_audio_dataset = Dataset.from_dict({"audio": train_files, "transcription": train_transcriptions}).cast_column("audio", Audio())
test_audio_dataset = Dataset.from_dict({"audio": test_files, "transcription": test_transcriptions}).cast_column("audio", Audio())
validation_audio_dataset = Dataset.from_dict({"audio": validation_files, "transcription": validation_transcriptions}).cast_column("audio", Audio())

dataset = DatasetDict({
    "train": train_audio_dataset,
    "test": test_audio_dataset,
    "validation": validation_audio_dataset
})
print(dataset)
output_dir = args["output"]
dataset.save_to_disk(os.path.join(output_dir,"test.hf"))
