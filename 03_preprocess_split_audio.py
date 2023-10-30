import argparse
import os
import sys
import pandas as pd
import pdb
from multiprocessing import Pool
from tqdm import tqdm
from pydub import AudioSegment
import concurrent.futures
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import whisper


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

logging.info("Starting")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    type=str,
    default="results",
    help="Directory containing subdirectories with audio and srt files.",
)

parser.add_argument(
    "--output",
    type=str,
    default="chunks",
    help="Directory where the resulting wav file chunks should be saved.",
)

parser.add_argument(
    "--processes",
    type=int,
    default=2,
    help="Number of processes to use for multiprocessing.",
)

parser.add_argument(
    "--parquet_dir",
    type=str,
    default="subs_preprocessed",
    help="Name of directory where parquet file(s) are stored.",
)
args = parser.parse_args()

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

output_dir = args.output

parquet_files = [d for d in os.listdir(args.parquet_dir)]    

def split_audio(audio_df : pd.DataFrame):                    
    f = audio_df.iloc[0, audio_df.columns.get_loc("program")]
    audio_file_name = audio_df.iloc[0, audio_df.columns.get_loc("audio")]
    audio_path = f"{args.data}/{f}/{audio_file_name}"

    audio = AudioSegment.from_wav(audio_path)
    os.makedirs(output_dir, exist_ok=True)
    filenames = []
    for i, row in audio_df.iterrows():
        audio_chunk = audio[row["start_bucket"] : row["end_bucket"]]
        os.makedirs(os.path.join(output_dir, row['program']), exist_ok=True)
        filename = f"{output_dir}/{row['program']}/{row['observation_nr']}.wav"
        #detect_language(filename) #detect language doesn't work, either whisper-tiny sucks or I'm not doing it correctly
        audio_chunk.export(filename, format="wav")
        filenames.append(filename)

    return filenames

def detect_language(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    pred_tokens = model.generate(input_features, max_new_tokens=448)
    pred_text = processor.batch_decode(pred_tokens, skip_special_tokens=True)
    pred_language = processor.batch_decode(pred_tokens[:, 1:2], skip_special_tokens=False)
    #
    print(pred_text)
    print(pred_language)

# Multiple parquet files do not work yet, we overwrite the same metadata.csv file, but it is easy to fix.
metadata_df = pd.DataFrame()
for parquet_file in tqdm(parquet_files):
    logging.info(f"Processing parquet file {parquet_file}")

    df_subs = pd.read_parquet(args.parquet_dir + "/" + parquet_file)
    df_subs["bucket_filename"] = (
            df_subs["observation_nr"].astype(str) + ".wav"
    )

    # Keep only first obs in each observation_nr group
    df_subs_unique = df_subs.drop_duplicates(["observation_nr", "audio"], keep="first")
    df_subs_unique = df_subs_unique.drop(df_subs_unique.tail(1).index) 
    
    audio_groups = df_subs_unique[["program", "audio", "start_bucket", "end_bucket", "bucket_filename",
                                   "observation_nr"]].groupby("audio")
    df_list = [audio_groups.get_group(df) for df in audio_groups.groups]
    
    logging.info("Splitting audio")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        out_filenames = list(tqdm(executor.map(split_audio, df_list), total=len(df_list)-1))

    df_subs_unique = df_subs_unique.copy()
    df_subs_unique["bucket_path"] = [bucket_path for program_paths_list in out_filenames for bucket_path in program_paths_list]

    result = df_subs_unique[['bucket_filename', 'text_timestamps_bucket']].rename(columns={'bucket_filename': 'file_name',
                                                                                           'text_timestamps_bucket':
                                                                                               'transcription'})
    result.to_csv(f"{output_dir}/{df_subs['program'][0]}/metadata.csv", index=False)

logging.info("Done")