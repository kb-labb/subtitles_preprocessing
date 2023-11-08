import argparse
import torch
import os
import sys
import time
import pandas as pd
import numpy as np
import pdb
from multiprocessing import Pool
from tqdm import tqdm
from pydub import AudioSegment
import concurrent.futures
import logging
from faster_whisper import WhisperModel


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
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
    default=8,
    help="Number of processes to use for multiprocessing.",
)

parser.add_argument(
    "--parquet_dir",
    type=str,
    default="subs_preprocessed",
    help="Name of directory where parquet file(s) are stored.",
)
args = parser.parse_args()
output_dir = args.output

parquet_files = [d for d in os.listdir(args.parquet_dir)]    
sampling_rate = 16000   
model_size = "large-v2"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Save processed audio as desired format
def save_processed_audio(audio_segment, output_path):
    audio_segment.export(output_path, format='wav')  # Saving as WAV format

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
        output_path = os.path.join(audio_path, filename)
        save_processed_audio(audio_chunk, filename)
        # language detection
        segments, info = model.transcribe(filename, vad_filter=True, beam_size=5)
        if info.language == 'sv' and info.language_probability > 0.7:
            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            audio_chunk.export(filename, format="wav")
            filenames.append(filename)
        else:
            os.remove(filename)

    return filenames

def main():
    
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
            out_filenames = list(tqdm(executor.map(split_audio, df_list), total=len(df_list)))
        #drop filenames that didn't pass language detection check by a left join 
        out_filenames_new = [x.replace(f"{output_dir}/{df_subs['program'][0]}/", '') for x in out_filenames[0]]
        out_filenames_df = pd.DataFrame({'bucket_filename':out_filenames_new})
        df_subs_unique = df_subs_unique.merge(out_filenames_df.drop_duplicates(), on=['bucket_filename'], how='left', indicator=True)
        df_subs_unique = df_subs_unique[df_subs_unique._merge == 'both']
        if df_subs_unique.empty:
            os.rmdir(f"{output_dir}/{df_subs['program'][0]}")
        else: 
            df_subs_unique = df_subs_unique.copy()
            df_subs_unique["bucket_path"] = [bucket_path for program_paths_list in out_filenames for bucket_path in program_paths_list]
            df_subs_unique['bucket_filename'] = df_subs_unique['bucket_filename'].str.slice_replace(0, 0, (df_subs_unique['program'].iloc[0]+"/")) 

            result = df_subs_unique[['bucket_filename', 'text_timestamps_bucket']].rename(columns={'bucket_filename': 'file_name',
                                                                                               'text_timestamps_bucket':
                                                                                                   'transcription'})
            result.to_csv(f"{output_dir}/{df_subs['program'][0]}/metadata.csv", index=False)

    logging.info("Done")



if __name__ == "__main__":
    start = time.time()
    print(args)
    main()
    print(time.time() - start)