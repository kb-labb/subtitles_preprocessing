import argparse
import torch
import os
import sys
import pandas as pd
import numpy as np
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

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

output_dir = args.output

parquet_files = [d for d in os.listdir(args.parquet_dir)]    
sampling_rate = 16000

def load_model():
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True)
    return model, utils

# Load and preprocess the audio
def preprocess_audio(audio):
    audio_resampled = audio.set_frame_rate(16000)
    samples = np.array(audio_resampled.get_array_of_samples())
    normalized_samples = samples / np.iinfo(samples.dtype).max
    return torch.tensor(normalized_samples, dtype=torch.float32)

def process_speech_chunks(speech_timestamps, audio_tensor, utils):
    collect_chunks = utils[-1]
    # Ensure that the first speech chunk starts from the beginning
    if len(speech_timestamps) >0:
        if speech_timestamps[0]['start'] > 0:
            speech_timestamps.insert(0, {'start': 0, 'end': speech_timestamps[0]['start']})
    # Ensure that the last speech chunk extends to the end of the audio
        if speech_timestamps[-1]['end'] < len(audio_tensor):
            speech_timestamps.append({'start': speech_timestamps[-1]['end'], 'end': len(audio_tensor)})

        speech_chunks_tensor = collect_chunks(speech_timestamps, audio_tensor)
        speech_chunks_np = speech_chunks_tensor.numpy()
        speech_chunks_int16 = (speech_chunks_np * np.iinfo(np.int16).max).astype(np.int16)
    return AudioSegment(speech_chunks_int16.tobytes(), frame_rate=sampling_rate, channels=1, sample_width=2)

# Process with Silero VAD
def detect_speech(audio_tensor, model, utils):
    get_speech_timestamps = utils[0]
    return get_speech_timestamps(audio_tensor, model, sampling_rate=sampling_rate)

# Detect language
def detect_language(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
    pred_tokens = model.generate(input_features, max_new_tokens=448)
    pred_text = processor.batch_decode(pred_tokens, skip_special_tokens=True)
    pred_language = processor.batch_decode(pred_tokens[:, 1:2], skip_special_tokens=False)
    return pred_language, pred_text

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
    model, utils = load_model() 
    for i, row in audio_df.iterrows():
        audio_chunk = audio[row["start_bucket"] : row["end_bucket"]]
        os.makedirs(os.path.join(output_dir, row['program']), exist_ok=True)
        filename = f"{output_dir}/{row['program']}/{row['observation_nr']}.wav"
        output_path = os.path.join(audio_path, filename)
        # do voice activity detection before language detect to improve performance
        audio_tensor = preprocess_audio(audio_chunk)
        speech_timestamps = detect_speech(audio_tensor, model, utils)
        if len(speech_timestamps) > 0: # some files contain no speech, i.e. no speech_timestamps, don't save these
            speech_audio_segment = process_speech_chunks(speech_timestamps, audio_tensor, utils)
            save_processed_audio(speech_audio_segment, filename)
            # language detection
            pred_lang, pred_text = detect_language(filename) 
            if pred_lang == ['<|sv|>']:
                print('found swedish: ', pred_text)
                audio_chunk.export(filename, format="wav")
                filenames.append(filename)
            else:
                os.remove(filename)

    return filenames


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
    #drop filenames that didn't pass language detection check by a left join 
    out_filenames_new = [x.replace(f"{output_dir}/{df_subs['program'][0]}/", '') for x in out_filenames[0]]
    out_filenames_df = pd.DataFrame({'bucket_filename':out_filenames_new})
    df_subs_unique = df_subs_unique.merge(out_filenames_df.drop_duplicates(), on=['bucket_filename'], how='left', indicator=True)
    df_subs_unique = df_subs_unique[df_subs_unique._merge == 'both']
    df_subs_unique = df_subs_unique.copy()
    df_subs_unique["bucket_path"] = [bucket_path for program_paths_list in out_filenames for bucket_path in program_paths_list]

    result = df_subs_unique[['bucket_filename', 'text_timestamps_bucket']].rename(columns={'bucket_filename': 'file_name',
                                                                                           'text_timestamps_bucket':
                                                                                               'transcription'})
    result.to_csv(f"{output_dir}/{df_subs['program'][0]}/metadata.csv", index=False)

logging.info("Done")