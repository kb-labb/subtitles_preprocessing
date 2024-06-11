import json
import pandas as pd
import os
import argparse
import tempfile
import soundfile as sf
import numpy as np
import subprocess
from sub_preproc.utils.audio import convert_and_read_audio


def ms_to_frames(ms, sr=16000):
    return int(ms / 1000 * sr)


def find_audio_extension(filename):
    """
    yt-dlp sometimes downloads audio in different format from the one
    specified in the initial json metadata. We use this function to
    check if the audio file exists and return the correct extension.
    """
    if os.path.isfile(filename + ".webm"):
        return filename + ".webm"
    elif os.path.isfile(filename + ".m4a"):
        return filename + ".m4a"
    elif os.path.isfile(filename + ".mp3"):
        return filename + ".mp3"
    elif os.path.isfile(filename + ".mp4"):
        return filename + ".mp4"
    elif os.path.isfile(filename + ".mkv"):
        return filename + ".mkv"
    elif os.path.isfile(filename + ".flac"):
        return filename + ".flac"
    else:
        return None


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_files",
        type=str,
        help="Path to file containg json filepaths",
        default="files.txt",
    )

    return parser.parse_args()


def create_parquet(json_file_paths):

    sub_ids = []
    audio_paths = []
    start_times = []
    end_times = []
    texts = []
    texts_whisper = []
    bleu_whisper = []
    wer_whisper = []
    first_whisper = []
    last_whisper = []
    bleu_wav2vec = []
    wer_wav2vec = []
    first_wav2vec = []
    last_wav2vec = []
    stages1_whisper = []
    stages2_whisper = []
    stages2_whisper_timestamps = []
    stages1_wav2vec = []
    silences = []
    sources = []
    audio_tensors = []

    with open(json_file_path, "r") as f:
        data = json.load(f)

    json_dir = os.path.dirname(json_file_path)
    json_base_name = os.path.splitext(os.path.basename(json_file_path))[0]

    # Find the audio extension for the file
    filename = os.path.join(json_dir, f'{json_base_name.split(".")[0]}')
    audio_path = find_audio_extension(filename)

    # Read the source audio file for the chunks
    audio, sr = convert_and_read_audio(audio_path)

    for chunk in data["chunks"]:

        sub_id = str(chunk["sub_ids"])
        start_time = chunk["start"]
        end_time = chunk["end"]
        text = chunk.get("text", "")
        text_whisper = chunk.get("text_whisper", "")

        if "whisper" in chunk["transcription"][0]["model"]:
            whisper_scores = chunk["transcription"][0].get("scores", {})
            bleu_whisper_score = whisper_scores.get("bleu", None)
            wer_whisper_score = whisper_scores.get("wer", None)
            first_whisper_score = whisper_scores.get("first", None)
            last_whisper_score = whisper_scores.get("last", None)
        else:
            bleu_whisper_score = wer_whisper_score = first_whisper_score = last_whisper_score = (
                None
            )
        if len(chunk["transcription"]) > 1:
            if "wav2vec" in chunk["transcription"][1]["model"]:
                wav2vec_scores = chunk["transcription"][1].get("scores", {})
                bleu_wav2vec_score = wav2vec_scores.get("bleu", None)
                wer_wav2vec_score = wav2vec_scores.get("wer", None)
                first_wav2vec_score = wav2vec_scores.get("first", None)
                last_wav2vec_score = wav2vec_scores.get("last", None)
            else:
                bleu_wav2vec_score = wer_wav2vec_score = first_wav2vec_score = (
                    last_wav2vec_score
                ) = 0
        else:
            bleu_wav2vec_score = wer_wav2vec_score = first_wav2vec_score = last_wav2vec_score = 0
        if "filters" in chunk:
            filters = chunk["filters"]
            stages1_whisper.append(filters.get("stage1_whisper", False))
            stages2_whisper.append(filters.get("stage2_whisper", False))
            stages2_whisper_timestamps.append(filters.get("stage2_whisper_timestamps", []))
            stages1_wav2vec.append(filters.get("stage1_wav2vec", False))
            silences.append(filters.get("silence", False))
        else:
            stages1_whisper.append(False)
            stages2_whisper.append(False)
            stages2_whisper_timestamps.append(False)
            stages1_wav2vec.append(False)
            silences.append(False)

        start_frame = ms_to_frames(start_time, sr)
        end_frame = ms_to_frames(end_time, sr)
        audio_tensor = audio[start_frame:end_frame]

        sub_ids.append(sub_id)
        sources.append(data["metadata"]["data_source"])
        audio_paths.append(audio_path)
        start_times.append(start_time)
        end_times.append(end_time)
        texts.append(text)
        texts_whisper.append(text_whisper)
        bleu_whisper.append(bleu_whisper_score)
        wer_whisper.append(wer_whisper_score)
        first_whisper.append(first_whisper_score)
        last_whisper.append(last_whisper_score)
        bleu_wav2vec.append(bleu_wav2vec_score)
        wer_wav2vec.append(wer_wav2vec_score)
        first_wav2vec.append(first_wav2vec_score)
        last_wav2vec.append(last_wav2vec_score)
        audio_tensors.append(audio_tensor)

    df = pd.DataFrame(
        {
            "sub_id": sub_ids,
            "source": sources,
            "audio": audio_paths,
            "start_time": start_times,
            "end_time": end_times,
            "text": texts,
            "text_whisper": texts_whisper,
            "bleu_whisper": bleu_whisper,
            "wer_whisper": wer_whisper,
            "first_whisper": first_whisper,
            "last_whisper": last_whisper,
            "bleu_wav2vec": bleu_wav2vec,
            "wer_wav2vec": wer_wav2vec,
            "first_wav2vec": first_wav2vec,
            "last_wav2vec": last_wav2vec,
            "stage1_whisper": stages1_whisper,
            "stage2_whisper": stages2_whisper,
            "stage2_whisper_timestamps": stages2_whisper_timestamps,
            "stage1_wav2vec": stages1_wav2vec,
            "silence": silences,
            "audio_tensor": audio_tensors,
        }
    )

    parquet_dir = audio_path.split(".")[:-1]
    df.to_parquet(".".join(parquet_dir + ["parquet"]))


if __name__ == "__main__":

    args = get_args()

    # Read the txt with JSON file paths
    with open(args.json_files, "r") as f:
        json_file_paths = f.read().splitlines()
        for json_file_path in json_file_paths:
            create_parquet(json_file_paths)
