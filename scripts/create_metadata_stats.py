import json
import pandas as pd
import os
import argparse


def ms_to_frames(ms, sr=16000):
    return int(ms / 1000 * 16000)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_files",
        type=str,
        help="Path to file containing json filepaths",
        default="files.txt",
    )

    parser.add_argument(
        "--source",
        type=str,
        help="Source of the files. Use smdb if you have a json file with json file paths and corresponding audio paths.",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output parquet file",
        default="output.parquet",
    )

    return parser.parse_args()


def find_audio_extension(filename):
    if os.path.isfile(filename + ".webm"):
        return filename + ".webm"
    elif os.path.isfile(filename + ".wav"):
        return filename + ".wav"
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


def process_json_file(json_file, audio_path=None, type=None):
    with open(json_file, "r") as f:
        data = json.load(f)

    print(f"Processing: {json_file}")

    json_dir = os.path.dirname(json_file)
    json_base_name = os.path.splitext(os.path.basename(json_file))[0]

    if type == "smdb":
        audio_path = audio_path
    else:
        filename = os.path.join(json_dir, f'{json_base_name.split(".")[0]}')
        audio_path = find_audio_extension(filename)

    rows = []

    for chunk in data["chunks"]:
        start_time = chunk["start"]
        end_time = chunk["end"]
        duration = end_time - start_time
        text = chunk.get("text", "")
        text_whisper = chunk.get("text_whisper", "")

        transcription = chunk.get("transcription", [])
        if chunk["transcription"] == []:
            print(f"Chunk has no transcription")
        else:
            if "whisper" in chunk["transcription"][0]["model"]:
                whisper_scores = chunk["transcription"][0].get("scores", {})
                bleu_whisper_score = whisper_scores.get("bleu", None)
                wer_whisper_score = whisper_scores.get("wer", None)
                first_whisper_score = whisper_scores.get("first", None)
                last_whisper_score = whisper_scores.get("last", None)
            else:
                bleu_whisper_score = wer_whisper_score = first_whisper_score = (
                    last_whisper_score
                ) = None

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
                bleu_wav2vec_score = wer_wav2vec_score = first_wav2vec_score = (
                    last_wav2vec_score
                ) = 0

            if "filters" in chunk:
                filters = chunk["filters"]
                stage1_whisper = filters.get("stage1_whisper", False)
                stage2_whisper = filters.get("stage2_whisper", False)
                stage2_whisper_timestamps = filters.get("stage2_whisper_timestamps", [])
                stage1_wav2vec = filters.get("stage1_wav2vec", False)
                silence = filters.get("silence", False)
            else:
                stage1_whisper = False
                stage2_whisper = False
                stage2_whisper_timestamps = False
                stage1_wav2vec = False
                silence = False

            rows.append(
                {
                    "channel": data.get("metadata", {}).get("channel", "Unknown Channel"),
                    "id": data.get("metadata", {}).get("id", "Unknown ID"),
                    "program_title": data.get("metadata", {}).get(
                        "program_title", "Unknown Title"
                    ),
                    "video_file": data.get("metadata", {}).get("video_file", "Unknown Video File"),
                    "caption_file": data.get("metadata", {}).get(
                        "caption_file", "Unknown Caption File"
                    ),
                    "is_asrun": data.get("metadata", {}).get(
                        "is_asrun",
                        (
                            True
                            if "ASRUN" in data.get("metadata", {}).get("channel", "").upper()
                            else False
                        ),
                    ),
                    "duration_ms": duration,
                    "production_year": data.get("metadata", {}).get("production_year", None),
                    "program_classification": data.get("metadata", {}).get(
                        "program_classification", None
                    ),
                    "source": data.get("metadata", {}).get("data_source", "Unknown Source"),
                    "audio": audio_path,
                    "start_time": chunk.get("start", None),  
                    "end_time": chunk.get("end", None),  
                    "text": chunk.get("text", ""),
                    "text_whisper": chunk.get("text_whisper", ""),
                    "transcription_whisper": (
                        transcription[0].get("text", "") if len(transcription) > 0 else ""
                    ),
                    "transcription_wav2vec2": (
                        transcription[1].get("text", "") if len(transcription) > 1 else ""
                    ),
                    "bleu_whisper": bleu_whisper_score,
                    "wer_whisper": wer_whisper_score,
                    "first_whisper": first_whisper_score,
                    "last_whisper": last_whisper_score,
                    "bleu_wav2vec": bleu_wav2vec_score,
                    "wer_wav2vec": wer_wav2vec_score,
                    "first_wav2vec": first_wav2vec_score,
                    "last_wav2vec": last_wav2vec_score,
                    "stage1_whisper": stage1_whisper,
                    "stage2_whisper": stage2_whisper,
                    "stage2_whisper_timestamps": stage2_whisper_timestamps,
                    "stage1_wav2vec": stage1_wav2vec,
                    "silence": silence,
                }
            )

    return pd.DataFrame(rows)


def create_parquet(args):

    all_dataframes = []

    # use json_file if you have a json file with json file paths and corresponding audio paths
    if args.source == "json_file":
        with open(args.json_files) as fh:
            data = json.load(fh)
        for entry in data:
            df = process_json_file(entry[0], entry[1], type="json_file")
            all_dataframes.append(df)
    else:
        with open(args.json_files, "r") as f:
            json_file_paths = f.read().splitlines()
        for json_file_path in json_file_paths:
            df = process_json_file(json_file_path)
            all_dataframes.append(df)

    # Concatenate all the DataFrames into one big DataFrame
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    final_df.to_parquet(args.output)


if __name__ == "__main__":
    args = get_args()
    create_parquet(args)
