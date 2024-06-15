import argparse
import datetime
import glob
import json
import logging
import os
import re
import time

import soundfile as sf
import torch
from tqdm import tqdm

from sub_preproc.utils.audio import get_audio_chunk, read_audio
from sub_preproc.utils.dataset import RawAudioFileChunkerDataset, custom_collate_fn
from sub_preproc.utils.text import clean_subtitle

# def clean_text(text):
#     text = re.sub(r"\s+", " ", text)  # Youtube has newlines in the text
#     # To handle: "när man har hittat sitt drömhus– –vilken strategi ska jag ha"
#     text = re.sub("- -", " ")
#     text = re.sub("– –", " ")
#     return text


def find_audio_extension(filename):
    """
    yt-dlp sometimes downloads audio in different format from the one
    specified in the initial json metadata. We use this function to
    check if the audio file exists and return the correct extension.
    """
    print(filename)
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
    parser.add_argument("--out_dir", help="output folder", type=str, default="audio_chunks")
    parser.add_argument(
        "--sound_format",
        help="Input sound format for extracting from mp4",
        type=str,
        default="wav",
    )
    parser.add_argument(
        "--chunk_sound_format", help="Output sound format for chunks", type=str, default="wav"
    )
    parser.add_argument("--sample_rate", type=int, default=16_000)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument(
        "--source",
        type=str,
        help="Use smdb if you have a json file with json file paths and corresponding audio paths.",
    )

    return parser.parse_args()


#### ALTERNATIVE 1 WITH  DATALOADER ####
def main():
    os.makedirs("logs", exist_ok=True)
    now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    logger = logging.getLogger(__name__)

    args = get_args()

    if (
        args.source == "smdb"
    ):  # in case of smdb - json file with tuples of json and audio files paths

        audio_files = []
        json_files = []

        with open(args.json_files) as fh:
            data = json.load(fh)

        for entry in data:
            json_files.append(entry[0])
            audio_files.append(entry[1])

    else:
        json_files = []
        with open(args.json_files) as fh:
            print("fh ", fh)
            for line in fh:
                print("line ", line)
                json_files.append(line.strip())

        audio_files = []
        for file in json_files:
            filename = file.split(".")[0]
            audio_file = find_audio_extension(filename)
            audio_files.append(audio_file)

    audio_dataset = RawAudioFileChunkerDataset(
        audio_paths=audio_files, json_paths=json_files, out_dir=args.out_dir
    )

    dataloader_datasets = torch.utils.data.DataLoader(
        audio_dataset,
        batch_size=1,
        num_workers=6,
        collate_fn=custom_collate_fn,
        shuffle=False,
    )

    for i, data in enumerate(tqdm(dataloader_datasets)):
        pass


#### ALTERNATIVE 2 WITHOUT DATALOADER ####
#### MULTIPROCESSING NEEDS TO BE ADDED ####
def check_and_extract_chunks(fn_subs, args, sample_rate):
    to_log = []

    def log_time(log_point, prev):
        to_log.append(f"{log_point:<20s}{time.time() - prev:.4f}")
        return time.time()

    datadir = os.path.dirname(fn_subs)
    # Check if file exists
    if "file.json" not in os.path.basename(fn_subs):
        # Each file is not stored in a separate folder in
        # youtube and rixvox. We create a folder with the
        # same name as the file.
        print("HERE")
        basename = os.path.basename(fn_subs).split(".")[0]
        savedir = os.path.join(datadir, basename, "chunks")
        audio_path = find_audio_extension(fn_subs.split(".")[0])
    else:
        savedir = os.path.join(datadir, "chunks")
        audio_path = os.path.join(datadir, f"file.{args.sound_format}")

    os.makedirs(savedir, exist_ok=True)

    start = time.time()
    prev = start

    try:
        with open(fn_subs) as fin:
            subs_dict = json.load(fin)
    except json.JSONDecodeError:
        to_log.append(f"could not open {fn_subs} due to JSONDecodeError")
        return to_log

    # read audio into memory
    audio = read_audio(audio_path, sample_rate)

    prev = log_time("read_audio", prev)

    n_chunks = 0
    for i, chunk in enumerate(subs_dict["chunks"]):
        _, chunk_audio = get_audio_chunk(chunk, audio, sample_rate)
        n_chunks += 1
        with sf.SoundFile(
            os.path.join(
                savedir,
                f"chunk_{i}.{args.chunk_sound_format}",
            ),
            "w",
            args.sample_rate,
            channels=1,
        ) as fout:
            fout.write(chunk_audio)
        with open(os.path.join(savedir, f"chunk_{i}.txt"), "w") as fout:
            # TODO: Logic for when to output text vs text_whisper
            text = clean_subtitle(chunk["text"])
            print(text, file=fout)
    prev = log_time(str(n_chunks), prev)
    return to_log


if __name__ == "__main__":
    main()
# with open("UCidOzV_wWS97f9dQlnAD7dQ/_KmCSyOGm34.sv.json") as f:
#     sub_dict = json.load(f)

# args = get_args()
# check_and_extract_chunks("UCidOzV_wWS97f9dQlnAD7dQ/_KmCSyOGm34.sv.json", args, 16_000)
