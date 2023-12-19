import numpy as np
import os
import glob
import multiprocessing as mp
from os import makedirs
import pandas as pd
import pysrt
import argparse
from tqdm import tqdm
import logging


def get_files(dir):
    """
    Get all srt and wav files in a directory and its subdirectories.

    Folder structure:
    ../XA_cmore_cmoreseries_2023-03-01_100000_110000/file.srt
    ../XA_cmore_cmoreseries_2023-03-01_100000_110000/file.wav

    We set broadcast_id to be the folder name before file.srt and file.wav.
    """
    logging.info("Finding srts and wavs")
    dir = os.path.expanduser(dir)  # Deal with ~ in default args.data path
    srt_files = glob.iglob(dir + "/**/*.srt", recursive=True)
    wav_files = glob.iglob(dir + "/**/*.wav", recursive=True)

    files_dict = {}

    for srt_file in srt_files:
        broadcast_id = srt_file.split("/")[-2]
        files_dict[broadcast_id] = {
            "srt_file": srt_file.split("/")[-1],
            "base_dir": dir,
        }

    for wav_file in wav_files:
        broadcast_id = wav_file.split("/")[-2]
        files_dict[broadcast_id]["wav_file"] = wav_file.split("/")[-1]
        files_dict[broadcast_id]["base_dir"] = dir

    return files_dict


def srt_time_to_ms(time):
    """
    Convert srt time to milliseconds.
    """
    ms = (
        time.hours * 3600000
        + time.minutes * 60000
        + time.seconds * 1000
        + time.milliseconds
    )
    return ms


def read_srt(filepath):
    """
    Read srt file and return start, end and text for each subtitle block
    as a list of dictionaries.
    """
    sub = pysrt.open(filepath)
    sub_block_data = []
    for sub_block in sub:
        sub_block_data.append(
            {
                "start_ms": srt_time_to_ms(sub_block.start),
                "end_ms": srt_time_to_ms(sub_block.end),
                "text": sub_block.text_without_tags,
            }
        )

    return sub_block_data


def extract_subtitles(broadcast):
    """
    Extract subtitles from srt file and return a dictionary with
    broadcast_id, subtitle block data, srt file name and wav file name.

    Args:
        broadcast (tuple):
            (broadcast_id, {'srt_file': 'file.srt', 'wav_file': 'file.wav'})
        folder (str): path to folder containing broadcast
    """
    broadcast_id = broadcast[0]
    srt_file = broadcast[1]["srt_file"]
    base_dir = broadcast[1]["base_dir"]

    srt_path = os.path.join(base_dir, broadcast_id, srt_file)

    sub_blocks = read_srt(srt_path)

    sub_dict = {
        "id": broadcast_id,
        "sub_block_data": sub_blocks,
        "srt": srt_file,
        "audio": broadcast[1]["wav_file"],
    }

    return sub_dict


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="~/data_network/delat/srt_only",
        help="Directory containing subdirectories with audio and srt files.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="subs_preprocessed",
        help="Name of output dir.",
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=16,
        help="Number of processes to use for multiprocessing.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info("Starting")
    args = get_args()
    broadcasts = get_files(args.data)
    broadcasts_list = list(broadcasts.items())  # list of tuples for multiprocessing

    logging.info("Reading and extracting subtitles")
    subtitles = list(
        tqdm(
            mp.Pool(args.processes).imap(extract_subtitles, broadcasts_list),
            total=len(broadcasts_list),
        )
    )

    ## TODO: Refactor below to perhaps avoid pandas
    df_subs = pd.DataFrame(subtitles)

    # Expand nested column sub_block_data to individual rows
    df_subs = df_subs.explode("sub_block_data")

    # Make the keys of sub_block_data column (nested dicts) into separate columns
    # (start_ms, end_ms, text)
    df_subs.reset_index(inplace=True)
    df_subs = pd.concat(
        [
            df_subs.drop(["sub_block_data"], axis=1),
            pd.json_normalize(df_subs["sub_block_data"]),
        ],
        axis=1,
    )

    df_subs["duration_s"] = (df_subs["end_ms"] - df_subs["start_ms"]) / 1000

    logging.info("Splitting srt:s into buckets")
    # Divide the subtitle blocks into 30 second buckets
    df_groups = []
    for group, df_group in tqdm(
        df_subs.groupby("id"),
        total=df_subs.groupby("id").ngroups,
    ):
        start = df_group["start_ms"].iloc[0]
        bucket_nr = 0
        bucket_cumsum = []
        bucket_nrs = []
        for i, end in enumerate(df_group["end_ms"]):
            if ((end - start) / 1000) >= 30:
                bucket_nr += 1
                start = df_group["start_ms"].iloc[i]

            prev_segment_length = (end - start) / 1000
            bucket_cumsum.append(prev_segment_length)
            bucket_nrs.append(bucket_nr)

        df_group["observation_nr"] = bucket_nrs
        df_group["bucket_cumsum"] = bucket_cumsum
        df_groups.append(df_group)

    df_groups = pd.concat(df_groups)
    df_groups = df_groups.reset_index(drop=True)

    logging.info("Cumsum and relative times")
    # Maximum value of bucket_cumsum in each bucket (observation_nr group) is the duration of the observation
    df_groups["bucket_duration_s"] = df_groups.groupby(["observation_nr", "audio"])[
        "bucket_cumsum"
    ].transform(max)

    # Relative start and end times for each subtitle block within a bucket (observation_nr grouping)
    df_groups["start_relative"] = df_groups["start_ms"] - df_groups.groupby(
        ["observation_nr", "audio"]
    )["start_ms"].transform(min)
    df_groups["end_relative"] = df_groups["end_ms"] - df_groups.groupby(
        ["observation_nr", "audio"]
    )["start_ms"].transform(min)

    # Round to nearest 20 ms (Whisper quantizes to nearest 20 ms for its timestamps)
    df_groups["start_relative"] = (
        np.round(df_groups["start_relative"] / 20) * 20
    ) / 1000
    df_groups["end_relative"] = (np.round(df_groups["end_relative"] / 20)) * 20 / 1000

    # start_bucket is the start_ms of the bucket in an observation_nr group
    df_groups["start_bucket"] = df_groups.groupby(["observation_nr", "audio"])[
        "start_ms"
    ].transform(min)

    # end_bucket is the end_ms of the bucket in an observation_nr group
    df_groups["end_bucket"] = df_groups.groupby(["observation_nr", "audio"])[
        "end_ms"
    ].transform(max)

    def format_timestamp(timestamp):
        """
        Format according to Whisper timestamp token format: <|x.xx|>
        """
        timestamp = "<|" + f"{timestamp:.2f}" + "|>"
        return timestamp

    logging.info("Mapping groups")
    df_groups["start_timestamp"] = df_groups["start_relative"].map(format_timestamp)
    df_groups["end_timestamp"] = df_groups["end_relative"].map(format_timestamp)
    df_groups["text_timestamps"] = (
        df_groups["start_timestamp"] + df_groups["text"] + df_groups["end_timestamp"]
    )

    # Create a new column that joins the text_timestamps for each observation_nr group
    df_groups["text_timestamps_bucket"] = df_groups.groupby(
        ["observation_nr", "audio"]
    )["text_timestamps"].transform(lambda x: " ".join(x))

    if not os.path.exists(os.path.join(args.output)):
        makedirs(os.path.join(args.output))
    logging.info("Saving parquet file")
    df_groups[
        [
            "program",
            "audio",
            "observation_nr",
            "start_ms",
            "end_ms",
            "start_bucket",
            "end_bucket",
            "text",
            "text_timestamps_bucket",
            "start_relative",
            "end_relative",
            "bucket_duration_s",
        ]
    ].to_parquet(os.path.join(args.output, "subs_preprocessed.parquet"), index=False)
    logging.info("Ending")
