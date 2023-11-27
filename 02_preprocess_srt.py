import numpy as np
import os
from os import listdir,makedirs
import pandas as pd
import pysrt
import argparse
from tqdm import tqdm
import pdb
import re
import logging
import time

logging.basicConfig(level=logging.DEBUG,
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
    default="subs_preprocessed",
    help="Name of output dir.",
)

args = parser.parse_args()


program_names = [d for d in os.listdir(args.data)]
for p in program_names:
    files = [d for d in os.listdir(os.path.join(args.data, p))]
    df = pd.DataFrame(files, columns=["program"])
    files = {}
    for program in df["program"].tolist():
        files[program] = os.listdir(os.path.join(args.data, p))

    df["files"] = df["program"].map(files)
    df["audio"] = df["files"].map(lambda x: [file for file in x if file.endswith(".wav")][0])
    df["srt"] = df["files"].map(lambda x: [file for file in x if file.endswith(".srt")][0])
    df["program"] = p
    df.drop("files", axis=1, inplace=True)
    df.drop(df.tail(1).index,inplace=True) # drop duplicate row

    logging.info("Reading srt:s")
    # Read every srt file in df and save each line and timestamp in a dataframe
    df_subs = []
    for x, srt, audio in tqdm(zip(df["program"].tolist(), df["srt"].tolist(), df["audio"].tolist()), total=df.shape[0]):
        sub = pysrt.open(os.path.join(args.data, p, srt))
        sub_block_data = []
        for sub_block in sub:
            # sub_block.text = re.sub("<.*?>", "", sub_block.text)  # sub_block.text_without_tag instead
            sub_block_data.append(
                {
                    "program": p,
                    "start": sub_block.start,
                    "end": sub_block.end,
                    "text": sub_block.text_without_tag,
                    "srt": srt,
                    "audio": audio,
                }
            )

        df_sub = pd.DataFrame(sub_block_data)
        df_subs.append(df_sub)
    df_subs = pd.concat(df_subs).reset_index(drop=True)
    # Convert srt timestamps to milliseconds
    df_subs["start_ms"] = df_subs["start"].map(
        lambda x: x.hours * 3600000 + x.minutes * 60000 + x.seconds * 1000 + x.milliseconds
    )
    df_subs["end_ms"] = df_subs["end"].map(
        lambda x: x.hours * 3600000 + x.minutes * 60000 + x.seconds * 1000 + x.milliseconds
    )
    df_subs["duration_s"] = (df_subs["end_ms"] - df_subs["start_ms"]) / 1000
    df_subs['force_start'] = np.nan 
    df_subs['force_end'] = np.nan 
    df_subs['force_start'] = df_subs['start_ms'][abs(df_subs["start_ms"].diff())>4000]
    df_subs['force_end'] = df_subs['end_ms'][abs(df_subs["end_ms"].diff(-1))>4000]
    logging.info("Splitting srt:s into buckets")
    # Divide the subtitle blocks into 30 second buckets
    df_groups = []
    for group, df_group in tqdm(
        df_subs.groupby("audio"),
        total=df_subs.groupby("audio").ngroups,
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
    df_groups["bucket_duration_s"] = df_groups.groupby(["observation_nr", "audio"])["bucket_cumsum"].transform(max)

    # Relative start and end times for each subtitle block within a bucket (observation_nr grouping)
    df_groups["start_relative"] = df_groups["start_ms"] - df_groups.groupby(["observation_nr", "audio"])["start_ms"].transform(min)
    df_groups["end_relative"] = df_groups["end_ms"] - df_groups.groupby(["observation_nr", "audio"])["start_ms"].transform(min)

    # Round to nearest 20 ms (Whisper quantizes to nearest 20 ms for its timestamps)
    df_groups["start_relative"] = (np.round(df_groups["start_relative"] / 20) * 20) / 1000
    df_groups["end_relative"] = (np.round(df_groups["end_relative"] / 20)) * 20 / 1000

    # start_bucket is the start_ms of the bucket in an observation_nr group
    df_groups["start_bucket"] = df_groups.groupby(["observation_nr", "audio"])["start_ms"].transform(min)

    # end_bucket is the end_ms of the bucket in an observation_nr group
    df_groups["end_bucket"] = df_groups.groupby(["observation_nr", "audio"])["end_ms"].transform(max)

    def format_timestamp(timestamp):
        """
        Format according to Whisper timestamp token format: <|x.xx|>
        """
        timestamp = "<|" + f"{timestamp:.2f}" + "|>"
        return timestamp


    logging.info("Mapping groups")
    df_groups["start_timestamp"] = df_groups["start_relative"].map(format_timestamp)
    df_groups["end_timestamp"] = df_groups["end_relative"].map(format_timestamp)
    df_groups["text_timestamps"] = df_groups["start_timestamp"] + df_groups["text"] + df_groups["end_timestamp"]

    # Create a new column that joins the text_timestamps for each observation_nr group
    df_groups["text_timestamps_bucket"] = df_groups.groupby(["observation_nr", "audio"])["text_timestamps"].transform(
        lambda x: " ".join(x)
    )

    if not os.path.exists(os.path.join(args.output,p)):
        makedirs(os.path.join(args.output,p))
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
    ].to_parquet(os.path.join(args.output,p, "subs_preprocessed.parquet"), index=False)
    logging.info("Ending")
