import os
import glob
import json
import datetime
import pandas as pd
from tqdm import tqdm
from sub_preproc.utils.utils import SILENCE, decode_program_id, ms_to_time
from typing import Optional


#  "start": end,
#  "end": srt_time_to_ms(sub.start),
#  "duration": srt_time_to_ms(sub.start) - end,
#  "text": SILENCE,
#  "duplicate": False,
#  "live": False,

# "start": chunk_start,
# "end": chunk_end,
# "duration": chunk_end - chunk_start,
# "subs": chunk,
# "text_whisper": subs_to_whisper(chunk),
# "text": subs_to_raw_text(chunk),


def process_sub_chunk_dict(subs: dict) -> dict:
    sub_entries = []
    for sub in subs["subs"]:
        n_words = len(sub["text"].split()) if sub["text"] != SILENCE else 0
        sub_entry = {
            "n_words": n_words,
            "duration": sub["duration"],
            "duplicate": sub["duplicate"],
            "live": sub["live"],
            "silence": sub["text"] != SILENCE,
        }
        sub_entries.append(sub_entry)

    chunk_entries = []
    for chunk in subs["chunks"]:
        n_words = len(chunk["text"].split())
        chunk_entry = {
            "n_words": n_words,
            "duration": chunk["duration"],
            "silence": chunk["text"] == "",
        }
        chunk_entries.append(chunk_entry)


def cum_mean(x, prev_mean, n):
    return (x + n * prev_mean) / (n + 1)


def get_stats_single(fn, df=None) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame(
            columns=[
                "channel",
                "subchannel",
                "broadcast_date",
                "from_time",
                "to_time",
                "subs_duration",
                "avg_subs_duration",
                "subs_duration_duplicate",
                "subs_duration_livesub",
                "subs_duration_silence",
                "subs_count",
                "duplicates_count",
                "livesubs_count",
                "silence_count",
                "n_chunks",
                "chunks_duration",
                "n_silent_chunks",
                "silent_chunks_duration",
            ]
        )

    program_id = fn.split("/")[-2]
    channel, subchannel, year, month, day, from_time, to_time = decode_program_id(program_id)
    from_time = from_time[:6]
    to_time = to_time[:6]
    broadcast_date = datetime.date.fromisoformat("-".join([year, month, day]))
    from_time = datetime.time.fromisoformat(
        ":".join([from_time[i - 1 : i + 1] for i in range(1, len(from_time), 2)])
    )
    to_time = datetime.time.fromisoformat(
        ":".join([to_time[i - 1 : i + 1] for i in range(1, len(to_time), 2)])
    )
    subs_duration = 0
    avg_subs_duration = 0
    subs_duration_duplicate = 0
    subs_duration_livesub = 0
    subs_duration_silence = 0
    subs_count = 0
    duplicates_count = 0
    livesubs_count = 0
    silence_count = 0
    n_chunks = 0
    chunks_duration = 0
    n_silent_chunks = 0
    silent_chunks_duration = 0
    with open(fn) as fh:
        subs = json.load(fh)
    for sub in subs["subs"]:
        subs_count += 1
        subs_duration += sub["duration"]
        avg_subs_duration = cum_mean(sub["duration"], avg_subs_duration, subs_count - 1)

        if sub["duplicate"]:
            duplicates_count += 1
            subs_duration_duplicate += sub["duration"]
        elif sub["live"]:
            livesubs_count += 1
            subs_duration_livesub += sub["duration"]
        elif sub["text"] == SILENCE:
            silence_count += 1
            subs_duration_silence += sub["duration"]
    for chunk in subs["chunks"]:
        n_chunks += 1
        chunks_duration += chunk["duration"]
        if chunk["text"] == "":
            n_silent_chunks += 1
            silent_chunks_duration += chunk["duration"]

    df.loc[len(df)] = {
        "channel": channel,
        "subchannel": subchannel,
        "broadcast_date": broadcast_date,
        "from_time": from_time,
        "to_time": to_time,
        "subs_duration": subs_duration,
        "avg_subs_duration": avg_subs_duration,
        "subs_duration_duplicate": (subs_duration_duplicate),
        "subs_duration_livesub": (subs_duration_livesub),
        "subs_duration_silence": subs_duration_silence,
        "subs_count": subs_count,
        "duplicates_count": duplicates_count,
        "livesubs_count": livesubs_count,
        "silence_count": silence_count,
        "n_chunks": n_chunks,
        "chunks_duration": chunks_duration,
        "n_silent_chunks": n_silent_chunks,
        "silent_chunks_duration": silent_chunks_duration,
    }
    return df


def get_stats_file_list(file_names: list) -> Optional[pd.DataFrame]:
    # file_names = glob.iglob(f"{in_data}/**/file.json", recursive=True)
    df = None
    for fn in tqdm(file_names):
        try:
            df = get_stats_single(fn, df)
        except FileNotFoundError:
            print(f"{fn} not found")
    return df


if __name__ == "__main__":
    smdb01 = [
        "/home/robkur/data/delat/srt_only_smdb01/" + x.strip("\n./") + ".new"
        for x in open(f"/home/robkur/data/delat/srt_only_smdb01/json-files.txt").readlines()
    ]
    smdb04 = [
        "/home/robkur/data/delat/srt_only/" + x.strip("\n./") + ".new"
        for x in open(f"/home/robkur/data/delat/srt_only/json-files.txt").readlines()
    ]
    # df = get_stats_file_list(smdb01 + smdb04)
    # df.to_csv("stats.csv", index=False)
    df = pd.read_csv("stats.csv")

    print(
        [
            x
            for x in df[
                [
                    "subs_duration",
                    "subs_duration_duplicate",
                    "subs_duration_livesub",
                    "subs_duration_silence",
                    "n_chunks",
                    "n_silent_chunks",
                    "chunks_duration",
                    "silent_chunks_duration",
                ]
            ].mean()
        ]
    )
    print(
        [
            ms_to_time(x, False)
            for x in df[
                [
                    "subs_duration",
                    "subs_duration_duplicate",
                    "subs_duration_livesub",
                    "subs_duration_silence",
                    "chunks_duration",
                    "silent_chunks_duration",
                ]
            ].sum()
        ]
    )
    print(
        ms_to_time(
            int(
                (
                    df["subs_duration"]
                    - df["subs_duration_silence"]
                    - df["subs_duration_livesub"]
                    - df["subs_duration_duplicate"]
                ).mean()
            ),
            False,
        )
    )
    print(
        ms_to_time(
            (
                df["subs_duration"]
                - df["subs_duration_silence"]
                - df["subs_duration_livesub"]
                - df["subs_duration_duplicate"]
            ).sum(),
            False,
        )
    )
    print(
        ms_to_time(
            (df["chunks_duration"] - df["silent_chunks_duration"]).sum(),
            False,
        )
    )
    print(
        ms_to_time(
            (df["chunks_duration"]).sum(),
            False,
        )
    )
    print(
        ms_to_time(
            (df["silent_chunks_duration"]).sum(),
            False,
        )
    )
    print(
        (df["n_chunks"]).sum(),
    )
    print(
        (df["n_silent_chunks"]).sum(),
    )
