import subprocess as sp
import os
from os import makedirs
from src.sub_preproc.utils.utils import fuse_subtitles
import argparse
import time
import csv
import pysrt

CHANNELS = [
    "cmore/cmorefirst",
    "cmore/cmorehist",
    "cmore/cmoreseries",
    "cmore/cmorestars",
    "cmore/sfkanalen",
    "tv4/sjuan",
    "tv4/tv12",
    "tv4/tv4",
    "tv4/tv4fakta",
    "tv4/tv4film",
    "viasat/vfilmaction",
    "viasat/vfilmfamily",
    "viasat/vfilmhits",
    "viasat/vfilmpremiere",
    "viasat/vseries",
    "tv3/tv3",
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--o",
        type=str,
        default="srt",
        help="Directory where wav and srt files end up.",
    )
    parser.add_argument(
        "--i",
        type=str,
        default="input",
        help="Directory containing mp4 files.",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="subtitles_info.csv",
        help="Directory where csv file with info about subs will be saved.",
    )

    return parser.parse_args()


def check_and_extract(videofile, file, savedir):
    """Extract .mp3 or .wav at 16Hz and .srt files from .mp4."""

    # Check if Swedish subtitles exist
    out = sp.run(
        [
            "ffprobe",
            "-loglevel",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index:stream_tags=language",
            "-of",
            "csv=p=0",
            videofile,
        ],
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
        encoding="cp437",
    ).stdout.splitlines()
    subtitles = [x for x in out]
    # Check index of Swedish subtitles
    swe_idx = [i for i, x in enumerate(out) if "swe" in x]

    if len(swe_idx) >= 1:
        sv_subs = 1
        if os.path.isfile(f"{(os.path.join(savedir))}.srt") == False:
            # Save subtitle in srt format
            out = sp.run(
                [
                    "ffmpeg",
                    "-i",
                    videofile,
                    "-map",
                    f"s:{swe_idx[0]}",
                    "-f",
                    "srt",
                    f"{savedir}/file.srt",
                ],
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                universal_newlines=True,
            )
            # fuse_subtitles(f"{savedir}/file.srt", f"{savedir}/file.srt")
            fused_subs = fuse_subtitles(pysrt.open(f"{savedir}/file.srt"))
            fused_subs.save(path=f"{savedir}/file.srt")
        else:
            pass
    else:
        sv_subs = 0

    return (sv_subs, subtitles)


def create_statistics(kanal, file, sv_subs, subs, csv_file):
    """Create a dataframe with info about available subtitles."""

    exists = os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as save_file:
        fieldnames = ["channel", "filename", "swedish_subtitles", "subtitles"]
        writer = csv.DictWriter(save_file, fieldnames=fieldnames)

        # Write the header row
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "channel": kanal,
                "filename": file[:-4],
                "swedish_subtitles": sv_subs,
                "subtitles": subs,
            }
        )


def main():
    args = get_args()
    print(args)
    rootdir = args.i
    savedir = args.o
    csv = args.csv

    for subdir, dirs, files in os.walk(rootdir):
        kanal = subdir.split("/")[
            4:6
        ]  # adjust to your path, should point to the channel/subchannel part of the directory (e.g. "cmore/cmorefirst")
        # kanal = subdir.split("/")[8:10] #adjust to your path, should point to the channel/subchannel part of the directory (e.g. "cmore/cmorefirst")
        kanal = ("/").join(kanal)
        # filter only channels with Swedish subtitles
        if kanal in CHANNELS:
            for file in files:
                videofile_path = os.path.join(subdir, file)
                year = file[-28:-24]
                month = file[-23:-21]
                date = file[-20:-18]
                if not os.path.exists(os.path.join(savedir, kanal, year, month, date, file[:-4])):
                    makedirs(os.path.join(savedir, kanal, year, month, date, file[:-4]))
                srt_savedir = os.path.join(savedir, kanal, year, month, date, file[:-4])
                if os.path.isfile(f"{srt_savedir}/file.srt") == False:
                    sv_subs, subs = check_and_extract(videofile_path, file, srt_savedir)
                    create_statistics(kanal, file, sv_subs, subs, csv)

                    if sv_subs == 0:
                        os.rmdir(os.path.join(savedir, kanal, year, month, date, file[:-4]))
                else:
                    print(f"Skipping, {file[:-4]}.srt already exists.")


if __name__ == "__main__":
    start = time.time()
    main()
    print(time.time() - start)
