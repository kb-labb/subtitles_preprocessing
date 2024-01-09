import json
import glob
import pysrt
import os
import argparse
import subprocess as sp
import sub_preproc.utils.utils as utils
from sub_preproc.dedup import dup_marker_single
from typing import Optional
from tqdm import tqdm
from sub_preproc.utils.make_chunks import make_chunks

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
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_data")
    parser.add_argument("--out_data")

    return parser.parse_args()


def check_for_sv_subs(videofile: str) -> Optional[int]:
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
    # Check index of Swedish subtitles
    swe_idx = [i for i, x in enumerate(out) if "swe" in x]

    if swe_idx:
        return swe_idx[0]
    else:
        return None


def extract_subs(videofile: str, sub_id: int, savedir: str) -> None:
    """Extract .mp3 or .wav at 16Hz and .srt files from .mp4."""
    if not os.path.isfile(f"{(os.path.join(savedir))}.srt"):
        # Save subtitle in srt format
        _ = sp.run(
            [
                "ffmpeg",
                "-i",
                videofile,
                "-map",
                f"s:{sub_id}",
                "-f",
                "srt",
                f"{savedir}/file.srt",
            ],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            universal_newlines=True,
        )


def main():
    args = get_args()
    print(args)

    seen = set()

    for channel_plus in tqdm(CHANNELS):
        filenames = glob.iglob(f"{args.in_data}/{channel_plus}/**/*mp4", recursive=True)
        saved_filenames = []
        for fn in tqdm(filenames):
            program_id = fn.split("/")[-2]
            channel, subchannel, year, month, day, from_time, to_time = utils.decode_program_id(
                program_id
            )
            savedir = os.path.join(
                args.out_data, channel, subchannel, year, month, day, program_id
            )
            assert channel_plus == "/".join([channel, subchannel])

            swe_sub_id = check_for_sv_subs(fn)
            if swe_sub_id:
                os.makedirs(savedir, exist_ok=True)
                extract_subs(fn, swe_sub_id, savedir)
                subs = pysrt.open(os.path.join(savedir, "file.srt"))
                fused = utils.fuse_subtitles(subs)
                dup_marked = dup_marker_single(fused, seen)
                live_subbed = utils.mark_live_subs(dup_marked)
                subs_dict = utils.subrip_to_dict(
                    live_subbed, channel, subchannel, year, month, day, from_time, to_time
                )
                subs_chunks_dict = make_chunks(subs_dict)
                with open(os.path.join(savedir, "file.json"), "w") as fout:
                    json.dump(subs_chunks_dict, fout)
                saved_filenames.append(os.path.join(savedir, "file.json"))
                # TODO
                # if there are chunks
                # extract audio
                # read audio into ndarray as faster_whisper likes it
                # from faster_whisper.audio import decode_audio
                # or with librosa or soundfile as huggingface does it
                # check chunks based on frames-array (sec * sampling_rate)
                # export sub-array into wav if necessary
                # write with soundfile
                # problem: loss of quality but not necessary for training anyway
    with open(os.path.join(args.out_data, "sub_and_chunk_dicts.txt"), "w") as fout:
        for fn in saved_filenames:
            print(fn, file=fout)


if __name__ == "__main__":
    main()
