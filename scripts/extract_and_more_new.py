import argparse
import csv
import datetime
import glob
import itertools as it
import json
import logging
import os
import pathlib
import pickle
import subprocess as sp
import time
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import pysrt
import soundfile as sf
import sub_preproc.utils.utils as utils
import torch
import torch.multiprocessing as mp
from sub_preproc.dedup import dup_marker_single, dup_marker_single_list
from sub_preproc.detect_language import detect_language
from sub_preproc.utils.make_chunks import make_chunks
from sub_preproc.utils.utils import SILENCE

# from faster_whisper import WhisperModel
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

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

    parser.add_argument("--in_data", type=str, required=True)
    parser.add_argument("--out_data", type=str, required=True)
    parser.add_argument("--sound_format", type=str, default="wav")
    parser.add_argument("--chunk_sound_format", type=str, default="flac")
    parser.add_argument("--sample_rate", type=int, default=16_000)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--seen", type=str, default=None)
    parser.add_argument("--processes", type=int, default=1)

    return parser.parse_args()


def extract_sound(input_file: str, output_path: str, sound_format: str) -> None:
    if os.path.isfile(f"{(os.path.join(output_path))}/file.{sound_format}"):
        return
    if sound_format == "mp3":
        # Save sound in mp3 format
        _ = sp.run(
            [
                "ffmpeg",
                "-i",
                input_file,
                "-acodec",
                "libmp3lame",
                f"{output_path}/file.mp3",
            ],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            text=True,
        )
    # Save sound in wav format
    elif sound_format == "wav":
        _ = sp.run(
            [
                "ffmpeg",
                "-i",
                input_file,
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                f"{output_path}/file.wav",
            ],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            text=True,
        )
    # exporting to flac takes a loong time
    elif sound_format == "flac":
        _ = sp.run(
            [
                "ffmpeg",
                "-i",
                input_file,
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "flac",
                "-compression_level",
                "5",
                f"{output_path}/file.flac",
            ],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            text=True,
        )


def read_audio(sound_file: str, target_sample_rate) -> np.ndarray:
    audio, sample_rate = sf.read(sound_file)
    audio = librosa.to_mono(audio)
    audio = librosa.resample(
        audio,
        orig_sr=sample_rate,
        target_sr=target_sample_rate,
        res_type="kaiser_best",
    )
    audio = np.asarray(audio, dtype=np.float32)
    return audio


def get_audio_chunk(
    chunk, audio, sample_rate, model, processor
) -> Optional[Tuple[Dict[str, Any], np.ndarray, Dict[str, float]]]:
    start = chunk["start"]
    end = chunk["end"]
    chunk_audio = audio[start * sample_rate // 1_000 : end * sample_rate // 1_000]
    if chunk["text"] == "":
        return None
        # yield chunk, chunk_audio
    elif model is None:
        return chunk, chunk_audio, {}
    else:
        # _, info = model.transcribe(chunk_audio, vad_filter=True, beam_size=5)
        # if info.language == "sv" and info.language_probability > 0.5:
        #     yield chunk, chunk_audio
        inputs = (
            processor.feature_extractor(
                chunk_audio, return_tensors="pt", sampling_rate=16_000
            )
            .input_features.to("cuda:0")
            .to(torch.float16)
        )
        language_probs = detect_language(model, processor.tokenizer, inputs)[0]
        # l, p = max(language_probs.items(), key=lambda x: x[1])
        return chunk, chunk_audio, language_probs


def precheck_for_chunks(sub_dict) -> bool:
    for chunk in sub_dict["chunks"]:
        # if it's a silent chunk then there will be only one sub
        # if it is a non-silent chunk with one sub then check if it isn't silence
        if len(chunk["subs"]) > 1 or chunk["subs"][0]["text"] != SILENCE:
            return True
    return False


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
    if not os.path.isfile(f"{(os.path.join(savedir))}/file.srt"):
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


def process_subs(fn, args, seen):
    to_log = []

    def log_time(log_point, prev):
        to_log.append(f"{log_point:<20s}{time.time() - prev:.4f}")
        return time.time()

    start = time.time()
    prev = start

    program_id = fn.split("/")[-1].split(".")[0]
    channel, subchannel, year, month, day, from_time, to_time = utils.decode_program_id(
        program_id
    )

    savedir = os.path.join(
        args.out_data, channel, subchannel, year, month, day, program_id
    )

    skip_extract = False
    if os.path.exists(os.path.join(savedir, "file.srt")):
        skip_extract = True
    skip_sub_processing = False
    if os.path.exists(os.path.join(savedir, "file.json")):
        skip_sub_processing = True

    swe_sub_id = False
    if not skip_extract:
        swe_sub_id = check_for_sv_subs(fn)
        prev = log_time("check_subs", prev)

    if skip_extract or swe_sub_id:
        if not skip_extract:
            os.makedirs(savedir, exist_ok=True)
            extract_subs(fn, swe_sub_id, savedir)
            prev = log_time("extract_subs", prev)
        if not skip_sub_processing:
            # read srt file
            try:
                subs = pysrt.open(os.path.join(savedir, "file.srt"))
                prev = log_time("read_srt", prev)
            except UnicodeDecodeError:
                to_log.append(f"failed to read {savedir}/file.srt")
                return to_log

            # fuse subs
            fused = utils.fuse_subtitles(subs)
            prev = log_time("fuse", prev)

            # livesub marking
            live_subbed = utils.mark_live_subs(fused)
            prev = log_time("livesubs", prev)

            # to dict
            subs_dict = utils.subrip_to_dict(
                live_subbed, channel, subchannel, year, month, day, from_time, to_time
            )
            prev = log_time("to_dict", prev)

            # duplicate marking
            subs_dict, seen = dup_marker_single_list(subs_dict["subs"], seen)
            prev = log_time("dups", prev)

        else:
            with open(os.path.join(savedir, "file.json"), "w") as fin:
                subs_dict = json.load(fin)

        # make chunks
        for mini, maxi in [
            (1, 5_000),
            (5_000, 10_000),
            (10_000, 20_000),
            (20_000, 30_000),
        ]:
            subs_dict = make_chunks(subs_dict, min_threshold=mini, max_threshold=maxi)
        prev = log_time("chunks", prev)

        with open(os.path.join(savedir, "file.json"), "w") as fout:
            json.dump(subs_dict, fout, indent=4)
        prev = log_time("write_json", prev)


def extract_audio(fn_video, fn_subs, args, model, processor):
    to_log = []

    def log_time(log_point, prev):
        to_log.append(f"{log_point:<20s}{time.time() - prev:.4f}")
        return time.time()

    savedir = "/".join(fn_subs.split("/")[:-1])

    start = time.time()
    prev = start

    try:
        with open(fn_subs) as fin:
            subs_dict = json.load(fin)
    except json.JSONDecodeError:
        to_log.append(f"could not open {fn_subs} due to JSONDecodeError")
        return to_log

    if precheck_for_chunks(subs_dict):
        prev = log_time("audio?", prev)
        # extract audio from mp4
        os.makedirs(os.path.join(savedir, "chunks"), exist_ok=True)
        extract_sound(
            input_file=fn_video, output_path=savedir, sound_format=args.sound_format
        )
        prev = log_time("extract_audio", prev)
    return


def check_and_extract_chunks(fn_subs, args, model, processor, sample_rate):
    to_log = []

    def log_time(log_point, prev):
        to_log.append(f"{log_point:<20s}{time.time() - prev:.4f}")
        return time.time()

    savedir = "/".join(fn_subs.split("/")[:-1])

    start = time.time()
    prev = start

    try:
        with open(fn_subs) as fin:
            subs_dict = json.load(fin)
    except json.JSONDecodeError:
        to_log.append(f"could not open {fn_subs} due to JSONDecodeError")
        return to_log

    n_chunks = 0
    # read audio into memory
    audio = read_audio(
        os.path.join(savedir, f"file.{args.sound_format}"),
        target_sample_rate=args.sample_rate,
    )

    # # remove audio file
    # p = pathlib.Path(os.path.join(savedir, f"file.{args.sound_format}"))
    # p.unlink()

    prev = log_time("read_audio", prev)

    for threshold, chunks_with_threshold_xy in subs_dict["chunks"].items():
        for i, chunk in enumerate(chunks_with_threshold_xy):
            _, chunk_audio, language_probs = get_audio_chunk(
                chunk, audio, sample_rate, model, processor
            )
            n_chunks += 1
            with sf.SoundFile(
                os.path.join(
                    savedir,
                    "chunks",
                    f"chunk_{threshold}_{i}.{args.chunk_sound_format}",
                ),
                "w",
                args.sample_rate,
                channels=1,
            ) as fout:
                fout.write(chunk_audio)
            with open(
                os.path.join(savedir, "chunks", f"chunk_{threshold}_{i}.txt"), "w"
            ) as fout:
                print(chunk["text_whisper"], file=fout)
    prev = log_time(str(n_chunks), prev)
    return to_log


def main():
    args = get_args()
    print(args)

    now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    logging.basicConfig(
        filename=f"{args.log_dir}/{now}.log", encoding="utf-8", level=logging.DEBUG
    )


if __name__ == "__main__":
    main()
