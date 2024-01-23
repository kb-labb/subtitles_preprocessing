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
import torch
import torch.multiprocessing as mp

# from faster_whisper import WhisperModel
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

import sub_preproc.utils.utils as utils
from sub_preproc.dedup import dup_marker_single
from sub_preproc.detect_language import detect_language
from sub_preproc.utils.make_chunks import make_chunks
from sub_preproc.utils.utils import SILENCE

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


def get_sv_sound_chunks(
    chunks, audio, sample_rate, model, processor
) -> Iterable[Tuple[Dict[str, Any], np.ndarray]]:
    for chunk in chunks:
        start = chunk["start"]
        end = chunk["end"]
        chunk_audio = audio[start * sample_rate // 1_000 : end * sample_rate // 1_000]
        if chunk["text"] == "":
            yield chunk, chunk_audio
        elif model is None:
            yield chunk, chunk_audio
        else:
            # _, info = model.transcribe(chunk_audio, vad_filter=True, beam_size=5)
            # if info.language == "sv" and info.language_probability > 0.5:
            #     yield chunk, chunk_audio
            inputs = (
                processor.feature_extractor(chunk_audio, return_tensors="pt", sampling_rate=16_000)
                .input_features.to("cuda:0")
                .to(torch.float16)
            )
            language_probs = detect_language(model, processor.tokenizer, inputs)[0]
            l, p = max(language_probs.items(), key=lambda x: x[1])
            if l == "sv" and p > 0.5:
                yield chunk, chunk_audio


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


def do_stuff(fn, args, channel_plus, saved_filenames, metadata, seen, model, processor):
    to_log = []
    log_points = [
        "check_subs",
        "extract_subs",
        "read_srt",
        "fuse",
        "dups",
        "livesubs",
        "to_dict",
        "chunks",
        "write_json",
        "audio?",
        "extract_audio",
        "read_audio",
    ]

    def log_time(log_point, prev):
        to_log.append(f"{log_point:<20s}{time.time() - prev:.4f}")
        return time.time()

    start = time.time()
    prev = start
    program_id = fn.split("/")[-1].split(".")[0]
    channel, subchannel, year, month, day, from_time, to_time = utils.decode_program_id(program_id)
    savedir = os.path.join(args.out_data, channel, subchannel, year, month, day, program_id)
    assert channel_plus == "/".join([channel, subchannel])

    swe_sub_id = check_for_sv_subs(fn)
    prev = log_time("check_subs", prev)

    if swe_sub_id:
        os.makedirs(savedir, exist_ok=True)
        extract_subs(fn, swe_sub_id, savedir)
        prev = log_time("extract_subs", prev)
        subs = pysrt.open(os.path.join(savedir, "file.srt"))
        prev = log_time("read_srt", prev)
        fused = utils.fuse_subtitles(subs)
        prev = log_time("fuse", prev)
        dup_marked, seen = dup_marker_single(fused, seen)
        prev = log_time("dups", prev)
        live_subbed = utils.mark_live_subs(dup_marked)
        prev = log_time("livesubs", prev)
        subs_dict = utils.subrip_to_dict(
            live_subbed, channel, subchannel, year, month, day, from_time, to_time
        )
        prev = log_time("to_dict", prev)
        subs_chunks_dict = make_chunks(subs_dict)
        prev = log_time("chunks", prev)
        with open(os.path.join(savedir, "file.json"), "w") as fout:
            json.dump(subs_chunks_dict, fout, indent=4)
        prev = log_time("write_json", prev)
        saved_filenames.append(os.path.join(savedir, "file.json"))
        if not args.skip_audio:
            # audio
            n_chunks = 0
            if precheck_for_chunks(subs_chunks_dict):
                prev = log_time("audio?", prev)
                os.makedirs(os.path.join(savedir, "chunks"), exist_ok=True)
                extract_sound(input_file=fn, output_path=savedir, sound_format=args.sound_format)
                prev = log_time("extract_audio", prev)
                audio = read_audio(
                    os.path.join(savedir, f"file.{args.sound_format}"),
                    target_sample_rate=args.sample_rate,
                )

                p = pathlib.Path(os.path.join(savedir, f"file.{args.sound_format}"))
                p.unlink()

                prev = log_time("read_audio", prev)
                for i, (chunk, chunk_audio) in enumerate(
                    get_sv_sound_chunks(
                        subs_chunks_dict["chunks"], audio, args.sample_rate, model, processor
                    )
                ):
                    n_chunks += 1
                    with sf.SoundFile(
                        os.path.join(savedir, "chunks", f"chunk_{i}.{args.chunk_sound_format}"),
                        "w",
                        args.sample_rate,
                        channels=1,
                    ) as fout:
                        fout.write(chunk_audio)
                    with open(os.path.join(savedir, "chunks", f"chunk_{i}.txt"), "w") as fout:
                        print(chunk["text_whisper"], file=fout)
                    metadata.append(
                        (
                            os.path.join(
                                savedir, "chunks", f"chunk_{i}.{args.chunk_sound_format}"
                            ),
                            chunk["text"],
                            chunk["text_whisper"],
                        )
                    )
                prev = log_time(n_chunks, prev)
    prev = log_time("total", start)

    return to_log


def main():
    args = get_args()
    print(args)

    now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    logging.basicConfig(
        filename=f"{args.log_dir}/{now}.log", encoding="utf-8", level=logging.DEBUG
    )

    # model_size = "large-v3"
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")
    # model = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "distil-whisper/distil-large-v2"

    if args.skip_audio:
        model = None
        processor = None
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

    metadata: List[Tuple[str, str, str]] = [("file_name", "text", "text_whisper")]

    if args.seen is None:
        seen = set()
    else:
        seen = pickle.load(args.seen)

    saved_filenames = []

    manager = mp.Manager()
    seen = manager.dict()

    for channel_plus in tqdm(CHANNELS):
        filenames = glob.iglob(f"{args.in_data}/{channel_plus}/**/*mp4", recursive=True)
        my_fun = partial(
            do_stuff,
            args=args,
            channel_plus=channel_plus,
            saved_filenames=saved_filenames,
            metadata=metadata,
            seen=seen,
            model=model,
            processor=processor,
        )
        # if True:
        with mp.get_context("spawn").Pool(processes=args.processes) as pool:
            xs = pool.imap(my_fun, tqdm(filenames))
            # xs = map(my_fun, tqdm(filenames))
            for to_log in xs:
                for x in to_log:
                    logging.debug(x)
                logging.debug("")
        # for fn in tqdm(filenames):
        #     do_stuff(fn, args, channel_plus, saved_filenames, metadata, seen)
    with open(os.path.join(args.out_data, "sub_and_chunk_dicts.txt"), "w") as fout:
        for fn in saved_filenames:
            print(fn, file=fout)
    with open(os.path.join(args.out_data, "seen_subs.pickle"), "wb") as fout:
        pickle.dump(seen, fout)
    with open(os.path.join(args.out_data, "metadata.csv"), "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerows(metadata)


if __name__ == "__main__":
    main()
