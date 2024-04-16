import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from sub_preproc.utils.dataset import convert_audio_to_wav


def read_audio(sound_file: str, target_sample_rate) -> np.ndarray:
    if os.path.splitext(sound_file)[1] in ["wav", "flac", "mp3"]:
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
    else:
        audio, sr = convert_and_read_audio(sound_file)
        return audio


def convert_and_read_audio(audio_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            convert_audio_to_wav(audio_path, os.path.join(tmpdirname, "tmp.wav"))
            audio, sr = sf.read(os.path.join(tmpdirname, "tmp.wav"))
        except Exception as e:
            print(f"Error reading audio file {audio_path}. {e}")
            os.makedirs("logs", exist_ok=True)
            with open("logs/error_audio_files.txt", "a") as f:
                f.write(f"{audio_path}\n")
            return None
    return audio, sr


def get_audio_chunk(chunk, audio, sample_rate) -> Optional[Tuple[Dict[str, Any], np.ndarray]]:
    start = chunk["start"]
    end = chunk["end"]
    chunk_audio = audio[start * sample_rate // 1_000 : end * sample_rate // 1_000]
    # Might need to run chunk["text"].strip() unless it has been done before in the pipeline
    if chunk["text"] == "":
        return None
    return chunk, chunk_audio
