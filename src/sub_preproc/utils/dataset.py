import json
import os
import re
import csv
import subprocess
import tempfile

import soundfile as sf
import torch
from torch.utils.data import Dataset

from sub_preproc.utils.text import clean_subtitle
from sub_preproc.utils.make_chunks import n_non_silent_chunks


class AudioDataset(Dataset):
    """
    Takes multiple spectograms and returns one spectogram at a time.
    AudioFileChunker returns AudioDataset objects like this one so we can load
    the spectograms batch by batch with a dataloader.
    sub_dict is also included as an attribute so the main process doesn't have
    to read the json file again.
    """

    def __init__(self, spectograms, sub_dict):
        self.spectograms = spectograms
        self.sub_dict = sub_dict

    def __len__(self):
        return len(self.spectograms)

    def __getitem__(self, idx):
        return self.spectograms[idx]


class AudioFileChunkerDataset(Dataset):
    """
    Pytorch Dataset that converts audio file to wav, chunks
    audio file according to start/end times for observations
    specified in json file, and preprocesses data to spectograms.

    Args:
        json_paths (list): List of paths to json files
        model_name (str): Model name to use for the processor

    Returns:
        out_dict (dict): Dictionary with the following keys:
            "dataset": AudioDataset, or None if error reading audio file
            "metadata": Metadata from the json file
            "audio_path": Path to the audio file
            "json_path": Path to the json file
            "is_transcribed": Whether the audio has been transcribed
            "is_langdetected": Whether the audio has been language detected
    """

    def __init__(
        self, json_paths, model_name, processor, logger, chunks_or_subs="chunks", my_filter=None
    ):
        self.json_paths = json_paths
        self.model_name = model_name
        self.processor = processor
        self.chunks_or_subs = chunks_or_subs
        self.logger = logger
        if my_filter is None:
            self.my_filter = lambda x: True
        else:
            self.my_filter = my_filter
        # if "whisper" in model_name:
        #     self.processor = WhisperProcessor.from_pretrained(model_name)
        # elif "wav2vec2" in model_name:
        #     self.processor = AutoProcessor.from_pretrained(model_name)

    def __len__(self):
        return len(self.json_paths)

    def check_if_transcribed(self, sub_dict):
        """
        We include information about whether transcription and langdetect has already been
        performed (using the same model). Useful for skipping already transcribed files.
        """
        if "transcription" in sub_dict[self.chunks_or_subs][0]:
            models = [t["model"] for t in sub_dict[self.chunks_or_subs][0]["transcription"]]
            is_transcribed = (
                True if len(sub_dict[self.chunks_or_subs][0]["transcription"]) > 0 else False
            )
            is_transcribed_same_model = self.model_name in models
            is_langdetected = any(
                [
                    ("language" in transcription)
                    for transcription in sub_dict[self.chunks_or_subs][0]["transcription"]
                ]
            )
        else:
            is_transcribed = False
            is_transcribed_same_model = False
            is_langdetected = False

        return is_transcribed, is_transcribed_same_model, is_langdetected

    def ms_to_frames(self, ms, sr=16000):
        return int(ms / 1000 * sr)

    def read_audio(self, audio_path):
        if audio_path.endswith(".wav"):
            return sf.read(audio_path)
        with tempfile.TemporaryDirectory() as tmpdirname:
            try:
                convert_audio_to_wav(audio_path, os.path.join(tmpdirname, "tmp.wav"))
                audio, sr = sf.read(os.path.join(tmpdirname, "tmp.wav"))
            except Exception as e:
                self.logger.info(f"Error reading audio file {audio_path}. {e}")
                raise Exception(e)
                # print(f"Error reading audio file {audio_path}. {e}")
                # os.makedirs("logs", exist_ok=True)
                # with open("logs/error_audio_files.txt", "a") as f:
                #     f.write(f"{audio_path}\n")
                # return None
        return audio, sr

    def json_chunks(self, sub_dict):
        for chunk in filter(lambda x: self.my_filter(x), sub_dict[self.chunks_or_subs]):
            yield chunk["start"], chunk["end"]

    def audio_chunker(self, audio_path, sub_dict, sr=16000):
        audio, sr = self.read_audio(audio_path)

        for start, end in self.json_chunks(sub_dict):
            start_frame = self.ms_to_frames(start, sr)
            end_frame = self.ms_to_frames(end, sr)
            yield audio[start_frame:end_frame]

    def __getitem__(self, idx):
        json_path = self.json_paths[idx]
        if type(json_path) == list and len(json_path) == 2:
            json_path, audio_path = json_path
        elif len(json_path.split()) == 2:
            json_path, audio_path = json_path.split()
        else:
            audio_path = None

        with open(json_path) as f:
            try:
                sub_dict = json.load(f)
                if len(list(filter(lambda x: self.my_filter(x), sub_dict["chunks"]))) == 0:
                    # self.logger.info(f"Nothing do to for {json_path}")
                    out_dict = {
                        "dataset": None,
                        "metadata": None,
                        "audio_path": audio_path,
                        "json_path": json_path,
                        "is_transcribed": None,
                        "is_transcribed_same_model": None,
                        "is_langdetected": None,
                    }
                    return out_dict

                if "audio_path" in sub_dict["metadata"]:
                    audio_path = sub_dict["metadata"]["audio_path"]
                if n_non_silent_chunks(sub_dict) == 0:
                    out_dict = {
                        "dataset": None,
                        "metadata": sub_dict["metadata"],
                        "audio_path": audio_path,
                        "json_path": json_path,
                        "is_transcribed": None,
                        "is_transcribed_same_model": None,
                        "is_langdetected": None,
                    }
                    return out_dict
            except json.JSONDecodeError:
                self.logger.info(f"failed reading json-file {json_path}")
                raise Exception(e)
                out_dict = {
                    "dataset": None,
                    "metadata": None,
                    "audio_path": audio_path,
                    "json_path": json_path,
                    "is_transcribed": None,
                    "is_transcribed_same_model": None,
                    "is_langdetected": None,
                }
                return out_dict

        if not audio_path:
            self.logger.info(f"no audio path for  {json_path}")
            out_dict = {
                "dataset": None,
                "metadata": None,
                "audio_path": audio_path,
                "json_path": json_path,
                "is_transcribed": None,
                "is_transcribed_same_model": None,
                "is_langdetected": None,
            }
            return out_dict

        spectograms = []
        for audio_chunk in self.audio_chunker(audio_path, sub_dict):
            spectograms.append(
                self.processor(
                    audio_chunk, sampling_rate=16000, return_tensors="pt"
                ).input_features
                if "whisper" in self.model_name
                else self.processor(
                    audio_chunk, sampling_rate=16000, return_tensors="pt"
                ).input_values
            )

        if "whisper" in self.model_name:
            # Wav2vec2 processor doesn't pad up to 30s by default
            try:
                spectograms = torch.cat(spectograms, dim=0)
            except Exception as e:
                self.logger.info(
                    f"failed concatenating the spectograms for {json_path} with exception {e}"
                )
                # raise Exception(e)
                out_dict = {
                    "dataset": None,
                    "metadata": sub_dict["metadata"],
                    "audio_path": audio_path,
                    "json_path": json_path,
                    "is_transcribed": None,
                    "is_transcribed_same_model": None,
                    "is_langdetected": None,
                }
                return out_dict

        mel_dataset = AudioDataset(spectograms, sub_dict)

        is_transcribed, is_transcribed_same_model, is_langdetected = self.check_if_transcribed(
            sub_dict
        )

        out_dict = {
            "dataset": mel_dataset,
            "metadata": sub_dict["metadata"],
            "audio_path": audio_path,
            "json_path": json_path,
            "is_transcribed": is_transcribed,
            "is_transcribed_same_model": is_transcribed_same_model,
            "is_langdetected": is_langdetected,
        }

        return out_dict


class RawAudioFileChunkerDataset(Dataset):
    """
    Pytorch Dataset that converts audio file to wav, chunks
    audio file according to start/end times for observations
    specified in json file.

    Args:
        audio_paths (list): List of paths to audio files
        json_paths (list): List of paths to json files
        out_dir (str): Directory to save audio chunks

    Returns:
        out_dict (dict): Dictionary with the following keys:
            "dataset": AudioDataset, or None if error reading audio file
            "metadata": Metadata from the json file
            "audio_path": Path to the audio file
            "json_path": Path to the json file
    """

    def __init__(self, audio_paths, json_paths, out_dir):
        self.audio_paths = audio_paths
        self.json_paths = json_paths
        self.out_dir = out_dir

    def __len__(self):
        return len(self.audio_paths)

    def ms_to_frames(self, ms, sr=16000):
        return int(ms / 1000 * sr)

    def read_audio(self, audio_path):
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

    def audio_chunker_to_file(self, audio_path, sub_dict, sr=16000):
        audio, sr = self.read_audio(audio_path)
        # Extract filename without extension
        basename = os.path.basename(audio_path).split(".")[0]
        # filters = ["stage1_whisper", "stage2_whisper", "stage2_whisper_timestamps", "stage1_wav2vec"]
        os.makedirs(self.out_dir + "/stage1_wav2vec", exist_ok=True)
        json_info = []
        for i, chunk in enumerate(sub_dict["chunks"]):
            start_frame = self.ms_to_frames(chunk["start"], sr)
            end_frame = self.ms_to_frames(chunk["end"], sr)

            # Save audio chunk
            chunk_audio = audio[start_frame:end_frame]
            if "filters" in chunk:
                if chunk["filters"]["stage1_wav2vec"] == True:
                    duration = chunk["duration"]
                    chunk_audio_path = os.path.join(
                        self.out_dir, "stage1_wav2vec", f"{basename}_{i}_duration_{duration}.wav"
                    )
                    with sf.SoundFile(chunk_audio_path, "w", sr, channels=1) as f:
                        f.write(chunk_audio)
                    # Save ground truth text
                    text = chunk["text"]
                    text_path = os.path.join(self.out_dir, "stage1_wav2vec", f"{basename}_{i}_duration_{duration}.txt")
                    with open(text_path, "w") as f:
                        text = clean_subtitle(text)
                        f.write(text)
        
        return None

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        json_path = self.json_paths[idx]
        
        with open(json_path) as f:
            sub_dict = json.load(f) 

        self.audio_chunker_to_file(audio_path, sub_dict)

        return None


def convert_audio_to_wav(input_file, output_file):
    # fmt: off
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ar', '16000',  # Set the audio sample rate to 16kHz
        '-ac', '1',      # Set the number of audio channels to 1 (mono)
        '-c:a', 'pcm_s16le',
        '-loglevel', 'warning',
        '-hide_banner',
        '-nostats',
        '-nostdin',
        output_file
    ]
    # fmt: on
    subprocess.run(command)


def custom_collate_fn(batch: dict) -> list:
    """
    Collate function to allow dictionaries with Datasets in the batch.
    """
    # Remove None values
    batch = [b for b in batch if b is not None]

    # Return None if batch is empty
    if len(batch) == 0:
        return None

    # Return the batch
    return batch


def wav2vec_collate_fn(batch):
    """
    We need to pad the input_values to the longest sequence,
    since wav2vec2 doesn't do this by default.
    """
    # Remove None values
    batch = [b[0] for b in batch if b is not None]

    # Pad the input_values to the longest sequence
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

    return batch


def make_transcription_chunks(
    transcriptions,
    model_name,
):
    transcription_chunks = []

    for transcript in transcriptions:
        transcription_dict = {
            "text": transcript.encode("utf-8").decode("utf-8"),
            "model": model_name,
        }
        transcription_chunks.append(transcription_dict)

    return transcription_chunks


def make_transcription_chunks_w2v(
    transcriptions,
    word_timestamps,
    model_name,
):
    transcription_chunks = []

    for i, transcript in enumerate(transcriptions):
        transcription_dict = {
            "text": transcript,
            "word_timestamps": word_timestamps[i],
            "model": model_name,
        }
        transcription_chunks.append(transcription_dict)

    return transcription_chunks
