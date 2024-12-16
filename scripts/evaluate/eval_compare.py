# Read in Whisper
import argparse
import difflib
import os
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import load_dataset
from IPython.display import HTML, display
from jiwer import wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from num2words import num2words
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.pt_utils import KeyDataset

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_kblab", type=str, default="whisper-medium")
argparser.add_argument("--checkpoint", type=str, default="135000")
argparser.add_argument("--split", type=str, default="validation")
argparser.add_argument("--results_dir", type=str, default="results")
args = argparser.parse_args()


def normalize_text(text):
    """
    Normalize speech text transcript by removing punctuation, converting numbers to words,
    replacing hyphens joining words with whitespace, and lowercasing the text.

    Args:
        text (str): The text to normalize.
    Returns:
        str: The norm text.
    """

    text = text.lower()

    # # Convert numbers to words
    # text = re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang="sv"), text)
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)
    # Remove multiple spaces and replace with single space
    text = re.sub(r"\s+", " ", text)
    ## Remove whitespace between numbers
    # text = re.sub(r"(?<=\d) (?=\d)", "", text)
    # Strip leading and trailing whitespace
    text = text.strip()

    return text


def calculate_bleu(text1, text2):
    """
    Calculate BLEU score between two texts.
    """

    if text1 is None or text2 is None:
        return None
    else:
        chencherry = SmoothingFunction()
        return sentence_bleu(
            references=[text1.split()],
            hypothesis=text2.split(),
            smoothing_function=chencherry.method4,
        )


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    args.model_kblab,
    cache_dir="cache",
    torch_dtype=torch_dtype,
    attn_implementation="flash_attention_2",
)

model.to("cuda")
model.eval()

processor = AutoProcessor.from_pretrained(args.model_kblab, cache_dir="cache")

gen_kwargs = {
    "task": "transcribe",
    "language": "<|sv|>",
}

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    batch_size=32,
    return_timestamps=False,
    chunk_length_s=30,
    generate_kwargs=gen_kwargs,
)

fleurs = load_dataset("google/fleurs", "sv_se", split=args.split, cache_dir="cache")

output_texts = []
for out in pipe(KeyDataset(fleurs, "audio")):
    output_texts.append(out["text"])


data = {
    "gold_standard": fleurs["raw_transcription"],
    "gold_standard_norm": [normalize_text(ex["transcription"]) for ex in fleurs],
    "whisper": output_texts,
    "whisper_norm": [normalize_text(t) for t in output_texts],
}

df = pd.DataFrame(data)
df["wer"] = df.apply(lambda row: wer(row["gold_standard_norm"], row["whisper_norm"]), axis=1)

df["checkpoint"] = args.checkpoint
df[["gold_standard", "whisper", "wer", "checkpoint"]].to_csv(
    os.path.join(
        args.results_dir, "qualitative", f"asr-output_{args.model_kblab}_{args.checkpoint}.csv"
    ),
    index=False,
)


####
import glob

csv_files = glob.glob(os.path.join(args.results_dir, "qualitative", "asr-output_*.csv"))
# Read in all CSV files
dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Combine `index` and `checkpoint` to ensure uniqueness for pivoting
df["index"] = df.groupby("checkpoint").cumcount()

# Pivot using the composite index
df_wide = df.melt(id_vars=["index", "checkpoint"], var_name="variable", value_name="value")
df_wide = df_wide.pivot(index="index", columns=["checkpoint", "variable"], values="value")
df_wide.columns = ["{}_{}".format(var, chkpt) for chkpt, var in df_wide.columns]
df_wide = df_wide.reset_index()


# Function to highlight differences
def highlight_diffs(ref, hyp):
    diff = difflib.ndiff(ref.split(), hyp.split())
    diff_list = []
    for word in diff:
        if word.startswith("+"):
            diff_list.append(f"<span style='color:green'>[{word[2:]}]</span>")
        elif word.startswith("-"):
            diff_list.append(f"<span style='color:red'>({word[2:]})</span>")
        else:
            diff_list.append(word[2:])
    return " ".join(diff_list)


df_wide["gold_standard_norm"] = df_wide["gold_standard_50250"].apply(normalize_text)
df_wide["whisper_50250_norm"] = df_wide["whisper_50250"].apply(normalize_text)
df_wide["whisper_135000_norm"] = df_wide["whisper_135000"].apply(normalize_text)

df_wide["diff_whispers_50250"] = df_wide.apply(
    lambda row: highlight_diffs(row["whisper_50250_norm"], row["whisper_135000_norm"]),
    axis=1,
)

df_wide["diff_whisper135000_gold"] = df_wide.apply(
    lambda row: highlight_diffs(row["whisper_135000_norm"], row["gold_standard_norm"]),
    axis=1,
)

df_wide["diff_whisper50250_gold"] = df_wide.apply(
    lambda row: highlight_diffs(row["whisper_50250_norm"], row["gold_standard_norm"]),
    axis=1,
)
df_wide = df_wide.rename(
    columns={
        "gold_standard_50250": "gold_standard",
        "diff_whispers_50250": "diff_whispers",
    },
)


# Display as a styled DataFrame
def display_df(df):
    styled_df = df[
        [
            "gold_standard",
            "whisper_50250",
            "whisper_135000",
            "gold_standard_norm",
            "whisper_50250_norm",
            "whisper_135000_norm",
            "wer_50250",
            "wer_135000",
            "diff_whisper50250_gold",
            "diff_whisper135000_gold",
            "diff_whispers",
            "display_audio",
        ]
    ]
    return styled_df.style.map(lambda x: "color: red;" if isinstance(x, float) and x > 0.2 else "")


# Write the audio files to disk temporarily to be able to display them
output_dir = os.path.join("audio_temp", args.split)
filepaths = []
os.makedirs(output_dir, exist_ok=True)
for i, ex in tqdm(enumerate(fleurs)):
    filepath = os.path.join(output_dir, os.path.basename(ex["audio"]["path"]))
    # Make it to absolute path
    filepath = os.path.abspath(filepath)
    sf.write(filepath, ex["audio"]["array"], 16000)
    filepaths.append(filepath)

# Display the audio files
df_wide["display_audio"] = filepaths
df_wide["display_audio"] = df_wide["display_audio"].apply(
    lambda x: f'<audio controls><source src="{x}" type="audio/wav"></audio>'
)

display_df(df_wide).to_html(
    os.path.join(args.results_dir, "qualitative", "asr_output_comparison.html")
)
