# Read in Whisper
import argparse
import os
import re
import string
import unicodedata

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset
from jiwer import wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from num2words import num2words
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.pt_utils import KeyDataset

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_kblab", type=str, default="whisper-medium")
argparser.add_argument("--model_openai", type=str, default="whisper-medium")
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
        str: The normalized text.
    """

    text = text.lower()

    # # Convert numbers to words
    text = re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang="sv"), text)

    text = text.translate(str.maketrans("", "", string.punctuation))
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

model_openai = AutoModelForSpeechSeq2Seq.from_pretrained(
    args.model_openai,
    cache_dir="cache",
    torch_dtype=torch_dtype,
    attn_implementation="flash_attention_2",
)
model.to("cuda")
model_openai.to("cuda")

model_openai.eval()
model.eval()

processor = AutoProcessor.from_pretrained(args.model_kblab, cache_dir="cache")
processor_openai = AutoProcessor.from_pretrained(args.model_openai, cache_dir="cache")

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

pipe_openai = pipeline(
    "automatic-speech-recognition",
    model=model_openai,
    tokenizer=processor_openai.tokenizer,
    feature_extractor=processor_openai.feature_extractor,
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

output_texts_openai = []
for out in pipe_openai(KeyDataset(fleurs, "audio")):
    output_texts_openai.append(out["text"])


wer_list = [
    wer(normalize_text(fleurs["transcription"][i]), normalize_text(output_texts[i]))
    for i in range(len(fleurs))
]
wer_list_openai = [
    wer(normalize_text(fleurs["transcription"][i]), normalize_text(output_texts_openai[i]))
    for i in range(len(fleurs))
]

bleu_list = [
    calculate_bleu(normalize_text(fleurs["transcription"][i]), normalize_text(output_texts[i]))
    for i in range(len(fleurs))
]
bleu_list_openai = [
    calculate_bleu(
        normalize_text(fleurs["transcription"][i]), normalize_text(output_texts_openai[i])
    )
    for i in range(len(fleurs))
]

# Calculate average WER
kblab_wer = sum(wer_list) / len(wer_list)
openai_wer = sum(wer_list_openai) / len(wer_list_openai)

# Calculate average BLEU
kblab_bleu = sum(bleu_list) / len(bleu_list)
openai_bleu = sum(bleu_list_openai) / len(bleu_list_openai)

print(
    (
        f"Average WER for KBLab model: {kblab_wer} \n"
        f"Average WER for OpenAI model: {openai_wer} \n"
        f"Average BLEU for KBLab model: {kblab_bleu} \n"
        f"Average BLEU for OpenAI model: {openai_bleu} \n"
    )
)


results_dict = {
    "WER": [kblab_wer, openai_wer],
    "BLEU": [kblab_bleu, openai_bleu],
}

df = pd.DataFrame(results_dict)
df["organization"] = ["KBLab", "OpenAI"]
df["model"] = [args.model_kblab, args.model_openai]
df["eval_dataset"] = "fleurs_" + args.split

os.makedirs(args.results_dir, exist_ok=True)
df[["model", "eval_dataset", "organization", "WER", "BLEU"]].to_csv(
    f"{args.results_dir}/results_{args.model_kblab}_{args.split}.csv",
    index=False,
)


# res = pipe("audio_mono.wav")
# res_openai = pipe_openai("audio_mono.wav")

# # res
# # res_openai
# res["text"][11500:13500]
# res_openai["text"][680:1500]

# res.keys()
# res["chunks"][0:50]


## Combine results from different models
# df_small = pd.read_csv(f"results/results_whisper-small_{args.split}.csv")
# df_medium = pd.read_csv(f"results/results_whisper-medium_{args.split}.csv")
# df_large = pd.read_csv(f"results/results_whisper-large_{args.split}.csv")

# df = pd.concat([df_small, df_medium, df_large])
# df["parameters"] = ["244M", "244M", "769M", "769M", "1.5B"]
# df = df[["model", "parameters", "eval_dataset", "organization", "WER", "BLEU"]]
# df.to_csv(f"results/results_whisper_all_{args.split}.csv", index=False)

# # Print dataframe without index
# print(df.to_string(index=False))
