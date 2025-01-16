# Read in Whisper
import argparse
import os
import re
import string
import unicodedata
import json
import shutil
import tempfile
import soundfile as sf
import pandas as pd
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from jiwer import wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from num2words import num2words
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.pt_utils import KeyDataset

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_path", type=str, default="/leonardo_work/EUHPC_A01_006/experiments_whisper/outputs/2024-12-06_medium-stage1")
argparser.add_argument("--checkpoint", type=str, default="750")
argparser.add_argument("--split", type=str, default="validation")
argparser.add_argument("--results_dir", type=str, default="/leonardo_work/EUHPC_A01_006/kb-leonardo/agnes/subtitles_preprocessing/scripts/evaluate/results")
argparser.add_argument("--temp_dir", type=str, default = '/leonardo_work/EUHPC_A01_006/kb-leonardo/agnes/model_files/')
args = argparser.parse_args()

def normalize_text(text):
   

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
    

    if text1 is None or text2 is None:
        return None
    else:
        chencherry = SmoothingFunction()
        return sentence_bleu(
            references=[text1.split()],
            hypothesis=text2.split(),
            smoothing_function=chencherry.method4,
        )

#####################################
# Copy model files to new directory #
#####################################
files_to_copy = ["added_tokens.json", "config.json", "merges.txt", "normalizer.json", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json", "vocab.json", "generation_config.json"]
model_name = args.model_path.split('/')[-1]+ '_checkpoint_' + args.checkpoint
temp_model_dir = args.temp_dir + model_name 
os.makedirs(temp_model_dir, exist_ok=True)

#temp_dir = tempfile.TemporarydDirectory()

for file in files_to_copy:
    shutil.copy(args.model_path+'/'+file, temp_model_dir)
shutil.copy(args.model_path+'/checkpoint-'+args.checkpoint+'/model.safetensors', temp_model_dir)


##############
# Load model #
##############

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    temp_model_dir,
    cache_dir="cache",
    torch_dtype=torch_dtype,
    attn_implementation="sdpa",
)

model.to("cuda")
model.eval()

processor = AutoProcessor.from_pretrained(temp_model_dir, cache_dir="cache")

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

###################
# Evaluate fleurs #
###################

#fleurs = load_from_disk("/leonardo_work/EUHPC_A01_006/data/google___fleurs", "sv_se")
fleurs = load_dataset("google/fleurs", "sv_se", split="validation", cache_dir="/leonardo_work/EUHPC_A01_006/kb-leonardo/agnes/subtitles_preprocessing/scripts/evaluate/cache")

output_texts = []
for out in tqdm(pipe(KeyDataset(fleurs, "audio"))):
    output_texts.append(out["text"])

data_fleurs = {
    "gold_standard": fleurs["raw_transcription"],
    "gold_standard_norm": [normalize_text(ex["transcription"]) for ex in fleurs],
    "whisper": output_texts,
    "whisper_norm": [normalize_text(t) for t in output_texts],
}

df_fleurs= pd.DataFrame(data_fleurs)
df_fleurs["wer"] = df_fleurs.apply(lambda row: wer(row["gold_standard_norm"], row["whisper_norm"]), axis=1)
df_fleurs["bleu"] = df_fleurs.apply(lambda row: calculate_bleu(row["gold_standard_norm"], row["whisper_norm"]), axis=1)

####################
# Evaluate commonv #
####################

commonv = load_dataset("/leonardo_work/EUHPC_A01_006/kb-leonardo/agnes/subtitles_preprocessing/scripts/evaluate/cache/mozilla-foundation___common_voice_17_0", split="test", cache_dir="cache")

output_texts = []
for out in tqdm(pipe(KeyDataset(commonv, "audio"))):
    output_texts.append(out["text"])

data_commonv = {
    "gold_standard": commonv["sentence"],
    "gold_standard_norm": [normalize_text(ex["sentence"]) for ex in commonv],
    "whisper": output_texts,
    "whisper_norm": [normalize_text(t) for t in output_texts],
}

df_commonv= pd.DataFrame(data_commonv)
df_commonv["wer"] = df_commonv.apply(lambda row: wer(row["gold_standard_norm"], row["whisper_norm"]), axis=1)
df_commonv["bleu"] = df_commonv.apply(lambda row: calculate_bleu(row["gold_standard_norm"], row["whisper_norm"]), axis=1)

################
# Evaluate nst #
################
'''list_files = ["/leonardo_work/EUHPC_A01_006/data/big_parquets/nst_test +"/"+ f for f in os.listdir("/leonardo_work/EUHPC_A01_006/data/big_parquets/nst_test") if os.path.isfile(os.path.join("/leonardo_work/EUHPC_A01_006/data/big_parquets/nst_test", f))]
nst = load_dataset("parquet",data_files={'train': list_files},split="train",cache_dir="cache")

nst = nst.with_format("np", columns=["audio_tensor"], output_all_columns=True)

output_texts = []
for out in tqdm(pipe(KeyDataset(nst,"audio_tensor"))):
    output_texts.append(out["text"])

data_nst = {
    "gold_standard": nst["sentence"],
    "gold_standard_norm": [normalize_text(ex["sentence"]) for ex in nst],
    "whisper": output_texts,
    "whisper_norm": [normalize_text(t) for t in output_texts],
}

df_nst= pd.DataFrame(data_commonv)
df_nst["wer"] = df_nst.apply(lambda row: wer(row["gold_standard_norm"], row["whisper_norm"]), axis=1)
df_nst["bleu"] = df_nst.apply(lambda row: calculate_bleu(row["gold_standard_norm"], row["whisper_norm"]), axis=1)
'''
##################
# Export results #
##################

results_dict = {
    "model": model_name,
    "checkpoint": args.checkpoint,
    "fleurs_validation": df_fleurs.to_dict(),
    "commonv_test": df_commonv.to_dict()
}


os.makedirs(args.results_dir, exist_ok=True)
with open(f"{args.results_dir}/{model_name}.json", "w") as outfile: 
    json.dump(results_dict, outfile)

shutil.rmtree(temp_model_dir)
#os.removedirs(temp_dir)