import argparse
import gc
import logging
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import rixvox.text as rixvox_text
import sub_preproc.utils.text as sub_preproc_text
from rapidfuzz.distance.Levenshtein import (
    normalized_distance as levenshtein_dist_normalized,
)
from rixvox.metrics import (
    calculate_bleu,
    calculate_wer,
    first_word_fuzzy_score,
    last_word_fuzzy_score,
)
from tokenizers.models import BPE
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer

logging.basicConfig(
    filename="logs/preprocess_and_filter.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--data_dir",
    type=str,
    # required=True,
    default="/leonardo_work/EUHPC_A01_006/data/big_parquets/svt",
    help="Directory where all the parquets are stored.",
)
argparser.add_argument(
    "--output_dir",
    type=str,
    # required=True,
    default="/leonardo_scratch/fast/EUHPC_A01_006",
    help="Directory where the processed data will be saved.",
)
argparser.add_argument(
    "--parquet_filename",
    type=str,
    # required=True,
    default="svt1_0135.parquet",
    help="Filename of the parquet file to process.",
)
argparser.add_argument(
    "--dataset",
    type=str,
    choices=["svt", "rixvox", "smdb", "youtube", "isof", "sls"],
    default="svt",
    help="Dataset source to preprocess.",
)
argparser.add_argument(
    "--stage",
    type=str,
    choices=["stage1", "stage2", "stage_wav2vec2"],
    default="stage1",
    help="""Which quality filters to use for creating different stages
    of pretraining corpora (dataset annealing).""",
)

argparser.add_argument(
    "--model_name_or_path",
    type=str,
    default="openai/whisper-small",
    help="""Tiny, base, small and medium have the same preprocessing.
    Large uses different settings for the FeatureExtractor, so need 
    to run the preprocessing separately for large models.""",
)
argparser.add_argument(
    "--cache_dir",
    type=str,
    default="cache",
    help="Directory to cache the downloaded tokenizer/feature extractor/model weights.",
)
argparser.add_argument(
    "--language",
    type=str,
    default="sv",
    help="Language prefix token for the tokenizer.",
)
argparser.add_argument(
    "--task",
    type=str,
    default="transcribe",
    help="Task prefix token: either transcribe or translate.",
)

argparser.add_argument(
    "--sampling_rate",
    type=int,
    default=16000,
    help="Sampling rate of the audio files.",
)

# This should ideally be applied at training time in DataCollator, not at preprocessing time.
argparser.add_argument(
    "--bpe_dropout",
    type=float,
    default=0.0,
    help="""BPE dropout rate. Makes the tokenizer use differnt subwords to encode the same word.
    Good for regularization to prevent overfitting.""",
)
# SpecAugment is appled at training time, this only determines whether attention_mask is returned.
argparser.add_argument(
    "--apply_spec_augment",
    action="store_true",
    default=True,
    help="Apply SpecAugment to the input features.",
)
# Should probably be higher than 0.05. HF have misinterpreted the papers using it.
# https://discuss.huggingface.co/t/wav2vec2-config-why-is-mask-time-prob-0-05-and-not-0-5/25060
# See also: https://github.com/huggingface/transformers/pull/14525#issuecomment-980650156
argparser.add_argument(
    "--mask_time_prob",
    type=float,
    default=0.5,
    help="Probability of masking a time step.",
)
argparser.add_argument(
    "--mask_time_length",
    type=int,
    default=10,
    help="Number of time steps to mask.",
)
argparser.add_argument(
    "--mask_feature_prob",
    type=float,
    default=0.0,
    help="Probability of masking a feature.",
)
argparser.add_argument(
    "--mask_feature_length",
    type=int,
    default=10,
    help="Number of features to mask.",
)

argparser.add_argument(
    "--min_input_length",
    type=int,
    default=8000,
    help="Minimum length of input audio.",
)
argparser.add_argument(
    "--max_input_length",
    type=int,
    default=480000,
    help="Minimum length of input audio.",
)
argparser.add_argument(
    "--max_label_length",
    type=int,
    default=448,
    help="Maximum length of the tokenized labels.",
)
argparser.add_argument(
    "--max_previous_text_length",
    type=int,
    default=192,
    help="Maximum length of the tokenized prompt text before we truncate (on the left).",
)
argparser.add_argument(
    "--stats_dir",
    type=str,
    default="/leonardo_work/EUHPC_A01_006/data/big_parquets/stats",
    help="Directory to save the summary statistics.",
)
argparser.add_argument(
    "--overwrite",
    action="store_true",
    help="Whether to overwrite the existing parquet file.",
)
argparser.add_argument(
    "--stats_only",
    action="store_true",
    help="Only write the summary statistics and exit.",
)

args = argparser.parse_args()


def clean_text(text, svt=False):
    """
    Clean the ground truth text of unwanted characters and patterns.

    Args:
        text (str): The text to normalize.
        svt (bool): Whether the text is from SVT or not.
    Returns:
        str: The cleaned text.
    """

    # Replace abbreviations with their full form
    # text = expand_abbreviations(text)

    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove hyphens and double hyphens
    text = text.replace("- -", " ").replace("– –", " ")

    # Remove "/  /"  and everything between them
    text = re.sub(r"/.*?/", " ", text)

    # Remove everything between parentheses
    text = re.sub(r"\(.*?\)", "", text)

    # Remove everything between brackets
    text = re.sub(r"\[.*?\]", "", text)

    # Remove everything between asterisks *
    text = re.sub(r"\*.*?\*", "", text)

    # Remove "-" and "–" in the beginning of the text
    text = text.lstrip("-–").rstrip("-–")

    # Remove - and – from a string if the preceding character is >
    # (when training with timestamps like <|0.00|> in the text)
    text = re.sub(r"(?<=[>])[-–]", "", text)

    # Remove - – from a string if the following character ia <
    text = re.sub(r"[-–](?=[<])", "", text)

    # Remove ... from a string if the preceding character is >
    text = re.sub(r"(?<=[>])\.\.\.", "", text)

    # Remove hyphens and dashes in the beginning and in the end of a word, but leave hyphens in the middle of words
    text = re.sub(r"(?<=\s)[-–](?=\w)|(?<=\w)[-–](?=\s)", "", text)

    # Remove hyphens in examples such as -...med...
    text = re.sub(r"(?<=\s)[-–](?=\.\.\.)", "", text)

    # Use regex to remove '...' from the beginning of any text
    text = re.sub(r"^\.\.\.", "", text)

    # Remove ... ... and replace with single space
    text = re.sub(r"(.+)\.\.\. \.\.\.", r"\1 ", text)

    if svt:
        # fmt: off
        capitalized_words = [
            "TJUT", "GRÅT", "SKRATT", "SKRIK", "LJUD", "MUSIK","HÖRS","VRÅL","SNARK","FRÄSA","BRUMMA",
            "KNACK","RING","RÖST","PRAT","ÅSKA","VIND","REGN","HUND","MOTOR","BIL","STORM","DROP",
            "STEG","GLAS","KROCK","KLOCKA","JUBLAR","BUAR","APPLÅD","KLICK","SLAG","DUNK","SORL",
            "MUMMEL","GNYR","VISSLA","TJUTER","GAP","SUCK","PUST","HARKLAR","FJÄT","KNÄCK","SKRAMMEL",
            "PRASSLA","FLÄMT","FLADDRA","VÅG","BRIS","SPRICKA","SPRÄNGNING","PANG","FÖNSTER","DÖRR",
            "RULLAR","SLÄPAR","TUTA","LARM","SKALL","VIBRATION","TAPPAR","DUNKA","MULLER","FÖNSTERKROSS",
            "KRAFT","KNÄCKTES","RASSLA",
        ]
        # fmt: on

        # Regex  to match any of the words as a stem
        pattern = r"\b(?:" + "|".join(word + r"\w*" for word in capitalized_words) + r")\b"

        # Find any segment of capitalized words
        capitalized_segment_pattern = r"(\b[A-ZÅÄÖ]+\b(?:\s+\b[A-ZÅÄÖ]+\b)*)"

        # Search for capitalized segments that contain any of the target word variations
        match = re.search(capitalized_segment_pattern, text)
        if match and re.search(pattern, match.group()):
            # Remove the matched capitalized segment
            text = re.sub(capitalized_segment_pattern, "", text)

    # Remove multiple spaces, newlines, and breaks, and replace with single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text


def cer_head(text: str, transcription: str, len_lookback: int = 10) -> float:
    """
    Calculate the CER between the head of the text and the transcription based on
    number of characters (len_lookback) to look forward.
    """

    head_text = text[:len_lookback]
    head_transcription = transcription[:len_lookback]

    cer_match_head = levenshtein_dist_normalized(head_text, head_transcription)
    return cer_match_head


def cer_tail(text: str, transcription: str, len_lookback: int = 10) -> float:
    """
    Calculate the CER between the tail of the text and the transcription based on
    number of characters (len_lookback) to look back.
    """

    tail_text = text[-len_lookback:]
    tail_transcription = transcription[-len_lookback:]

    cer_match_tail = levenshtein_dist_normalized(tail_text, tail_transcription)
    return cer_match_tail


def calculate_metrics(row, score_function: callable, normalize_text: callable):
    """
    Calculate the score (bleu, wer, cer) between the normalized text and the transcriptions.

    Args:
        row: row of the DataFrame
        score_function: Function that calculates the score between two texts
        normalize_text: Function that normalizes the text
    Returns:
        score_whisper, score_wav2vec: score between the normalized text and the whisper transcription
            and score between the normalized text and the wav2vec transcription
    """
    text_normalized = row["text_normalized"]
    whisper_normalized = normalize_text(row["whisper_transcription"])
    wav2vec_normalized = normalize_text(row["wav2vec_transcription"])

    score_whisper = score_function(text_normalized, whisper_normalized)
    score_wav2vec = score_function(text_normalized, wav2vec_normalized)
    return score_whisper, score_wav2vec


def tokenize_ground_truth(row, text_column, text_timestamps_column, tokenizer, truncation=False):
    """
    Args:
        row: row of the DataFrame
        text_column: column name of the text
        text_timestamps_column: column name that contains timestamps with text
        tokenizer: tokenizer
        truncation: whether to truncate the tokenized text to max_length of the model
    """
    text = row[text_column]
    text_timestamps = row[text_timestamps_column]

    tokenizer.set_prefix_tokens(predict_timestamps=False)
    labels = tokenizer(text, truncation=truncation).input_ids
    tokenizer.set_prefix_tokens(predict_timestamps=True)
    labels_timestamps = tokenizer(text_timestamps, truncation=truncation).input_ids

    # Length of the text (to filter out when label_length > max_length of model)
    labels_length = len(labels)
    labels_timestamps_length = len(labels_timestamps)

    return labels, labels_timestamps, labels_length, labels_timestamps_length


def tokenize_prompt(row, text_column, tokenizer, truncation=False, max_length=192):
    """
    Args:
        row: row of the DataFrame
        text_column: column name of the text
        tokenizer: tokenizer
        truncation: whether to truncate the tokenized text to max_length of the model
    """
    text = row[text_column]

    if text is None:
        return None, 0

    tokenizer.set_prefix_tokens(predict_timestamps=False)
    prompt_tokens = tokenizer(text, truncation=truncation, add_special_tokens=False).input_ids

    # Truncate and -1 for <|startofprev|>
    prompt_tokens_left_truncated = prompt_tokens[-(max_length - 1) :]
    prompt_with_start_token = [tokenizer.convert_tokens_to_ids("<|startofprev|>")]
    prompt_with_start_token.extend(prompt_tokens_left_truncated)

    previous_tokens_length = len(prompt_with_start_token)

    return prompt_with_start_token, previous_tokens_length


def extract_audio_features(
    row, feature_extractor, return_attention_mask, sampling_rate=args.sampling_rate
):
    model_input_name = feature_extractor.model_input_names[0]  # "input_features"
    inputs = feature_extractor(
        row["audio_tensor"],
        sampling_rate=sampling_rate,
        return_attention_mask=return_attention_mask,
    )

    # Get mel spectrogram features and save as input_features
    input_features = inputs.get(model_input_name)[0]
    # process audio length
    input_length = len(row["audio_tensor"])
    if return_attention_mask:
        attention_mask = inputs.get("attention_mask")[0]
        return input_features, attention_mask, input_length
    else:
        return input_features, None, input_length


def get_all_metrics(df):

    normalize_text_fun = (
        rixvox_text.normalize_text if args.dataset == "rixvox" else sub_preproc_text.normalize_text
    )

    df["text_normalized"] = df["text"].apply(normalize_text_fun)
    df["whisper_transcription_normalized"] = df["whisper_transcription"].apply(normalize_text_fun)
    df["wav2vec_transcription_normalized"] = df["wav2vec_transcription"].apply(normalize_text_fun)

    df[["bleu_whisper", "bleu_wav2vec2"]] = df.apply(
        calculate_metrics,
        args=(calculate_bleu, normalize_text_fun),
        result_type="expand",
        axis=1,
    )

    df[["wer_whisper", "wer_wav2vec2"]] = df.apply(
        calculate_metrics,
        args=(calculate_wer, normalize_text_fun),
        result_type="expand",
        axis=1,
    )

    df["whisper_first"] = df.apply(
        lambda x: first_word_fuzzy_score(
            x["text_normalized"], x["whisper_transcription_normalized"]
        ),
        axis=1,
    )

    df["whisper_last"] = df.apply(
        lambda x: last_word_fuzzy_score(
            x["text_normalized"], x["whisper_transcription_normalized"]
        ),
        axis=1,
    )

    df["wav2vec2_first"] = df.apply(
        lambda x: first_word_fuzzy_score(
            x["text_normalized"], x["wav2vec_transcription_normalized"]
        ),
        result_type="expand",
        axis=1,
    )

    df["wav2vec2_last"] = df.apply(
        lambda x: last_word_fuzzy_score(
            x["text_normalized"], x["wav2vec_transcription_normalized"]
        ),
        axis=1,
    )

    df["whisper_cer_head"] = df.apply(
        lambda x: cer_head(x["text_normalized"], x["whisper_transcription_normalized"]),
        result_type="expand",
        axis=1,
    )

    df["whisper_cer_tail"] = df.apply(
        lambda x: cer_tail(x["text_normalized"], x["whisper_transcription_normalized"]),
        axis=1,
    )

    df["wav2vec2_cer_head"] = df.apply(
        lambda x: cer_head(x["text_normalized"], x["wav2vec_transcription_normalized"]),
        result_type="expand",
        axis=1,
    )

    df["wav2vec2_cer_tail"] = df.apply(
        lambda x: cer_tail(x["text_normalized"], x["wav2vec_transcription_normalized"]),
        axis=1,
    )

    return df


def filter_svt(df, stage=args.stage):
    """
    SVT has different filters because we have metadata for when (parts of)
    programs are broadcasted live (as_run) or pre-recorded.

    We trust pre-recorded content more than live content.
    """
    if stage == "stage1":
        df["stage1"] = (
            ((df["bleu_whisper"] >= 0.4) & df["as_run"])
            | ((df["bleu_whisper"] >= 0.15) & ~df["as_run"])
            | df["is_silence"]
        )
    elif stage == "stage2":
        # fmt: off
        df["stage2"] = (
            (((df["bleu_whisper"] >= 0.7) & df["as_run"])
            & (
                (df["whisper_cer_head"] <= 0.3)
                & (df["whisper_cer_tail"] <= 0.3)
            ))
            | (((df["bleu_whisper"] >= 0.4) & ~df["as_run"])
            & (
                (df["whisper_cer_head"] <= 0.4)
                & (df["whisper_cer_tail"] <= 0.4)
            ))
            | df["is_silence"]
        )
        # fmt: on
    elif stage == "stage_wav2vec2":
        df["stage_wav2vec2"] = (
            ((df["whisper_cer_head"] <= 0.2) & (df["whisper_cer_tail"] <= 0.2))
            & ((df["wav2vec2_cer_head"] <= 0.2) & (df["wav2vec2_cer_tail"] <= 0.2))
            & (
                ((df["bleu_whisper"] >= 0.8) & df["as_run"])
                | ((df["bleu_whisper"] >= 0.6) & ~df["as_run"])
            )
        )

    # Training with timestamps is a subset of stage1 and stage2 with additional stricter filters
    # We need this column in those stages to determine when to train with timestamps.
    df["stage2_whisper_timestamps"] = (
        ((df["whisper_cer_head"] <= 0.2) & (df["whisper_cer_tail"] <= 0.2))
        & ((df["wav2vec2_cer_head"] <= 0.4) & (df["wav2vec2_cer_tail"] <= 0.4))
        & (
            ((df["bleu_whisper"] >= 0.8) & df["as_run"])
            | ((df["bleu_whisper"] >= 0.6) & ~df["as_run"])
        )
    )

    return df


# Create a general filter function for all datasets except SVT
# fmt: off
def filter_general(
    df,
    stage=args.stage, 
    stage1_bleu=0.2, 
    stage2_bleu=0.6, stage2_cer_head=0.3, stage2_cer_tail=0.3,
    stage2_bleu_timestamps=0.6, 
    stage2_cer_head_whisper_timestaps=0.2, stage2_cer_tail_whisper_timestaps=0.2,
    stage2_cer_head_wav2vec2_timestaps=0.4, stage2_cer_tail_wav2vec2_timestaps=0.4,
):
    # fmt: on
    """
    General filter function for all datasets except SVT.

    Args:
        df: DataFrame containing audio_tensor, text, text_timestamps and other metadata
        stage: Name of the stage
        stage1_bleu: BLEU threshold for stage1
        stage2_bleu: BLEU threshold for stage2
        stage2_cer_head: CER threshold for the head of the text
        stage2_cer_tail: CER threshold for the tail of the text
        stage2_bleu_timestamps: BLEU threshold for stage2 with timestamps
        stage2_cer_head_whisper_timestaps: CER threshold for the head of the text with timestamps
        stage2_cer_tail_whisper_timestaps: CER threshold for the tail of the text with timestamps
        stage2_cer_head_wav2vec2_timestaps: CER threshold for the head of the text with timestamps
        stage2_cer_tail_wav2vec2_timestaps: CER threshold for the tail of the text with timestamps
    
    Returns:
        df: DataFrame with boolean columns for the different stages
    """
    if stage == "stage1":
        df["stage1"] = (df["bleu_whisper"] >= stage1_bleu) | df["is_silence"]
    elif stage == "stage2":
        df["stage2"] = (
            (
                (df["whisper_cer_head"] <= stage2_cer_head)
                & (df["whisper_cer_tail"] <= stage2_cer_tail)
            )
            & (df["bleu_whisper"] >= stage2_bleu)
            | df["is_silence"]
        )

    elif stage == "stage_wav2vec2":
        df["stage_wav2vec2"] = (
            (
                (df["whisper_cer_head"] <= stage2_cer_head)
                & (df["whisper_cer_tail"] <= stage2_cer_tail)
            )
            & (
                (df["wav2vec2_cer_head"] <= stage2_cer_head)
                & (df["wav2vec2_cer_tail"] <= stage2_cer_tail)
            )
            & ((df["bleu_whisper"] >= stage2_bleu) & (df["bleu_wav2vec2"] >= stage2_bleu))
            | (
                (df["bleu_wav2vec2"] >= 0.80)
                & (df["whisper_cer_head"] <= 0.2)
                & (df["whisper_cer_tail"] <= 0.2)
            )
        )

    # Training with timestamps is a subset of stage1 and stage2 with additional stricter filters
    # We need this column in those stages to determine when to train with timestamps.
    df["stage2_whisper_timestamps"] = (
        (
            (df["whisper_cer_head"] <= stage2_cer_head_whisper_timestaps)
            & (df["whisper_cer_tail"] <= stage2_cer_tail_whisper_timestaps)
        )
        & (
            (df["wav2vec2_cer_head"] <= stage2_cer_head_wav2vec2_timestaps)
            & (df["wav2vec2_cer_tail"] <= stage2_cer_tail_wav2vec2_timestaps)
        )
        & (df["bleu_whisper"] >= stage2_bleu_timestamps)
        | df["is_silence"]
    )

    return df


def filter_dataset(df, config, dataset=args.dataset, stage=args.stage, apply_filter=True):
    """
    Filter the dataset using metrics based on the dataset and stage.

    Args:
        df: DataFrame containing audio_tensor, text, text_timestamps and other metadata
        config: HF model config
        dataset: Name of the dataset
        stage: Name of the stage
        apply_filter: Whether to apply the filter or just return entire df with boolean columns
    Returns:
        df: Filtered DataFrame
    """

    # fmt: off
    if dataset == "svt":
        df = filter_svt(df, stage)
    elif dataset == "rixvox":
        df = filter_general(
            df,
            stage,
            stage1_bleu=0.3,
            stage2_bleu=0.6, stage2_cer_head=0.3, stage2_cer_tail=0.3,
            stage2_bleu_timestamps=0.6
        )
    elif dataset == "smdb":
        df = filter_general(
            df,
            stage,
        )

        # We made too many short chunks in Youtube, so sample a subet of them
        if (len(df) > 5000):
            df_short = df[(df["duration"] <= 10) & ~df["is_silence"]]
            df_long = df[(df["duration"] > 10) | df["is_silence"]]

            # Sample 30% of the short chunks
            df_short = df_short.sample(frac=0.3)

            df = pd.concat([df_short, df_long])
            # Delete and garbage collect the DataFrames
            del df_short, df_long
            gc.collect()
    elif dataset == "youtube":
        df = filter_general(
            df,
            stage,
        )
        # We made too many short chunks in Youtube, so sample a subet of them
        if (len(df) > 5000):
            df_short = df[(df["duration"] <= 15) & ~df["is_silence"]]
            df_long = df[(df["duration"] > 15) | df["is_silence"]]

            # Sample 10% of the short chunks
            df_short = df_short.sample(frac=0.1)

            df = pd.concat([df_short, df_long])
            # Delete and garbage collect the DataFrames
            del df_short, df_long
            gc.collect()

    elif dataset == "isof":
        df = filter_general(
            df,
            stage,
        )
    elif dataset == "sls":
        df = filter_general(
            df,
            stage,
        )
    # fmt: on

    # Remove if audio too short or too long
    df = df[
        (df["input_length"] >= args.min_input_length)
        & (df["input_length"] <= args.max_input_length)
    ]

    # Remove if tokenized text too long
    df = df[(df["labels_length"] <= config.max_length)]

    # Remove if tokenized text with timestamps too long
    logging.info(f'Number of obs exceeding token max length: {len(df[(df["labels_length"] > config.max_length)])}')
    df = df[(df["labels_timestamps_length"] <= config.max_length)]

    if apply_filter:
        # Return only rows from relevant stage
        return df[df[stage]]

    return df


def add_prev_text_column(df, dataset=args.dataset):
    """
    Add previous text column if the timestamps from consecutive rows match.
    Previous text can be used as prompt during training.
    """

    if dataset == "rixvox":
        df = df.sort_values(["speech_id", "chunk_id"])
        df["end_prev"] = df["end"].shift(1)

        # If "start" of the current row is equal to "end_prev" then insert the previous row's text
        df["prev_text_bool"] = df["start"] == df["end_prev"]
        df["previous_text"] = df["text"].shift(1)
        df.loc[~df["prev_text_bool"], "previous_text"] = None
    elif dataset == "svt" or dataset == "smdb" or dataset == "youtube":
        if dataset == "smdb" or dataset == "youtube":
            # sub_ids can sometimes be a string list that needs to be converted to a list with pd.eval
            df["sub_ids"] = df["sub_ids"].apply(pd.eval)

        if dataset == "youtube":
            # Remove rows with no sub_ids
            df["len_subids"] = df["sub_ids"].apply(len)
            df = df[df["len_subids"] > 0]

        df["sub_id_first"] = df["sub_ids"].apply(lambda x: abs(x[0]))
        df["sub_id_last"] = df["sub_ids"].apply(lambda x: abs(x[-1]))
        df = df.sort_values(["audio_path", "sub_id_first"]).reset_index(drop=True)
        df["end_prev"] = df["end"].shift(1)

        df["prev_text_bool"] = df["start"] == df["end_prev"]
        df["previous_text"] = df["text"].shift(1)
        df.loc[~df["prev_text_bool"], "previous_text"] = None
    else:
        df["previous_text"] = None
        logging.warning("No previous text column added for this dataset. Please implement in add_prev_text_column.")

    return df["previous_text"]
        
def prepare_dataset(
    df,
    feature_extractor,
    return_attention_mask,
    tokenizer,
    max_previous_text_length=args.max_previous_text_length,
    dataset=args.dataset,
    is_svt=False,
):
    """
    Args:
        df: DataFrame containing audio_tensor, text, text_timestamps and other metadata
        feature_extractor: feature extractor
        return_attention_mask: whether to return attention mask
        tokenizer: tokenizer
        dataset: Name of the dataset
        is_svt: whether the dataset is from SVT
    """

    # Clean the text to be more consistently formatted
    df["text"] = df["text"].apply(clean_text, args=(is_svt,))
    df["n_words"] = df["text"].apply(lambda x: len(x.split()))
    df["text_timestamps"] = df["text_whisper"].apply(clean_text, args=(is_svt,))
    
    # Add previous text column to use as prompt during training
    df["previous_text"] = add_prev_text_column(df, dataset=dataset)

    # Our timestamp formatting is incorrect, we need <|x.xx|> instead of <x.xx>
    df["text_timestamps"] = df["text_timestamps"].apply(
        lambda x: re.sub(r"(?<=[<])(\d{1,2}\.\d{2})(?=[>])", r"|\1|", x)
    )

    # non-speech segment boolean
    df["is_silence"] = df.apply(
        lambda x: x["text"].strip() == "" and x["wav2vec_transcription"] == "", axis=1
    )

    # Add <|nospeech|> to the text for non-speech segments.
    df["text"] = df.apply(
        lambda x: "<|nospeech|>" + x["text"] if x["is_silence"] else x["text"], axis=1
    )
    df["text_timestamps"] = df.apply(
        lambda x: (
            "<|nospeech|>" + x["text_timestamps"] if x["is_silence"] else x["text_timestamps"]
        ),
        axis=1,
    )

    if "audio_tensor" not in df.columns:
        try:
            df.rename(columns={"audio": "audio_tensor"}, inplace=True)
        except KeyError:
            raise ValueError("No audio_tensor or audio column in the DataFrame.")

    # Get spectogram features and length of the audio
    df[["input_features", "attention_mask", "input_length"]] = df.apply(
        extract_audio_features,
        args=(feature_extractor, return_attention_mask),
        result_type="expand",
        axis=1,
    )

    # Get tokenized labels and lengths
    df[["labels", "labels_timestamps", "labels_length", "labels_timestamps_length"]] = df.apply(
        tokenize_ground_truth,
        args=("text", "text_timestamps", tokenizer),
        result_type="expand",
        axis=1,
    )

    df[["previous_tokens", "previous_tokens_length"]] = df.apply(
        tokenize_prompt,
        args=("previous_text", tokenizer, False, max_previous_text_length),
        result_type="expand",
        axis=1,
    )

    # Calculate all metrics
    df = get_all_metrics(df)

    return df


def write_summary_statistics(df, stats_dir, dataset, stage):
    """
    Write summary statistics for the dataset to disk.

    Args:
        df: DataFrame containing audio_tensor, text, text_timestamps and other metadata
        stats_dir: Directory to save the summary statistics
        dataset: Name of the dataset
        stage: Name of the stage. When getting stats before filtering, use "original".
    """

    df["duration"] = (df["end"] - df["start"]) / 1000
    stats_dict = {
        "n": int(len(df)),
        "n_silence": int(len(df[df["is_silence"]])),
        "n_previous_text": int(sum(df["previous_text"].notnull())),
        "n_words": int(df["n_words"].sum()),
        "n_tokens": int(df["labels_length"].sum()),
        "duration_hours": float(df["duration"].sum() / 3600),
        "duration_hours_silence": float(df[df["is_silence"]]["duration"].sum() / 3600),
    }
    # Count how many rows have previous text

    if stage == "original":
        # Add stats for stage2_whisper_timestamps
        stats_dict["n_stage2_whisper_timestamps"] = None
    else:
        stats_dict["n_stage2_whisper_timestamps"] = int(len(df[df["stage2_whisper_timestamps"]]))

    # # Duration distribution
    # df["duration_group"] = pd.cut(df["duration"], bins=[0, 5, 10, 15, 20, 25, 30])
    # df["duration_freq"] = df.groupby("duration_group")["duration_group"].transform("count")
    # df["duration_rel_freq"] = df["duration_freq"] / len(df)
    # df["duration_group"] = df["duration_group"].astype(str)

    # df[["duration_group", "duration_freq", "duration_rel_freq"]].drop_duplicates().sort_values(
    #     "duration_group"
    # )

    stats_file = Path(stats_dir) / stage / dataset / f"{args.parquet_filename}.json"
    os.makedirs(stats_file.parent, exist_ok=True)
    pd.Series(stats_dict).to_json(stats_file, indent=4)


if __name__ == "__main__":

    try:
        logging.info(f"Beginning {args.stage} preprocessing of {args.dataset} dataset.")

        # 0. If output file already exists and is not empty, skip the preprocessing
        data_dir = Path(args.data_dir).parts[-1]
        output_dir = Path(args.output_dir) / args.stage / data_dir
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = output_dir / args.parquet_filename
        
        if os.path.exists(output_path) and not args.overwrite and not args.stats_only:
            if os.path.getsize(output_path) > 0:
                logging.info(f"File {output_path} exists and is not empty. Skipping preprocessing.")
                exit()

        # 1. Load the input parquet file
        input_path = os.path.join(args.data_dir, args.parquet_filename)
        logging.info(f"Loading the dataset: {input_path}.")
        df = pd.read_parquet(input_path)

        # 2. Load pretrained model config, tokenizer, and feature extractor
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            add_prefix_space=True,
        )

        # 3. Regularization settings.
        #   a) BPE dropout in the tokenizer (randomly uses different subwords to encode the same word)
        if args.bpe_dropout > 0:
            # Need a workaround to successfully load the tokenizer with BPE dropout.
            # See https://github.com/huggingface/tokenizers/issues/201#issue-584182286
            # Should only be used for training, not for inference/eval.
            workaround_files = tokenizer._tokenizer.model.save(args.cache_dir, "training_tokenizer")
            tokenizer._tokenizer.model = BPE.from_file(*workaround_files, dropout=args.bpe_dropout)

        tokenizer.set_prefix_tokens(
            language=args.language, task=args.task
        )  # Set predict_timestamps later

        #   b) SpecAugment (this doesn't apply SpecAugment, but sets feature extractor to return attention mask)
        config.apply_spec_augment = args.apply_spec_augment
        config.mask_time_prob = args.mask_time_prob

        # Whether or not to return attention mask is decided by whether we are using SpecAugment or not
        return_attention_mask = (
            getattr(config, "model_type", None) == "whisper"
            and getattr(config, "apply_spec_augment", False)
            and getattr(config, "mask_time_prob", 0) > 0
        )

        #### 4. Preprocessing step to clean text, tokenize, get feature vectors, re-calculate metrics, and apply filters
        logging.info(f"Preprocessing the dataset: {input_path}.")
        df = prepare_dataset(
            df,
            feature_extractor,
            return_attention_mask,
            tokenizer,
            is_svt=args.dataset == "svt",
        )

        # # 4a) Sanity check that audio tensor input_length roughly matches duration metadata
        # df["input_length"] = df["audio_tensor"].apply(len)
        # df["duration_tensor"] = df["input_length"] / args.sampling_rate

        # 4b) Statistics for the dataset before filtering
        write_summary_statistics(df, args.stats_dir, args.dataset, stage="original")

        #### 5. Filter the dataset based on the stage and dataset
        df = filter_dataset(df, config, args.dataset, args.stage, apply_filter=True)
            
        # 5b) Statistics for the dataset after filtering
        write_summary_statistics(df, args.stats_dir, args.dataset, args.stage)

        if args.stats_only:
            logging.info("Summary statistics written to disk. Exiting.")
            exit()

        # Add data source to the DataFrame
        df["data_source"] = args.dataset
        
        #### 6. Save the processed DataFrame to disk as a parquet file
        # Standardize audio_path name
        if "audio_path" in df.columns:
            pass
        elif "audio_file" in df.columns:
            df.rename(columns={"audio_file": "audio_path"}, inplace=True)
        
        # pandas can't save 2d array to parquet with pandas, so convert to list of arrays
        df["input_features"] = df["input_features"].apply(lambda x: list(x))
        
        # 6a) Select relevant columns
        df = df[[
            "input_features",
            "attention_mask",
            "labels",
            "labels_timestamps",
            "text",
            "text_timestamps",
            "previous_text",
            "previous_tokens",
            "duration",
            "audio_path",
            "is_silence",
            "stage2_whisper_timestamps",
            "data_source"
        ]].reset_index(drop=True)
        
        # 6b) Save the processed DataFrame to disk as a parquet file
        # Get the last directory in the data_dir path
        data_dir = Path(args.data_dir).parts[-1]
        output_dir = Path(args.output_dir) / args.stage / data_dir
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = output_dir / args.parquet_filename

        logging.info(f"Saving the processed dataset to: {output_filename}.")
        df.to_parquet(output_filename, index=False)
    except Exception as e:
        logging.exception(f"Error in preprocessing: {e}", stack_info=True)
