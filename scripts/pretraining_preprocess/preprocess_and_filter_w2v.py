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
    calculate_rouge,
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
    required=True,
    # default="/leonardo_work/EUHPC_A01_006/data/big_parquets/riksdagen_old",
    help="Directory where all the parquets are stored.",
)
argparser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    # default="/leonardo_scratch/fast/EUHPC_A01_006",
    help="Directory where the processed data will be saved.",
)
argparser.add_argument(
    "--parquet_filename",
    type=str,
    required=True,
    # default="riksdagen_old_0528.parquet",
    help="Filename of the parquet file to process.",
)
argparser.add_argument(
    "--dataset",
    type=str,
    choices=["svt", "rixvox", "smdb", "youtube", "isof", "sls", "nst"],
    default="rixvox",
    help="Dataset source to preprocess.",
)
argparser.add_argument(
    "--stage",
    type=str,
    choices=["stage1", "stage2", "stage_wav2vec2", "stage2_svt"],
    default="stage_wav2vec2",
    help="""Which quality filters to use for creating different stages
    of pretraining corpora (dataset annealing).""",
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

argparser.add_argument(
    "--min_input_length",
    type=int,
    default=800,
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

    # Remove if HTML tag containing letters or whitespace, e.g. < p>
    # But don't target <> that has numbers or punctuation like <|2.24|>.
    text = re.sub(r"<[/a-zA-Z\s]+>", "", text)

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

    df[["rouge_whisper", "rouge_wav2vec2"]] = df.apply(
        calculate_metrics,
        args=(calculate_rouge, normalize_text_fun),
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


def filter_svt(df):
    """
    SVT has different filters because we have metadata for when (parts of)
    programs are broadcasted live (as_run) or pre-recorded.

    We trust pre-recorded content more than live content.
    """

    df["stage_wav2vec2"] = (
        ((df["whisper_cer_head"] <= 0.2) & (df["whisper_cer_tail"] <= 0.2))
        & ((df["wav2vec2_cer_head"] <= 0.2) & (df["wav2vec2_cer_tail"] <= 0.2))
        & (
            (((df["bleu_whisper"] >= 0.9) & (df["rouge_whisper"] > 0.9)) & df["as_run"])
            | (((df["bleu_whisper"] >= 0.8) & (df["rouge_whisper" > 0.8])) & ~df["as_run"])
        )
    )

    return df


# Create a general filter function for all datasets except SVT
# fmt: off
def filter_general(
    df,
    stage2_bleu=0.8, 
    stage2_rouge=0.8,
    stage2_cer_head=0.2, stage2_cer_tail=0.2,
    stage2_cer_head_whisper_timestamps=0.2, stage2_cer_tail_whisper_timestamps=0.2,
    stage2_cer_head_wav2vec2_timestamps=0.4, stage2_cer_tail_wav2vec2_timestamps=0.4,
):
    # fmt: on
    """
    General filter function for all datasets except SVT.

    Args:
        df: DataFrame containing audio_tensor, text, text_timestamps and other metadata
        stage2_bleu: BLEU threshold for stage2
        stage2_cer_head: CER threshold for the head of the text
        stage2_cer_tail: CER threshold for the tail of the text
        stage2_bleu_timestamps: BLEU threshold for stage2 with timestamps
        stage2_cer_head_whisper_timestamps: CER threshold for the head of the text with timestamps
        stage2_cer_tail_whisper_timestamps: CER threshold for the tail of the text with timestamps
        stage2_cer_head_wav2vec2_timestamps: CER threshold for the head of the text with timestamps
        stage2_cer_tail_wav2vec2_timestamps: CER threshold for the tail of the text with timestamps
    
    Returns:
        df: DataFrame with boolean columns for the different stages
    """

    df["stage_wav2vec2"] = (
        (
            (df["whisper_cer_head"] <= stage2_cer_head)
            & (df["whisper_cer_tail"] <= stage2_cer_tail)
        )
        & (
            (df["wav2vec2_cer_head"] <= stage2_cer_head)
            & (df["wav2vec2_cer_tail"] <= stage2_cer_tail)
        )
        & ((df["bleu_whisper"] >= stage2_bleu) & (df["bleu_wav2vec2"] >= stage2_bleu) 
            & (df["rouge_whisper"] >= stage2_rouge) & (df["rouge_wav2vec2"] >= stage2_rouge))
        | (
            ((df["bleu_wav2vec2"] >= 0.85) & (df["rouge_wav2vec2"] >= 0.85))
            & (df["whisper_cer_head"] <= stage2_cer_head_whisper_timestamps)
            & (df["whisper_cer_tail"] <= stage2_cer_tail_whisper_timestamps)
        )
    )

    return df


def filter_dataset(df, dataset=args.dataset, stage=args.stage, apply_filter=True):
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
            stage2_bleu=0.8,
            stage2_rouge=0.8, 
            stage2_cer_head=0.2, stage2_cer_tail=0.2,
            stage2_cer_head_whisper_timestamps=0.2, stage2_cer_tail_whisper_timestamps=0.2,
        )
    elif dataset == "smdb":
        df = filter_general(
            df,
            stage2_bleu=0.8,
            stage2_rouge=0.8, 
            stage2_cer_head=0.2, stage2_cer_tail=0.2,
            stage2_cer_head_whisper_timestamps=0.2, stage2_cer_tail_whisper_timestamps=0.2,
        )

        # We made too many short chunks in Youtube, so sample a subet of them
        if (len(df) > 5000):
            df_short = df[(df["duration"] <= 10) & ~df["is_silence"]]
            df_long = df[(df["duration"] > 10) | df["is_silence"]]

            # Sample 45% of the short chunks
            df_short = df_short.sample(frac=0.45)

            df = pd.concat([df_short, df_long])
            # Delete and garbage collect the DataFrames
            del df_short, df_long
            gc.collect()
    elif dataset == "youtube":
        df = filter_general(
            df,
            stage2_bleu=0.8,
            stage2_rouge=0.8, 
            stage2_cer_head=0.2, stage2_cer_tail=0.2,
            stage2_cer_head_whisper_timestamps=0.2, stage2_cer_tail_whisper_timestamps=0.2,
        )
        # We made too many short chunks in Youtube, so sample a subet of them
        if (len(df) > 5000):
            df_short = df[(df["duration"] <= 15) & ~df["is_silence"]]
            df_long = df[(df["duration"] > 15) | df["is_silence"]]

            # Sample 30% of the short chunks
            df_short = df_short.sample(frac=0.3)

            df = pd.concat([df_short, df_long])
            # Delete and garbage collect the DataFrames
            del df_short, df_long
            gc.collect()

    elif dataset == "isof":
        df["duration"] = df["end"] - df["start"]

        df = filter_general(
            df,
            stage2_bleu=0.01,
            stage2_cer_head=1, stage2_cer_tail=1,
            stage2_rouge=0.01,
        )
    elif dataset == "sls":
        df = filter_general(
            df
        )
    elif dataset == "nst":
        df = filter_general(
            df,
            stage2_rouge=0,
            stage2_bleu=0, 
            stage2_cer_head=1, stage2_cer_tail=1,
            stage2_cer_head_whisper_timestamps=0, stage2_cer_tail_whisper_timestamps=0,
        )
    # fmt: on

    df["input_length"] = df["audio_tensor"].apply(len)
    # Remove if audio too short or too long
    df = df[
        (df["input_length"] >= args.min_input_length)
        & (df["input_length"] <= args.max_input_length)
    ]

    if apply_filter:
        # Return only rows from relevant stage
        return df[df[stage]]

    return df


def prepare_dataset(
    df,
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

    # non-speech segment boolean
    df["is_silence"] = df.apply(
        lambda x: x["text"].strip() == "" and x["wav2vec_transcription"] == "", axis=1
    )

    if "audio_tensor" not in df.columns:
        try:
            df.rename(columns={"audio": "audio_tensor"}, inplace=True)
        except KeyError:
            raise ValueError("No audio_tensor or audio column in the DataFrame.")
    if dataset == "isof":
        df.rename(columns={"start_time": "start", "end_time": "end", "audio": "audio_path"}, inplace=True)

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
        "n_words": int(df["n_words"].sum()),
        "duration_hours": float(df["duration"].sum() / 3600),
        "duration_hours_silence": float(df[df["is_silence"]]["duration"].sum() / 3600),
    }
    # Count how many rows have previous text

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
        if args.dataset == "nst":
            df = df[~df.text_whisper.str.contains("\\\\Komma")]
            df = df[~df.text_whisper.str.contains("\\\\Punkt")]


        #### 4. Preprocessing step to clean text, re-calculate metrics, and apply filters
        logging.info(f"Preprocessing the dataset: {input_path}.")
        df = prepare_dataset(
            df,
            is_svt=args.dataset == "svt",
        )

        # # 4a) Sanity check that audio tensor input_length roughly matches duration metadata
        # df["input_length"] = df["audio_tensor"].apply(len)
        # df["duration_tensor"] = df["input_length"] / args.sampling_rate

        # 4b) Statistics for the dataset before filtering
        write_summary_statistics(df, args.stats_dir, args.dataset, stage="original")

        #### 5. Filter the dataset based on the stage and dataset
        df = filter_dataset(df, args.dataset, args.stage, apply_filter=True)

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
        
        # 6a) Select relevant columns
        df = df[[
            "text",
            "text_normalized",
            "duration",
            "audio_tensor",
            "audio_path",
            "is_silence",
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
        logging.info(f"Finished preprocessing and saved the dataset to: {output_filename}.")
    except Exception as e:
        logging.exception(f"Error in preprocessing: {e}", stack_info=True)
