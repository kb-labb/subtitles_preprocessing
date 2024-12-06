import argparse
import glob
import json
import os

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Summarize shard stats")
parser.add_argument(
    "--stage",
    type=str,
    choices=["original", "stage1", "stage2", "stage2_wav2vec2"],
    default="stage1",
    help="Stage to summarize",
)
parser.add_argument(
    "--data_dir",
    type=str,
    help="Directory containing the stage stats data",
    default="/leonardo_work/EUHPC_A01_006/data/big_parquets/stats_alt",
)
args = parser.parse_args()


def get_summary_statistics(df, dataset):
    n_obs = float(df["n"].sum())
    n_silence = float(df["n_silence"].sum())
    n_previous_text = float(df["n_previous_text"].sum())
    n_words = float(df["n_words"].sum())
    n_tokens = float(df["n_tokens"].sum())
    duration = float(df["duration_hours"].sum())
    duration_silence = float(df["duration_hours_silence"].sum())
    mean_duration_obs_s = (duration / n_obs) * 3600

    try:
        mean_duration_silence_s = (duration_silence / n_silence) * 3600
    except ZeroDivisionError:
        mean_duration_silence_s = None

    return {
        "dataset": dataset,
        "n_obs": n_obs,
        "n_silence": n_silence,
        "n_previous_text": n_previous_text,
        "n_words": n_words,
        "n_tokens": n_tokens,
        "duration_hours": duration,
        "duration_hours_silence": duration_silence,
        "mean_duration_obs_s": mean_duration_obs_s,
        "mean_duration_silence_s": mean_duration_silence_s,
    }


if __name__ == "__main__":
    print(f"Summarizing stats for stage: {args.stage}")

    #### Stats according to dataset (youtube, smdb, rixvox, svt, ...) ####
    dataset_dirs = glob.glob(os.path.join(args.data_dir, args.stage, "*"))
    dataset_dirs = [d for d in dataset_dirs if os.path.isdir(d)]
    datasets = [os.path.basename(d) for d in dataset_dirs]

    stats_dict = {dataset: [] for dataset in datasets}
    for dataset, dataset_dir in tqdm(zip(datasets, dataset_dirs), total=len(datasets)):
        shard_files = glob.glob(dataset_dir + "/*")

        for file in shard_files:
            with open(file, "r") as f:
                data = json.load(f)
                data["filename"] = os.path.basename(file)
                stats_dict[dataset].append(data)

    all_dataframes = []
    all_stats = {}

    for dataset in datasets:
        df = pd.DataFrame(stats_dict[dataset])
        all_dataframes.append(df)

        summary_stats = get_summary_statistics(df, dataset)
        all_stats[dataset] = summary_stats

    # Write summary stats to file
    output_file = os.path.join(args.data_dir, args.stage, f"dataset_stats.json")

    with open(output_file, "w") as f:
        json.dump(all_stats, f, indent=4)

    print(f"Summary stats for all datasets written to {output_file}")

    #### Combined stats for all datasets ####
    # Combine all dataframes
    df_all = pd.concat(all_dataframes)

    # Summary stats for the entire stage
    summary_stats = get_summary_statistics(df_all, args.stage)
    output_file = os.path.join(args.data_dir, args.stage, f"{args.stage}_total_stats.json")
    print(f"Writing summary stats for the entire stage to {output_file}")

    # Write summary stats to file
    with open(output_file, "w") as f:
        json.dump(summary_stats, f, indent=4)

    #### Stats according to folder structure (riksdagen_web, riksdagen_old, svt, svt2, smdb, ...) ####
    stats_dict["riksdagen_web"] = []
    stats_dict["riksdagen_old"] = []
    stats_dict["svt1"] = []
    stats_dict["svt2"] = []

    # Split stats for rixvox into riksdagen_web and riksdagen_old
    # and for svt into svt1 and svt2
    for dataset_dir in tqdm(dataset_dirs):
        if "rixvox" in dataset_dir:
            shard_files = glob.glob(dataset_dir + "/*")
            riksdagen_web = [f for f in shard_files if "riksdagen_web" in f]
            riksdagen_old = [f for f in shard_files if "riksdagen_old" in f]

            for file in riksdagen_web:
                with open(file, "r") as f:
                    data = json.load(f)
                    data["filename"] = os.path.basename(file)
                    stats_dict["riksdagen_web"].append(data)

            for file in riksdagen_old:
                with open(file, "r") as f:
                    data = json.load(f)
                    data["filename"] = os.path.basename(file)
                    stats_dict["riksdagen_old"].append(data)
        elif "svt" in dataset_dir:
            shard_files = glob.glob(dataset_dir + "/*")
            svt1 = [f for f in shard_files if "svt1" in f]
            svt2 = [f for f in shard_files if "svt2" in f]

            for file in svt1:
                with open(file, "r") as f:
                    data = json.load(f)
                    data["filename"] = os.path.basename(file)
                    stats_dict["svt1"].append(data)

            for file in svt2:
                with open(file, "r") as f:
                    data = json.load(f)
                    data["filename"] = os.path.basename(file)
                    stats_dict["svt2"].append(data)

    for dataset in ["riksdagen_web", "riksdagen_old", "svt1", "svt2"]:
        df = pd.DataFrame(stats_dict[dataset])

        summary_stats = get_summary_statistics(df, dataset)
        all_stats[dataset] = summary_stats

    _ = all_stats.pop("rixvox")
    _ = all_stats.pop("svt")

    # Write summary stats to file
    output_file = os.path.join(args.data_dir, args.stage, f"folder_stats.json")

    with open(output_file, "w") as f:
        json.dump(all_stats, f, indent=4)

    print(f"Summary stats for all folders written to {output_file}")
