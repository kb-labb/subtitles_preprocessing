import pandas as pd

df = pd.read_parquet("/data/jussik/svt_statistics/svt_stats_new_norm/svt2_metadata_stats_new_normalize.parquet")

#df["duration"] = df["end_time"] - df["start_time"] #if not present in the file

columns_to_analyze = [
    "stage1_whisper",
    "stage2_whisper",
    "stage2_whisper_timestamps",
    "stage1_wav2vec",
]

results = {}

for col in columns_to_analyze:
    # Count of True and False for the current filter
    true_count = df[col].sum()
    false_count = len(df) - true_count

    total_duration_true = (
        df[df[col] == True]["duration"].sum() / 3_600_000
    )  
    total_duration_false = (
        df[df[col] == False]["duration"].sum() / 3_600_000
    ) 

    # Calculate number of unique files in 'audio' for True and False
    num_of_files_true = df[df[col] == True]["caption_file"].nunique()
    num_of_files_false = df[df[col] == False]["caption_file"].nunique()

    # Calculate number of True and False in 'is_asrun' for True and False filter values
    asrun_true_true_count = df[(df[col] == True) & (df["is_asrun"] == True)].shape[0]
    asrun_true_false_count = df[(df[col] == False) & (df["is_asrun"] == True)].shape[0]
    asrun_false_true_count = df[(df[col] == True) & (df["is_asrun"] == False)].shape[0]
    asrun_false_false_count = df[(df[col] == False) & (df["is_asrun"] == False)].shape[
        0
    ]

    results[col] = {
        "True_count": true_count,
        "False_count": false_count,
        "Total_duration_true_hours": total_duration_true,
        "Total_duration_false_hours": total_duration_false,
        "Num_of_files_true": num_of_files_true,
        "Num_of_files_false": num_of_files_false,
        "Asrun_True_True_count": asrun_true_true_count,
        "Asrun_True_False_count": asrun_true_false_count,
        "Asrun_False_True_count": asrun_false_true_count,
        "Asrun_False_False_count": asrun_false_false_count,
    }


results_df = pd.DataFrame(results).T

results_df.to_csv("/data/jussik/svt2_statistics_new_normalize.csv", index=True)

##############################################################################################################

columns_to_analyze = [
    "stage1_whisper",
    "stage2_whisper",
    "stage2_whisper_timestamps",
    "stage1_wav2vec",
]

results_list = []

# Group by 'program_title' and calculate statistics for each filter
for program_title, group in df.groupby("program_title"):
    program_data = {"Program_title": program_title}

    for col in columns_to_analyze:
        true_count = group[col].sum()
        false_count = len(group) - true_count

        program_data[f"{col}_True_count"] = true_count
        program_data[f"{col}_False_count"] = false_count

    program_data["Total_appearances"] = len(group)

    results_list.append(program_data)

results_df = pd.DataFrame(results_list)

results_df.sort_values(
    by="Total_appearances", ascending=False, inplace=True, ignore_index=True
)

results_df.to_csv("svt2_program_statistics_new_normalize.csv", index=False)
