import json

import pandas as pd

json_str = """{
    "smdb": {
        "dataset": "smdb",
        "n_obs": 206614.0,
        "n_silence": 12.0,
        "n_previous_text": 42574.0,
        "n_words": 4302963.0,
        "n_tokens": 8283899.0,
        "duration_hours": 740.9813766668,
        "duration_hours_silence": 0.0162222224,
        "mean_duration_obs_s": 12.91070767712004,
        "mean_duration_silence_s": 4.86666672
    },
    "isof": {
        "dataset": "isof",
        "n_obs": 7114.0,
        "n_silence": 1.0,
        "n_previous_text": 0.0,
        "n_words": 502798.0,
        "n_tokens": 833995.0,
        "duration_hours": 54.4540722221,
        "duration_hours_silence": 0.0061944444,
        "mean_duration_obs_s": 27.556179364571268,
        "mean_duration_silence_s": 22.299999839999998
    },
    "nst": {
        "dataset": "nst",
        "n_obs": 153396.0,
        "n_silence": 0.0,
        "n_previous_text": 0.0,
        "n_words": 1380654.0,
        "n_tokens": 3491518.0,
        "duration_hours": 223.1193055555,
        "duration_hours_silence": 0.0,
        "mean_duration_obs_s": 5.236313202429008,
        "mean_duration_silence_s": null
    },
    "youtube": {
        "dataset": "youtube",
        "n_obs": 620362.0,
        "n_silence": 189061.0,
        "n_previous_text": 240253.0,
        "n_words": 18684087.0,
        "n_tokens": 36281414.0,
        "duration_hours": 2520.7877174989,
        "duration_hours_silence": 102.8020447214,
        "mean_duration_obs_s": 14.628290873709286,
        "mean_duration_silence_s": 1.9575023986810607
    },
    "riksdagen_web": {
        "dataset": "riksdagen_web",
        "n_obs": 359868.0,
        "n_silence": 66.0,
        "n_previous_text": 342200.0,
        "n_words": 19701901.0,
        "n_tokens": 39236581.0,
        "duration_hours": 2407.4258233338996,
        "duration_hours_silence": 0.0370133336,
        "mean_duration_obs_s": 24.08308869919537,
        "mean_duration_silence_s": 2.0189091054545454
    },
    "riksdagen_old": {
        "dataset": "riksdagen_old",
        "n_obs": 947921.0,
        "n_silence": 2465.0,
        "n_previous_text": 883475.0,
        "n_words": 45025857.0,
        "n_tokens": 95188339.0,
        "duration_hours": 6300.2753194442,
        "duration_hours_silence": 1.505094167,
        "mean_duration_obs_s": 23.927090073960933,
        "mean_duration_silence_s": 2.198109128275862
    },
    "svt1": {
        "dataset": "svt1",
        "n_obs": 280150.0,
        "n_silence": 2.0,
        "n_previous_text": 220572.0,
        "n_words": 9952548.0,
        "n_tokens": 18966572.0,
        "duration_hours": 1790.5286666664,
        "duration_hours_silence": 0.0027777778,
        "mean_duration_obs_s": 23.00875673745865,
        "mean_duration_silence_s": 5.000000040000001
    },
    "svt2": {
        "dataset": "svt2",
        "n_obs": 286021.0,
        "n_silence": 0.0,
        "n_previous_text": 225717.0,
        "n_words": 10102230.0,
        "n_tokens": 19261574.0,
        "duration_hours": 1841.1050777779,
        "duration_hours_silence": 0.0,
        "mean_duration_obs_s": 23.173047713281335,
        "mean_duration_silence_s": null
    }
}"""


df = pd.DataFrame(json.loads(json_str)).T

# Relative frequencies from n_obs
df["n_obs"] = df["n_obs"].astype(float)

# Group by dataset and calculate relative frequencies for n_obs
df["relative_n_obs"] = df.groupby("dataset")["n_obs"].sum() / df["n_obs"].sum()
# Group by dataset and calculate relative frequencies duration_hours
df["relative_duration_hours"] = (
    df.groupby("dataset")["duration_hours"].sum() / df["duration_hours"].sum()
)

# Reweight NST
df.loc[df["dataset"] == "nst", "relative_duration_hours"] = (
    df.loc[df["dataset"] == "nst", "relative_duration_hours"] * 2.5
)
df["relative_duration_hours"] = (
    df["relative_duration_hours"] / df["relative_duration_hours"].sum()
)

df["last_exhaust_repetition"] = df["relative_duration_hours"] / df["relative_n_obs"]
df["last_exhaust_repetition"] = (
    df["last_exhaust_repetition"] / df["last_exhaust_repetition"].min()
)
df["first_exhaust_repetition"] = df["relative_duration_hours"] / df["relative_n_obs"]
df["first_exhaust_repetition"] = (
    df["first_exhaust_repetition"] / df["first_exhaust_repetition"].max()
)

# Round relative_duration_hours to 3 decimals
df[["relative_duration_hours", "last_exhaust_repetition", "first_exhaust_repetition", "relative_n_obs"]] = df[
    ["relative_duration_hours", "last_exhaust_repetition", "first_exhaust_repetition", "relative_n_obs"]
].apply(pd.to_numeric)


df = df.round({"relative_duration_hours": 4, "last_exhaust_repetition": 4, "first_exhaust_repetition": 4, "relative_n_obs": 4})


print(
    df[
        [
            "dataset",
            "n_obs",
            "duration_hours",
            "relative_n_obs",
            "relative_duration_hours",
            "last_exhaust_repetition",
            "first_exhaust_repetition",
        ]
    ].to_string(index=False)
)

df["relative_duration_hours"].sum()
