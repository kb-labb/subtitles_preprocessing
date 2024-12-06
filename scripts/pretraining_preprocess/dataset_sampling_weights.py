import json

import pandas as pd

json_str = """{
    "youtube": {
        "dataset": "youtube",
        "n_obs": 1237708.0,
        "n_silence": 394316.0,
        "n_words": 36385971.0,
        "n_tokens": 69872094.0,
        "duration_hours": 4971.4165333338,
        "duration_hours_silence": 238.6921974995,
        "mean_duration_obs_s": 14.459872215418887,
        "mean_duration_silence_s": 2.179196154856004
    },
    "smdb": {
        "dataset": "smdb",
        "n_obs": 1900556.0,
        "n_silence": 13.0,
        "n_words": 53528369.0,
        "n_tokens": 101483402.0,
        "duration_hours": 8559.788228332698,
        "duration_hours_silence": 0.030888889099999997,
        "mean_duration_obs_s": 16.213801446522865,
        "mean_duration_silence_s": 8.553846212307691
    },
    "riksdagen_web": {
        "dataset": "riksdagen_web",
        "n_obs": 862439.0,
        "n_silence": 68.0,
        "n_words": 47531927.0,
        "n_tokens": 92281056.0,
        "duration_hours": 5827.0172900004,
        "duration_hours_silence": 0.0372258336,
        "mean_duration_obs_s": 24.323183719661845,
        "mean_duration_silence_s": 1.970779425882353
    },
    "riksdagen_old": {
        "dataset": "riksdagen_old",
        "n_obs": 2428568.0,
        "n_silence": 2513.0,
        "n_words": 115584992.0,
        "n_tokens": 237152506.0,
        "duration_hours": 16130.168095277699,
        "duration_hours_silence": 1.5091105561,
        "mean_duration_obs_s": 23.91063587389759,
        "mean_duration_silence_s": 2.1618774381058494
    },
    "svt1": {
        "dataset": "svt1",
        "n_obs": 1313111.0,
        "n_silence": 2.0,
        "n_words": 49997184.0,
        "n_tokens": 95940238.0,
        "duration_hours": 8796.831088889201,
        "duration_hours_silence": 0.0027777778,
        "mean_duration_obs_s": 24.117223844748178,
        "mean_duration_silence_s": 5.000000040000001
    },
    "svt2": {
        "dataset": "svt2",
        "n_obs": 1412842.0,
        "n_silence": 0.0,
        "n_words": 53363399.0,
        "n_tokens": 102443433.0,
        "duration_hours": 9521.7411888891,
        "duration_hours_silence": 0.0,
        "mean_duration_obs_s": 24.261926160179808,
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

#
df["last_exhaust_repetition"] = df["relative_duration_hours"] / df["relative_n_obs"]
df["last_exhaust_repetition"] = df["last_exhaust_repetition"] / df["last_exhaust_repetition"].min()
df["first_exhaust_repetition"] = df["relative_duration_hours"] / df["relative_n_obs"]
df["first_exhaust_repetition"] = (
    df["first_exhaust_repetition"] / df["first_exhaust_repetition"].max()
)


df[
    [
        "dataset",
        "relative_n_obs",
        "relative_duration_hours",
        "last_exhaust_repetition",
        "first_exhaust_repetition",
    ]
]
