import argparse
import glob
import json
import os

import pysrt
from tqdm import tqdm

from sub_preproc.utils.make_chunks import make_chunks
from sub_preproc.utils.utils import decode_program_id, subrip_to_dict, decode_svt_id

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder")

    return parser.parse_args()

def compute_chunks(
    subs_dict, thresholds=[(1, 5_000), (5_001, 10_000), (10_001, 20_000), (20_001, 30_000)]
):
    # make chunks
    subs_dict["chunks"] = []
    for mini, maxi in thresholds:
        subs_dict = make_chunks(subs_dict, min_threshold=mini, max_threshold=maxi)

    return subs_dict

def main():
    args = get_args()

    for fn in tqdm(glob.iglob(f"{args.folder}/**/*.srt", recursive=True)):
        name = "/".join(fn.split("/")[-1:])[:-4]
        path = "/".join(fn.split("/")[:-1])
        videoinfo = path + "/videoinfo.json"
        with open(videoinfo) as f:
            data = json.load(f)
            audiofile = data['filename'] 
            audio_path = path + "/" + audiofile+".wav" 
        subs_dict = subrip_to_dict(pysrt.open(fn), audio_path, name, -1, -1,-1,-1,-1, -1, "svt")
        thresholds=[(1, 5_000), (1, 30_000)]
        subs_dict = compute_chunks(subs_dict, thresholds)

        with open(f"{path}/{audiofile}.sv.json", "w") as fout:
            json.dump(subs_dict, fout, indent=4)


if __name__ == "__main__":
    main()
