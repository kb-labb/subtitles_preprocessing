import argparse
import glob
import json
import os

import pysrt
from tqdm import tqdm

import sub_preproc.utils.utils as utils
from sub_preproc.utils.make_chunks import make_chunks
from sub_preproc.utils.utils import decode_program_id, subrip_to_dict, decode_svt_id
from sub_preproc.dedup import dup_marker_single_list

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
    seen = {}

    filenames = []
    with open(args.folder) as fh:
        if args.folder.endswith(".json"):
            filenames = json.load(fh)
        else:
            for line in fh:
                filenames.append(line.strip())
    
    for fn in filenames:
        print('f ', fn)
        name = "/".join(fn.split("/")[-1:])[:-4]
        path = "/".join(fn.split("/")[:-1])
        info = "/".join(path.split("/")[-1:])
        metadata = path +"/"+info+".json"
        with open(metadata) as f:
            data = json.load(f)
            audiofile = data['metadata']['video_file'] 
            audio_path = path + "/" + audiofile+".wav" 
        subs = pysrt.open(fn)
        # fuse subtitles
        fused = utils.fuse_subtitles(subs)
        # livesub marking
        live_subbed = utils.mark_live_subs(fused)
        # make dict
        subs_dict = subrip_to_dict(live_subbed, audio_path, name, -1, -1,-1,-1,-1, -1, "svt")
        # deduplicate
        subs_dict['subs'], seen = dup_marker_single_list(subs_dict["subs"], seen)
        thresholds=[(1, 5_000), (1, 30_000)]
        subs_dict = compute_chunks(subs_dict, thresholds)
        with open(f"{path}/{audiofile}.sv.json", "w") as fout:
            json.dump(subs_dict, fout, indent=4)


if __name__ == "__main__":
    main()
