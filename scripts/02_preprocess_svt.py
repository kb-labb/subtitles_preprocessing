import argparse
import glob
import json

import pysrt
from tqdm import tqdm

from sub_preproc.utils.make_chunks import make_chunks
from sub_preproc.utils.utils import decode_program_id, subrip_to_dict, decode_svt_id

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder")

    return parser.parse_args()


def main():
    args = get_args()

    for fn in tqdm(glob.iglob(f"{args.folder}/**/*.srt", recursive=True)):
        name = "/".join(fn.split("/")[-1:])[:-9]
        print('name ', name)
        path = "/".join(fn.split("/")[:-1])
        print('path ', path)
        subs_dict = subrip_to_dict(pysrt.open(fn), name,-1,-1,-1,-1,-1, -1, "svt")
        subs_dict = make_chunks(subs_dict)
        with open(f"{path}/{name}.json", "w") as fout:
            json.dump(subs_dict, fout, indent=4)


if __name__ == "__main__":
    main()
