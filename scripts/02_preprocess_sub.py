import pysrt
import glob
import json
import argparse
from tqdm import tqdm
from sub_preproc.utils.utils import subrip_to_dict, decode_program_id
from sub_preproc.utils.make_chunks import make_chunks


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder")

    return parser.parse_args()


def main():
    args = get_args()

    for fn in tqdm(glob.iglob(f"{args.folder}/**/file.srt", recursive=True)):
        path = "/".join(fn.split("/")[:-1])
        subs_dict = subrip_to_dict(pysrt.open(fn), *decode_program_id(fn.split("/")[-2]))
        subs_dict = make_chunks(subs_dict)
        with open(f"{path}/file.json", "w") as fout:
            json.dump(subs_dict, fout, indent=4)


if __name__ == "__main__":
    main()
