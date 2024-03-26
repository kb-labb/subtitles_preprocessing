import pysrt
import glob
import json
import argparse
from tqdm import tqdm
from sub_preproc.utils.utils import subrip_to_dict, textgrid_to_dict, decode_program_id, decode_swedia_id
from sub_preproc.utils.make_chunks import make_chunks


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder")

    return parser.parse_args()


def main():
    args = get_args()
    for fn in tqdm(glob.iglob(f"{args.folder}/**/*.TextGrid", recursive=True)):
        name = "/".join(fn.split("/")[-1:])[:-9]
        path = "/".join(fn.split("/")[:-1])
        subs_dict = textgrid_to_dict(fn, *decode_swedia_id(name), "swedia")
        subs_dict = make_chunks(subs_dict)
        with open(f"{path}/{name}.json", "w") as fout:
            json.dump(subs_dict, fout, indent=4)



if __name__ == "__main__":
    main()
