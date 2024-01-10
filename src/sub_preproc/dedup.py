import glob
import time
from typing import Set, Tuple

import pysrt
from tqdm import tqdm


def dup_marker_single(
    subs: pysrt.SubRipFile,
    seen: Set[int],
    lookback: int = 4,
) -> Tuple[pysrt.SubRipFile, Set[int]]:
    duplicate = 0
    duplicate_candidate = 0
    subs = subs.copy()
    for s_i, sub in enumerate(subs):
        text_hash = hash(sub.text_without_tags)
        if text_hash not in seen:
            seen.add(text_hash)
            duplicate_candidate = 0
        else:
            duplicate_candidate += 1

        if duplicate_candidate >= lookback:
            duplicate = lookback
        else:
            duplicate -= 1

        if duplicate > 0:
            subs[s_i - lookback].text = "<duplicate>" + subs[s_i - lookback].text + "</duplicate>"
    # get the rest
    for s_i in range(lookback, 0, -1):
        if duplicate > 0:
            subs[-s_i].text = "<duplicate>" + subs[-s_i].text + "</duplicate>"
        duplicate -= 1
    return subs, seen


def mark_duplicates(in_data: str, lookback: int = 4) -> None:
    seen: Set[int] = set()
    file_names = glob.iglob(f"{in_data}/**/file.srt", recursive=True)

    for fn in tqdm(file_names):
        subs = pysrt.open(fn)
        subs, seen = dup_marker_single(subs, seen, lookback)
        subs.save(fn.split(".")[0] + ".dupes.srt")


if __name__ == "__main__":
    import sys

    start = time.time()
    mark_duplicates(sys.argv[1])
    print(f"took {time.time() - start:.2f} seconds")
