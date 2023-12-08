# %%

import pysrt
import os
from tqdm import tqdm
import glob
import sys
from src.utils import fuse_subtitles
import re
import shutil
import time
from datasets import load_from_disk

MAX_LIST_SIZE = 100_000
# gpt2 tokenzer eos-token
NEWLINE = "<|endoftext|>"


# %%
def read_subs_and_fuse(in_data, out_data):
    file_names = glob.glob(f"{in_data}/**/file.srt", recursive=True)

    for fn in tqdm(file_names):
        x = fn.split("/")[-2]
        fn_out = f"{out_data}/{x}/file.srt"
        os.makedirs(f"{out_data}/{x}/", exist_ok=True)
        fuse_subtitles(fn, fn_out)


# %%


# %%
import datasets


def save_subs_as_hf_dataset(in_data, out_data):
    print("Collecting filenames")
    # start = time.time()
    # file_names = glob.glob(f"{in_data}/**/file.srt", recursive=True)
    # print(f"Done collecting filenames. Took {time.time() - start:.2f} seconds")

    def my_gen():
        file_names = glob.iglob(f"{in_data}/**/file.srt", recursive=True)
        for fn in tqdm(file_names):
            subs = pysrt.open(fn)
            doc = []
            for sub in subs:
                line = sub.text_without_tags.replace("\n", NEWLINE)
                doc.append(line)
            yield {"text": "\n".join(doc), "origin": fn}

    dataset = datasets.Dataset.from_generator(my_gen)
    dataset.save_to_disk(out_data)


# %%
def save_subs_as_txt(in_data, out_data):
    try:
        os.makedirs(out_data, exist_ok=False)
    except FileExistsError:
        remove_dir = (
            input("Directory exists. Do you want to remove it and continue?\ny/n\n")
            == "y"
        )
        if remove_dir:
            shutil.rmtree(out_data)
            os.makedirs(out_data, exist_ok=False)
        else:
            print("Aborting without removing the previous data")
            return
    fn_out = f"{out_data}/subs.txt"
    fn_index = f"{out_data}/index.txt"
    file_names = glob.iglob(f"{in_data}/**/file.srt", recursive=True)
    for fn in tqdm(file_names):
        data = []
        index = []
        subs = pysrt.open(fn)
        for sub in subs:
            data.append(sub.text_without_tags.replace("\n", NEWLINE))
            index.append((fn, sub.index))
        with open(fn_out, "a") as fout:
            for d in data:
                print(d, file=fout)
        with open(fn_index, "a") as fout:
            for i in index:
                print(i, file=fout)


# %%
def my_iter(some_iterable):
    for item in some_iterable:
        yield item


# %%
def mark_duplicates(dup_fn, index_fn, data_fn):
    with open(dup_fn, encoding="utf-8", errors="replace") as dup_candidates, open(
        index_fn
    ) as index:
        duplicate = False
        subs = None
        fout = None
        prev_sub_fn = None
        for dc, id in tqdm(zip(dup_candidates, index)):
            # dc = next(dup_candidates)
            # id = next(index)
            # while dc and id:
            sub_fn, sub_id = re.sub(r"[\s\(\)\']", "", id).split(",")
            sub_id = int(sub_id)
            if sub_fn != prev_sub_fn:
                if fout:
                    fout.save()
                prev_sub_fn = sub_fn
                subs = my_iter(pysrt.open(sub_fn))
                program_id = sub_fn.split("/")[-2]
                os.makedirs(f"{data_fn}/{program_id}", exist_ok=True)
                fout = pysrt.SubRipFile(path=f"{data_fn}/{program_id}/file.srt")

            sub_item = next(subs)
            assert sub_item.index == sub_id

            if "<duplicate>" in dc:
                duplicate = True

            if duplicate:
                sub_item.text = "<duplicate>" + sub_item.text + "</duplicate>"

            fout.append(sub_item)

            if "</duplicate>" in dc:
                duplicate = False

            # try:
            #     dc = next(dup_candidates)
            #     id = next(index)
            # except UnicodeDecodeError:
            #     print(dc, id)
        if fout:
            fout.save()


def mark_duplicates_hf(dup_hf_fn, output_dir):
    dups = load_from_disk(dup_hf_fn)
    for item in tqdm(dups):
        srt_fn = item["origin"]
        subs = pysrt.open(srt_fn)
        folder_path = "/".join(srt_fn.split("/")[-7:-1])
        os.makedirs(f"{output_dir}/{folder_path}", exist_ok=True)
        fout = pysrt.SubRipFile(path=f"{output_dir}/{folder_path}/file.srt")
        duplicate = False
        for sub_item, dup_sub in zip(subs, item["text"].split("\n")):
            if "<duplicate>" in dup_sub:
                duplicate = True

            if duplicate:
                sub_item.text = "<duplicate>" + sub_item.text + "</duplicate>"

            fout.append(sub_item)

            if "</duplicate>" in dup_sub:
                duplicate = False
        if fout:
            fout.save()


# %%
# save_subs_as_txt("/home/robkur/data/delat/undertexter/", "hej")
if __name__ == "__main__":
    # save_subs_as_txt("/home/robkur/data/delat/srt_only/", "srt_only")
    # save_subs_as_hf_dataset("/home/robkur/data/delat/srt_only/", "srt_only_hf")

    # save_subs_as_hf_dataset(
    #     "/home/robkur/data/delat/srt_only/tv4/tv4/2023/10", "test_hf"
    # )
    mark_duplicates_hf(
        "/home/robkur/workspace/text-dedup/output/suffix_array/test_data/",
        "test_deduped",
    )

    # mark_duplicates("/home/robkur/workspace/subtitles_preprocessing/hej/subs.dedup-marked", "/home/robkur/workspace/subtitles_preprocessing/hej/index.txt", "./tmp")
