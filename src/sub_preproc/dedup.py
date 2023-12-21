# %%

import pysrt
import os
from tqdm import tqdm
import glob
import sys
from src.utils import utils
import re
import shutil
import time
import datasets
from datasets import load_from_disk
import multiprocessing as mp
from functools import partial
from collections import deque

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
        utils.fuse_subtitles(fn, fn_out)


# %%


# %%


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
def read_write_subs(fn, fn_out):
    data = []
    index = []
    subs = pysrt.open(fn)
    # suffix = fn.split("/")[-2]
    # with open(f"{fn_out}/subs.{suffix}", "w") as fout, open(
    #     f"{fn_out}/index.{suffix}", "w"
    # ) as findex:
    if True:
        for sub in subs:
            data.append(sub.text_without_tags.replace("\n", NEWLINE))
            index.append((fn, sub.index))
            # print(sub.text_without_tags.replace("\n", NEWLINE), file=fout)
            # print((fn, sub.index), file=findex)
    return data, index


def save_subs_as_txt(in_data, out_data, n_processes):
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
    with open(fn_out, "w") as fout, open(fn_index, "w") as findex:
        my_fun = partial(read_write_subs, fn_out=out_data)
        with mp.Pool(processes=n_processes) as pool:
            results = pool.imap(my_fun, tqdm(file_names))
            # for _ in results:
            #     pass
            # results = tqdm(pool.map(read_write_subs, (file_names)))
            for data, index in results:
                for d in data:
                    print(d, file=fout)
                for i in index:
                    print(i, file=findex)


def stupid_dedup_deque(in_data, out_data):
    dups = 0
    total = 0
    seen = set()
    file_names = glob.iglob(f"{in_data}/**/file.srt", recursive=True)

    n = 4
    for fn in file_names:
        fn_out = "tmp.srt"
        utils.fuse_subtitles(fn, fn_out)
        subs = pysrt.open(fn_out)

        prev_candidate = deque([False] * n, maxlen=n)
        prev_items = deque([], maxlen=n)

        program_id = fn.split("/")[-2]
        xa, channel_1, channel_2, date = program_id.split("_")[:4]
        year, month, day = date.split("-")
        out_path = "/".join([xa, channel_1, channel_2, year, month, day])
        os.makedirs(f"{out_data}/{out_path}/{program_id}", exist_ok=True)
        fout = pysrt.SubRipFile(path=f"{out_data}/{out_path}/{program_id}/file.srt")

        duplicate = 0
        for sub in subs:
            prev_items.append(sub)
            text = hash(sub.text_without_tags)
            if text not in seen:
                seen.add(text)
                prev_candidate.append(False)
            else:
                prev_candidate.append(True)

            if all(prev_candidate):
                duplicate = n
            else:
                duplicate = max((0, duplicate - 1))

            if duplicate > 0:
                dups += 1
                prev_items[0].text = "<duplicate>" + prev_items[0].text + "</duplicate>"
            fout.append(prev_items[0])
            total += 1
            # print(f"{dups} / {dups} ≃ {dups/total:.2%}", end="\r")
        # get the rest out of the deque
        while prev_items:
            if duplicate > 0:
                dups += 1
                prev_items[0].text = "<duplicate>" + prev_items[0].text + "</duplicate>"
            duplicate = max((0, duplicate - 1))
            fout.append(prev_items.popleft())

        fout.save()
        try:
            print(f"{dups} / {dups} ≃ {dups/total:.2%}", end="\r")
        except ZeroDivisionError:
            print("empty")
    try:
        print("Finally...")
        print(f"{dups} / {dups} ≃ {dups/total:.2%}", end="\n")
    except ZeroDivisionError:
        print("empty")


def stupid_dedup_counting(in_data, out_data):
    dups = 0
    total = 0
    seen = set()
    file_names = glob.iglob(f"{in_data}/**/file.srt", recursive=True)

    n = 4

    for f_i, fn in enumerate(file_names):
        fn_out = "tmp.srt"
        utils.fuse_subtitles(fn, fn_out)
        subs = pysrt.open(fn_out)

        program_id = fn.split("/")[-2]
        xa, channel_1, channel_2, date = program_id.split("_")[:4]
        year, month, day = date.split("-")
        out_path = "/".join([xa, channel_1, channel_2, year, month, day])
        os.makedirs(f"{out_data}/{out_path}/{program_id}", exist_ok=True)
        fout = pysrt.SubRipFile(path=f"{out_data}/{out_path}/{program_id}/file.srt")

        duplicate = 0
        duplicate_candidate = 0
        for s_i, sub in enumerate(subs):
            text_hash = hash(sub.text_without_tags)
            if text_hash not in seen:
                seen.add(text_hash)
                duplicate_candidate = 0
            else:
                duplicate_candidate += 1

            if duplicate_candidate >= n:
                duplicate = n
            else:
                duplicate -= 1

            if duplicate > 0:
                dups += 1
                subs[s_i - n].text = "<duplicate>" + subs[s_i - n].text + "</duplicate>"
            if s_i - n >= 0:
                fout.append(subs[s_i - n])
            total += 1
        # get the rest
        for s_i in range(n, 0, -1):
            if duplicate > 0:
                dups += 1
                subs[-s_i].text = "<duplicate>" + subs[-s_i].text + "</duplicate>"
            duplicate -= 1
            if len(subs) > s_i:
                fout.append(subs[-s_i])

        fout.save()
        if f_i % 100 == 0:
            try:
                print(
                    f"files done: {f_i:_}; {dups} / {total} ≃ {dups/total:.2%}",
                    end="\r",
                )
            except ZeroDivisionError:
                print("empty")
    try:
        print("\nFinally...")
        print(f"files done: {f_i:_}; {dups} / {total} ≃ {dups/total:.2%}", end="\n")
    except ZeroDivisionError:
        print("empty")


def count_stuff(in_data):
    file_names = glob.iglob(f"{in_data}/**/file.srt", recursive=True)

    counts = {}
    for f_i, fn in enumerate(file_names):
        subs = pysrt.open(fn)
        channel = "/".join(fn.split("/")[2:4])
        if channel not in counts:
            counts[channel] = {"total": 0, "dups": 0, "n-files": 0}
        total = 0
        dups = 0
        for sub in subs:
            if sub.text.startswith("<duplicate"):
                dups += 1
            total += 1
        counts[channel]["total"] += total
        counts[channel]["dups"] += dups
        counts[channel]["n-files"] += 1
        if f_i % 1_000 == 0:
            for channel in counts:
                dups = counts[channel]["dups"]
                total = counts[channel]["total"]
                n_files = counts[channel]["n-files"]
                try:
                    print(
                        f"{channel:<20} {dups:_}, {total:_}, {dups / total:.2%}, {n_files:_}"
                    )
                except ZeroDivisionError:
                    print(f"{channel:<20} {dups:_}, {total:_}, -, {n_files:_}")
    print("Finally....")
    for channel in counts:
        dups = counts[channel]["dups"]
        total = counts[channel]["total"]
        n_files = counts[channel]["n-files"]
        try:
            print(f"{channel:<20} {dups:_}, {total:_}, {dups / total:.2%}, {n_files:_}")
        except ZeroDivisionError:
            print(f"{channel:<20} {dups:_}, {total:_}, -, {n_files:_}")


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
    start = time.time()
    # stupid_dedup("/home/robkur/data/delat/srt_only/tv4/tv4/2023/10/", "test")
    # stupid_dedup_counting("/home/robkur/data/delat/srt_only/", "srt_only_dedup")
    count_stuff("srt_only_dedup")
    # save_subs_as_txt("/home/robkur/data/delat/srt_only/", "srt_only", mp.cpu_count())
    # save_subs_as_txt(
    #     "/home/robkur/data/delat/srt_only/tv4/tv4/2023/10/", "test", mp.cpu_count()
    # )
    # save_subs_as_hf_dataset("/home/robkur/data/delat/srt_only/", "srt_only_hf")

    # save_subs_as_hf_dataset(
    #     "/home/robkur/data/delat/srt_only/tv4/tv4/2023/10", "test_hf"
    # )
    # mark_duplicates_hf(
    #     "/home/robkur/workspace/text-dedup/output/suffix_array/test_data/",
    #     "test_deduped",
    # )

    # mark_duplicates("/home/robkur/workspace/subtitles_preprocessing/hej/subs.dedup-marked", "/home/robkur/workspace/subtitles_preprocessing/hej/index.txt", "./tmp")
    print(f"took {time.time() - start:.2f} seconds")
