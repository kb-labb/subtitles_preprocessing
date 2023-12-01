import pysrt
import os
from tqdm import tqdm
import glob
from src.sub_preproc.utils import fuse_subtitles
import re
import datasets

MAX_LIST_SIZE = 100_000
# gpt2 tokenzer eos-token
NEWLINE = "<|endoftext|>"


def read_subs_and_fuse(in_data, out_data):
    file_names = glob.glob(f"{in_data}/**/file.srt", recursive=True)

    for fn in tqdm(file_names):
        x = fn.split("/")[-2]
        fn_out = f"{out_data}/{x}/file.srt"
        os.makedirs(f"{out_data}/{x}/", exist_ok=True)
        fuse_subtitles(fn, fn_out)


def save_subs_as_hf_dataset(in_data, out_data):
    file_names = glob.glob(f"{in_data}/**/file.srt", recursive=True)

    def my_gen():
        for fn in tqdm(file_names):
            subs = pysrt.open(fn)
            doc = []
            for sub in subs:
                line = sub.text_without_tags.replace("\n", NEWLINE)
                doc.append(line)
            yield {"text": "\n".join(doc)}

    dataset = datasets.Dataset.from_generator(my_gen)
    dataset.save_to_disk(out_data)


def save_subs_as_txt(in_data, out_data):
    data = []
    index = []
    os.makedirs(out_data, exist_ok=True)
    fn_out = f"{out_data}/subs.txt"
    fn_index = f"{out_data}/index.txt"
    file_names = glob.glob(f"{in_data}/**/file.srt", recursive=True)
    for fn in tqdm(file_names):
        subs = pysrt.open(fn)
        for sub in subs:
            data.append(sub.text_without_tags.replace("\n", NEWLINE))
            index.append((fn, sub.index))
        if len(data) > MAX_LIST_SIZE:
            with open(fn_out, "a") as fout:
                for d in data:
                    print(d, file=fout)
            with open(fn_index, "a") as fout:
                for i in index:
                    print(i, file=fout)
            data = []
            index = []


def my_iter(some_iterable):
    for item in some_iterable:
        yield item


def mark_duplicates(dup_fn, index_fn, data_fn):
    with open(dup_fn, encoding="utf-8", errors="replace") as dup_candidates, open(
        index_fn
    ) as index:
        duplicate = False
        subs = None
        fout = None
        prev_sub_fn = None
        for dc, id in zip(dup_candidates, index):
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


if __name__ == "__main__":
    save_subs_as_txt("/home/robkur/data/delat/undertexter/", "hej")
    # mark_duplicates("/home/robkur/workspace/subtitles_preprocessing/subs.dedup-marked", "/home/robkur/workspace/subtitles_preprocessing/index.txt", "./tmp")
