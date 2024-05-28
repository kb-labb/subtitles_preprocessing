from tqdm import tqdm
import json
import time
import multiprocessing as mp

HOUR = 1_000 * 60 * 60


def worker_fun(fn_j):
    try:
        with open(fn_j) as fh_j:
            d = json.load(fh_j)
            n_subs = len([1 for x in d["subs"][1:-1] if not x["duplicate"] and not x["live"] and x["id"] > 0])
            sv_subs = set()
            sv_sub_length = 0
            n_chunks = 0
            chunk_length = 0
            for chunk in d["chunks"]:
                if chunk["all_subs_swedish"]:
                    n_chunks += 1
                    chunk_length += chunk["duration"]
                    for ssi in chunk["sub_ids"]:
                        if ssi >= 0:
                            sv_subs.add(ssi)
            for ssi in sv_subs:
                assert d["subs"][1:-1][ssi]["id"] == ssi
                sv_sub_length += d["subs"][1:-1][ssi]["duration"]
        return len(sv_subs), sv_sub_length, n_chunks, chunk_length, fn_j
    except:
        return 0, 0, 0, 0, fn_j

def main(fn):
    with open(fn) as fh:
        total = 0
        files = []
        for line in fh:
            total += 1
            files.append(line.strip() + ".sv-marked.json")
    print(f"Going to process {total} files...")
    total_sv_subs = 0
    total_sv_sub_length = 0
    total_n_chunks = 0
    total_chunk_length = 0
    sv_files = []
    start = time.time()
    with mp.Pool(processes=30) as pool:
    # if True:
        for (n_sv_subs, sv_sub_length, n_chunks, chunk_length, fn) in pool.imap(worker_fun, tqdm(files), chunksize=100):
        # for (n_sv_subs, sv_sub_length, n_chunks, chunk_length, fn) in map(worker_fun, files):
            total_sv_subs += n_sv_subs
            total_sv_sub_length += sv_sub_length
            total_n_chunks += n_chunks
            total_chunk_length += chunk_length
            # print(f"n_chunks: {total_n_chunks} chunk_length: {total_chunk_length/HOUR:.2f} total_sub_length: {total_sv_sub_length/HOUR:.2f}", end="\r")
    print(f"n_chunks: {total_n_chunks} chunk_length: {total_chunk_length/HOUR:.2f} total_sub_length: {total_sv_sub_length/HOUR:.2f}", end="\n")
    return total_sv_subs, total_sv_sub_length, total_n_chunks, total_chunk_length


if __name__ == "__main__":
    import sys
    fn = sys.argv[1]
    total_sv_subs, total_sv_sub_length, total_n_chunks, total_chunk_length = main(fn)
    # print(f"n_chunks: {total_n_chunks} chunk_length: {total_chunk_length} total_sub_length: {total_sv_sub_length}", end="\n")

