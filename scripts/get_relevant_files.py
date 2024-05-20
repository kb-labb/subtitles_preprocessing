from tqdm import tqdm
import json
import time
import multiprocessing as mp

HOUR = 1_000 * 60 * 60


def worker_fun(fn_j, mark_remaining_chunks=True):
    try:
        with open(fn_j) as fh_j:
            d = json.load(fh_j)
            n_subs = len([1 for x in d["subs"][1:-1] if not x["duplicate"] and not x["live"] and x["id"] > 0])
            sv_subs = set()
            sv_sub_length = 0
            for chunk in d["chunks"]:
                if chunk["duration"] > 20_000:
                    # if chunk["language_probs"]["openai/whisper-large-v3"]["sv"] >= p:
                    if max(chunk["language_probs"]["openai/whisper-large-v3"].items(), key=lambda x: x[1])[0] == "sv":
                        for si in chunk["sub_ids"]:
                            if si > 0:
                                sv_subs.add(si)
            # total_subs += n_subs
            # total_sv_subs += len(sv_subs)
            for ssi in sv_subs:
                sv_sub_length += d["subs"][1:-1][ssi]["duration"]
            # total_sv_sub_length += sv_sub_length
            # if len(sv_subs) > 0:
            #     sv_files.append(fn_j)
            # print(f"sv_chunks:{total_sv_subs / total_subs:.2%} sv_files:{len(sv_files)/(i+1):.2%} seen:{i/total:.2%} total_sub_length:{total_sv_sub_length/HOUR:.2f}", end="\r")
        if mark_remaining_chunks:
            for chunk in d["chunks"]:
                sv_ssis = [ssi in sv_subs for ssi in chunk["sub_ids"] if ssi > 0]
                if all(sv_ssis):
                    chunk["all_subs_swedish"] = True
                    chunk["any_subs_swedish"] = True
                elif any(sv_ssis):
                    chunk["all_subs_swedish"] = False
                    chunk["any_subs_swedish"] = True
                else:
                    chunk["all_subs_swedish"] = False
                    chunk["any_subs_swedish"] = False
            with open(fn_j + ".sv-marked.json", "w") as fh_j:
                json.dump(d, fh_j, indent=2)

        return n_subs, len(sv_subs), sv_sub_length, fn_j
    except:
        return 0, 0, 0, fn_j

def get_lang_probs_geq_p(fn, p):
    with open(fn) as fh:
        total = 0
        files = []
        for line in fh:
            total += 1
            files.append(line.strip())
    print(f"Going to process {total} files...")
    total_subs = 0
    total_sv_subs = 0
    total_sv_sub_length = 0
    sv_files = []
    start = time.time()
    with mp.Pool(processes=30) as pool:
    # if True:
        for (n_subs, n_sv_subs, sv_sub_length, fn) in pool.imap(worker_fun, tqdm(files), chunksize=100):
        # for (n_subs, n_sv_subs, sv_sub_length, fn) in map(worker_fun, files):
            total_subs += n_subs
            total_sv_subs += n_sv_subs
            total_sv_sub_length += sv_sub_length
            if n_sv_subs > 0:
                sv_files.append(fn)
            # print(f"sv_chunks:{total_sv_subs / total_subs:.2%} sv_files:{len(sv_files)/total:.2%} total_sub_length:{total_sv_sub_length/HOUR:.2f} time:{time.time() - start:.2f}", end="\n")
    print(f"sv_chunks:{total_sv_subs / total_subs:.2%} sv_files:{len(sv_files)/total:.2%} total_sub_length:{total_sv_sub_length/HOUR:.2f}", end="\n")
    return total_subs, total_sv_subs, sv_files, total


if __name__ == "__main__":
    import sys
    fn = sys.argv[1]
    p = float(sys.argv[2])
    total_subs, total_sv_subs, sv_files, total_files = get_lang_probs_geq_p(fn, p)
    # with open(sys.argv[1] + f".sv_files_p-{p}", "w") as fh:
    with open(sys.argv[1] + f".sv_files_sv-first", "w") as fh:
        for fn in sv_files:
            print(fn, file=fh)

