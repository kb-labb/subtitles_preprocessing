import json
import os
import pysrt
import pandas as pd
import datetime
import glob
from tqdm import tqdm
from collections import deque

HOUR = 3_600_000
MINUTE = 60_000
SECOND = 1_000


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def fuse_subtitles(fn_in: str, fn_out) -> None:
    subs = pysrt.open(fn_in)

    mysubs = []
    prev = None
    start = -1
    end = -1
    index = 0
    for s in subs:
        if s.text != prev:
            if prev is not None and end - start > 0:
                ns = pysrt.srtitem.SubRipItem(start=start, end=end, text=prev, index=index)
                mysubs.append(ns)
            start = s.start
            end = s.end
            prev = s.text
            index += 1
        elif s.text == prev:
            end = s.end
    if prev is not None and end - start > 0:
        ns = pysrt.srtitem.SubRipItem(start=start, end=end, text=prev)
        mysubs.append(ns)

    new_subs = pysrt.SubRipFile(mysubs)

    new_subs.save(fn_out, encoding="utf-8")


def mark_live_subs(subs: pysrt.SubRipFile, debug: bool = False) -> pysrt.SubRipFile:
    new_subs = pysrt.SubRipFile()
    candidates = []
    n = 3
    context: deque = deque([], maxlen=2 * n + 1)

    if len(subs) <= 2 * n:
        return subs

    def add_livesub(sub):
        sub.text = "<live_sub>" + sub.text + "</live_sub>"
        sub.index = len(new_subs)
        new_subs.append(sub)
        candidates.append(True)

    def add_normal(sub):
        sub.index = len(new_subs)
        new_subs.append(sub)
        candidates.append(False)

    def is_livesub(sub, prev):
        if prev is None:
            return False
        if sub.text_without_tags.startswith(prev.text_without_tags):
            sub.text = "<live_candidate>" + sub.text + "</live_candidate>"
            # prev.text = "<live_candidate>" + prev.text
            return True
        else:
            return False

    def is_really_livesub(context, debug=False):
        # n_ = n if len(context) == 2 * n + 1 else 2 * n + 1 - len(context)
        try:
            if not context[n] and context[n + 1]:
                context[n] = True
        except IndexError:
            pass

        lc = tuple(context)
        # prev_sum = sum(context[i] for i in range(n))
        # foll_sum = sum(context[-i] for i in range(n, 0, -1))
        prev_sum = sum(lc[:n])
        foll_sum = sum(lc[n + 1 :])

        if debug:
            print(n, lc, prev_sum, foll_sum)

        # if foll_sum >= n_ - 1 and lc[n_]:
        #     return True
        # if prev_sum >= n_ - 1 and lc[n_]:
        #     return True
        # if foll_sum >= n_:
        #     return True
        if prev_sum == n and not lc[n] and foll_sum == 0:
            return False

        return prev_sum + lc[n] + foll_sum >= n

    for j in range(0, len(subs)):
        if len(context) > n:
            really_livesub = is_really_livesub(context)
            sub_i = subs[j - n - 1]
            # if j - n - 1 > 400:
            #     print(
            #         j - n - 1,
            #         sub_i.text_without_tags.replace("\n", " "),
            #         context,
            #         sum(context),
            #         really_livesub,
            #     )
            #     is_really_livesub(context, True)

            if really_livesub:
                add_livesub(sub_i)
            else:
                add_normal(sub_i)
        sub = subs[j]
        prev = subs[j - 1] if j > 0 else None
        curr = is_livesub(sub, prev)
        context.append(curr)

    # get the last n-1
    for j in range(len(subs) - n - 1, len(subs)):
        # really_livesub = context[n] + sum(context) > n
        really_livesub = is_really_livesub(context)
        sub_i = subs[j]
        # print(j, sub_i.text_without_tags.replace("\n", " "), really_livesub)
        # is_really_livesub(context, True)
        if really_livesub:
            add_livesub(sub_i)
        else:
            add_normal(sub_i)
        context.popleft()

    if debug:
        for i in range(0, len(new_subs)):
            print(
                i,
                # subs[i].text_without_tags.replace("\n", " "),
                new_subs[i].text_without_tags.replace("\n", " "),
                "live_sub" in new_subs[i].text,
                "live_candidate" in new_subs[i].text,
            )
    return new_subs


def mark_live_subs_folder(in_data: str, out_data: str) -> None:
    file_names = glob.iglob(f"{in_data}/**/file.srt", recursive=True)
    for fn in tqdm(file_names):
        program_id = fn.split("/")[-2]
        xa, channel_1, channel_2, date = program_id.split("_")[:4]
        year, month, day = date.split("-")
        out_path = "/".join([xa, channel_1, channel_2, year, month, day])
        os.makedirs(f"{out_data}/{out_path}/{program_id}", exist_ok=True)

        new_subs = mark_live_subs(pysrt.open(fn))
        new_subs.save(path=f"{out_data}/{out_path}/{program_id}/file.srt")


def get_stats_single(fn, df=None) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame(
            columns=[
                "channel",
                "subchannel",
                "broadcast_date",
                "from_time",
                "to_time",
                "subs_duration",
                "subs_duration_duplicate",
                "subs_duration_livesub",
                "subs_count",
                "duplicates_count",
                "livesubs_count",
            ]
        )
    _, channel, subchannel, broadcast_date, from_time, to_time = fn.split("/")[-2].split("_")
    from_time = from_time[:6]
    to_time = to_time[:6]
    broadcast_date = datetime.date.fromisoformat(broadcast_date)
    from_time = datetime.time.fromisoformat(
        ":".join([from_time[i - 1 : i + 1] for i in range(1, len(from_time), 2)])
    )
    to_time = datetime.time.fromisoformat(
        ":".join([to_time[i - 1 : i + 1] for i in range(1, len(to_time), 2)])
    )
    subs_duration = 0
    subs_duration_duplicate = 0
    subs_duration_livesub = 0
    subs_count = 0
    duplicates_count = 0
    livesubs_count = 0
    for sub in pysrt.open(fn):
        subs_count += 1
        subs_duration += srt_time_to_ms(sub.duration)
        # if sub.text.startswith("<duplicate>"):
        if "<duplicate>" in sub.text:
            duplicates_count += 1
            subs_duration_duplicate += srt_time_to_ms(sub.duration)
        elif sub.text.startswith("<live_sub>"):
            livesubs_count += 1
            subs_duration_livesub += srt_time_to_ms(sub.duration)

    df.loc[len(df)] = {
        "channel": channel,
        "subchannel": subchannel,
        "broadcast_date": broadcast_date,
        "from_time": from_time,
        "to_time": to_time,
        "subs_duration": ms_to_time(subs_duration),
        "subs_duration_duplicate": ms_to_time(subs_duration_duplicate),
        "subs_duration_livesub": ms_to_time(subs_duration_livesub),
        "subs_count": subs_count,
        "duplicates_count": duplicates_count,
        "livesubs_count": livesubs_count,
    }
    return df


def get_stats_folder(in_data: str) -> pd.DataFrame:
    file_names = glob.iglob(f"{in_data}/**/file.srt", recursive=True)
    df = None
    for fn in tqdm(file_names):
        df = get_stats_single(fn, df)
    return df


def srt_time_to_ms(time: pysrt.SubRipTime) -> int:
    return time.hours * HOUR + time.minutes * MINUTE + time.seconds * SECOND + time.milliseconds


def dt_time_to_ms(time: datetime.time) -> int:
    return (
        time.hour * HOUR + time.minute * MINUTE + time.second * SECOND + time.microsecond // SECOND
    )


def ms_to_time(ms: int) -> datetime.time:
    hours = ms // HOUR
    ms -= hours * HOUR
    minutes = ms // MINUTE
    ms -= minutes * MINUTE
    seconds = ms // SECOND
    ms -= seconds * SECOND
    microseconds = ms * 1_000
    return datetime.time(hours, minutes, seconds, microseconds)


def is_duplicate(text):
    return "<duplicate>" in text


def is_live(text):
    return "<live_sub>" in text


def srt_to_json(fn_in: str, fn_out, data_source="tv_smbd") -> None:
    subs = pysrt.open(fn_in)
    # XA_cmore_cmoreseries_2023-03-01_100000_110000
    _, channel, subchannel, year_month_day, from_time, to_time = fn_in.split("/")[-2].split("_")
    year, month, day = [int(x) for x in year_month_day.split("-")]
    from_time = int(from_time)  # datetime.time.isoformat(from_time)
    to_time = int(to_time)  # datetime.time.isoformat(to_time)

    subs_dict = {
        "metadata": {
            "channel": channel,
            "subchannel": subchannel,
            "year": year,
            "month": month,
            "day": day,
            "from_time": from_time,
            "to_time": to_time,
            "data_source": data_source,
        },
        "subs": [],
    }

    for sub in subs:
        subs_dict["subs"].append(
            {
                "start": srt_time_to_ms(sub.start),
                "end": srt_time_to_ms(sub.end),
                "duration": srt_time_to_ms(sub.end) - srt_time_to_ms(sub.start),
                "text": sub.text_without_tags,
                "duplicate": is_duplicate(sub.text),
                "live": is_live(sub.text),
            }
        )

    with open(fn_out, "w") as fout:
        json.dump(subs_dict, fout, indent=4)


if __name__ == "__main__":
    # fn = "/home/robkur/workspace/subtitles_preprocessing/srt_only_dedup/XA/tv4/tv4/2022/12/15/XA_tv4_tv4_2022-12-15_090000_100000/file.srt"
    # folder_name = "/home/robkur/workspace/subtitles_preprocessing/srt_only_dedup/XA/tv4/tv4/2022/12/"

    # subs = pysrt.open(fn)
    # new_subs = mark_live_subs(subs)
    # mytime = 0
    # for s in new_subs:
    #     mytime += srt_time_to_ms(s.duration)
    #     print(s.index, s.text.replace("\n", " "))
    # print(mytime)
    # print(ms_to_time(mytime))
    # print(get_stats_folder(folder_name))

    import sys

    folder_name = sys.argv[1]
    stats = get_stats_folder(folder_name)
    stats.to_csv("mystats.csv")
    print(stats)
    # mark_live_subs_folder(folder_name, sys.argv[2])

    # fn = sys.argv[1]
    # subs = pysrt.open(fn)
    # mark_live_subs(subs, True)
