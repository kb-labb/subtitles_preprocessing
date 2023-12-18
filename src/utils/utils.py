import os
import pysrt
import pandas as pd
import datetime
import glob
from tqdm import tqdm

HOUR = 3_600_000
MINUTE = 60_000
SECOND = 1_000


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
                ns = pysrt.srtitem.SubRipItem(
                    start=start, end=end, text=prev, index=index
                )
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


def mark_live_subs(subs: pysrt.SubRipFile) -> None:
    start = 0
    prev = None
    new_subs = pysrt.SubRipFile()

    if len(subs) == 0:
        return new_subs

    def add_livesub(prev, start_time):
        prev.text = "<live_sub>" + prev.text + "</live_sub>"
        prev.start = start_time
        prev.index = len(new_subs)
        new_subs.append(prev)

    def add_normal(prev):
        prev.index = len(new_subs)
        new_subs.append(prev)

    n = 3
    ls_candidate = 0
    ls = 0
    for si, sub in enumerate(subs):
        if si >= 0:
            prev = subs[si - 1]
        if prev is None:
            pass
        elif sub.text_without_tags.startswith(prev.text_without_tags):
            pass
        else:
            if si - start <= 1:
                add_normal(prev)
                ls_candidate = 0
            else:
                add_livesub(prev, start_time)
                ls_candidate += 1
            start = si
            start_time = sub.start
            if ls > 0:
                subs[si - n].text = "<duplicate>" + subs[si - n].text + "</duplicate>"
            if si - n >= 0:
                fout.append(subs[si - n])
    # add the last item as prev
    prev = subs[-1]
    si = len(subs)
    if si - start <= 1:
        add_normal(prev)
    else:
        add_livesub(prev, start_time)

    return new_subs


def mark_live_subs_folder(in_data: str, out_data: str) -> None:
    file_names = glob.iglob(f"{in_data}/**/file.srt", recursive=True)
    for fn in tqdm(file_names):
        program_id = fn.split("/")[-2]
        xa, channel_1, channel_2, date = program_id.split("_")[:4]
        year, month, day = date.split("-")
        out_path = "/".join([xa, channel_1, channel_2, year, month, day])
        os.makedirs(f"{out_data}/{out_path}/{program_id}", exist_ok=True)
        fout = pysrt.SubRipFile(path=f"{out_data}/{out_path}/{program_id}/file.srt")

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
    _, channel, subchannel, broadcast_date, from_time, to_time = fn.split("/")[
        -2
    ].split("_")
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
        if sub.text.startswith("<duplicate>"):
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
    return (
        time.hours * HOUR
        + time.minutes * MINUTE
        + time.seconds * SECOND
        + time.milliseconds
    )


def dt_time_to_ms(time: datetime.time) -> int:
    return (
        time.hour * HOUR
        + time.minute * MINUTE
        + time.second * SECOND
        + time.microsecond // SECOND
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

    folder_name = "/home/robkur/workspace/subtitles_preprocessing/srt_only_dedup/"
    mark_live_subs_folder(
        folder_name, "/home/robkur/workspace/subtitles_preprocessing/srt_only_livesubs/"
    )
