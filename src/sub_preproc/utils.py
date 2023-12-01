import pysrt


def fuse_subtitles(fn_in: str, fn_out) -> None:
    subs = pysrt.open(fn_in)

    mysubs = []
    prev = None
    start = -1
    end = -1
    index = 0
    for sub in subs:
        if sub.text != prev:
            if prev is not None and end - start > 0:
                ns = pysrt.srtitem.SubRipItem(
                    start=start, end=end, text=prev, index=index
                )
                mysubs.append(ns)
            start = sub.start
            end = sub.end
            prev = sub.text
            index += 1
        elif sub.text == prev:
            end = sub.end
    if prev is not None and end - start > 0:
        ns = pysrt.srtitem.SubRipItem(start=start, end=end, text=prev)
        mysubs.append(ns)

    new_subs = pysrt.SubRipFile(mysubs)

    new_subs.save(fn_out, encoding="utf-8")
