import pysrt


# Christopher P. Matthews
# christophermatthews1985@gmail.com
# Sacramento, CA, USA


def levenshtein(source, target):
    """From Wikipedia article; Iterative with two matrix rows."""
    if source == target:
        return 0
    elif len(source) == 0:
        return len(target)
    elif len(target) == 0:
        return len(source)
    v0 = [None] * (len(target) + 1)
    v1 = [None] * (len(target) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(source)):
        v1[0] = i + 1
        for j in range(len(target)):
            cost = 0 if source[i] == target[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]

    return v1[len(target)]


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


def fuse_live_subs(fn_in: str, fn_out: str) -> None:
    pass


def mark_live_subs(subs: pysrt.SubRipFile) -> None:
    longest = ""
    start = 0
    prev = None
    new_subs = pysrt.SubRipFile()
    for si, sub in enumerate(subs):
        if si >= 0:
            prev = subs[si - 1]
        # print(
        #     prev.text_without_tags.replace("\n", " "),
        #     "####",
        #     sub.text_without_tags.replace("\n", " "),
        # )
        if prev is None:
            longest = sub.text_without_tags
        elif sub.text_without_tags.startswith(prev.text_without_tags):
            longest = sub.text_without_tags
        elif sub.text_without_tags.startswith(prev.text_without_tags[:10]):
            # print(
            #     "##close enough##",
            #     sub.text_without_tags.replace("\n", " "),
            #     prev.text_without_tags.replace("\n", " "),
            #     levenshtein(
            #         sub.text_without_tags.replace("\n", " "),
            #         prev.text_without_tags.replace("\n", " "),
            #     ),
            # )
            longest = sub.text_without_tags
        else:
            if si - start <= 1:
                # print(
                #     prev.text_without_tags.replace("\n", " "),
                #     si - start,
                #     prev.start,
                #     prev.end,
                # )
                prev.index = len(new_subs)
                new_subs.append(prev)
            else:
                # print(
                #     "###",
                #     prev.text_without_tags.replace("\n", " "),
                #     si - start,
                #     start_time,
                #     prev.end,
                # )
                prev.text = "<live_sub>" + prev.text + "</live_sub>"
                prev.start = start_time
                prev.index = len(new_subs)
                new_subs.append(prev)
            # print("###", longest.replace("\n", " "), si - start)
            longest = sub.text_without_tags
            start = si
            start_time = sub.start
    # print("###", longest.replace("\n", " "), si - start)
    prev = subs[-1]
    if si - start <= 1:
        # print(
        #     prev.text_without_tags.replace("\n", " "),
        #     si - start,
        #     prev.start,
        #     prev.end,
        # )
        prev.index = len(new_subs)
        new_subs.append(prev)
    else:
        # print(
        #     "###",
        #     prev.text_without_tags.replace("\n", " "),
        #     si - start,
        #     start_time,
        #     prev.end,
        # )
        prev.text = "<live_sub>" + prev.text + "</live_sub>"
        prev.start = start_time
        prev.index = len(new_subs)
        new_subs.append(prev)
    return new_subs


if __name__ == "__main__":
    fn = "/home/robkur/workspace/subtitles_preprocessing/srt_only_dedup/XA/tv4/tv4/2022/12/15/XA_tv4_tv4_2022-12-15_090000_100000/file.srt"

    subs = pysrt.open(fn)
    new_subs = mark_live_subs(subs)
    for s in new_subs:
        print(s.index, s.text.replace("\n", " "))
