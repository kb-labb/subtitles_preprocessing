from os import makedirs
import glob
import pysrt
from sub_preproc.utils.utils import srt_to_json

subs_dict = srt_to_json(
    "/home/fatrek/data_network/delat/srt_only/cmore/sfkanalen/2023/03/03/XA_cmore_sfkanalen_2023-03-03_160000_170000/file.srt",
    fn_out="s.json",
)


def make_chunks(subs, min_threshold=10_000, max_threshold=30_000):
    chunks = []
    chunk = []
    total_length = 0
    chunk_start = 0
    chunk_end = 0

    def subs_to_whisper(subs):
        def whisper_time(ms):
            ms = ms // 20 * 20
            ms /= 1_000
            return f"<{ms:.2f}>"

        return "".join(
            "".join((whisper_time(sub["start"]), sub["text"], whisper_time(sub["end"])))
            for sub in subs
            if sub["text"] != "<|silence|>"
        )

    for sub in subs["subs"][1:-1]:
        sub = sub.copy()
        # add to chunk if total chunk length < 30s
        if sub["live"] or sub["duplicate"]:
            if total_length >= min_threshold:
                chunks.append(
                    {
                        "start": chunk_start,
                        "end": chunk_end,
                        "duration": chunk_end - chunk_start,
                        "subs": chunk,
                    }
                )
                chunk = []
                total_length = 0

            else:
                chunk = []
                total_length = 0
        else:
            if sub["duration"] + total_length > max_threshold:
                if sub["text"] == "<|silence|>":
                    silent_sub = sub.copy()
                    silent_sub["duration"] = max_threshold - total_length
                    silent_sub["start"] = total_length
                    silent_sub["end"] = max_threshold
                    chunk_end = sub["start"] + silent_sub["duration"]
                    chunk.append(silent_sub)
                    chunks.append(
                        {
                            "start": chunk_start,
                            "end": chunk_end,
                            "duration": chunk_end - chunk_start,
                            "subs": chunk,
                            "text_whisper": subs_to_whisper(chunk),
                        }
                    )
                    chunk = []
                    total_length = 0
                    sub["start"] = sub["start"] + silent_sub["duration"]
                    sub["duration"] -= silent_sub["duration"]

                    while sub["duration"] > max_threshold:
                        chunk_start = sub["start"]
                        chunk_end = sub["start"] + max_threshold
                        chunks.append(
                            {
                                "start": chunk_start,
                                "end": chunk_end,
                                "duration": chunk_end - chunk_start,
                                "subs": [
                                    {
                                        "text": "<|silence|>",
                                        "start": 0,
                                        "end": max_threshold,
                                        "duration": max_threshold,
                                    }
                                ],
                                "text_whisper": subs_to_whisper(chunk),
                            }
                        )
                        # chunk = []
                        # total_length = 0
                        sub["start"] += max_threshold
                        sub["duration"] -= max_threshold
                else:
                    chunks.append(
                        {
                            "start": chunk_start,
                            "end": chunk_end,
                            "duration": chunk_end - chunk_start,
                            "subs": chunk,
                            "text_whisper": subs_to_whisper(chunk),
                        }
                    )
                    chunk = []
                    total_length = 0

            if len(chunk) == 0:
                print("chunk_start before", chunk_start)
                if chunks:
                    print(chunks[-1])
                chunk_start = sub["start"]
                print("chunk_start", chunk_start)
                print(sub)
                print("#" * 30)

            chunk_end = sub["end"]
            sub["start"] = total_length
            sub["end"] = total_length + sub["duration"]
            chunk.append(sub)
            total_length += sub["duration"]

    if total_length >= min_threshold:
        chunks.append(
            {
                "start": chunk_start,
                "end": chunk_end,
                "duration": chunk_end - chunk_start,
                "subs": chunk,
                "text_whisper": subs_to_whisper(chunk),
            }
        )

    subs["chunks"] = chunks
    return subs


if __name__ == "__main__":
    chunks = make_chunks(subs_dict)

    chunks["chunks"][2]

    import json

    json.dump(chunks, open("chunks.json", "w"), indent=4)
