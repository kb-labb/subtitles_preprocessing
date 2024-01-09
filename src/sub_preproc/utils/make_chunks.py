def make_chunks(subs, min_threshold=10_000, max_threshold=30_000, start_index=1, end_index=1):
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

        parts = []
        for sub in subs:
            if sub["text"] != "<|silence|>":
                part = "".join((whisper_time(sub["start"]), sub["text"], whisper_time(sub["end"])))
                parts.append(part)
        return "".join(parts)

    for sub in subs["subs"][start_index:-end_index]:
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
            # else: we throw away the chunk
            chunk = []
            total_length = 0
        else:
            if sub["duration"] + total_length > max_threshold:
                if sub["text"] == "<|silence|>":
                    filler_silence = sub.copy()
                    filler_silence["duration"] = max_threshold - total_length
                    filler_silence["start"] = total_length
                    filler_silence["end"] = max_threshold
                    chunk_end = sub["start"] + filler_silence["duration"]
                    chunk.append(filler_silence)
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
                    sub["start"] = sub["start"] + filler_silence["duration"]
                    sub["duration"] -= filler_silence["duration"]

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
                chunk_start = sub["start"]

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