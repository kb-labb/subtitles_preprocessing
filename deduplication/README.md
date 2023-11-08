# Deduplication

TV-Programs get repeated many times.

## Solution:

Convert subtitles into text-documents to be deduplicated with
<https://github.com/google-research/deduplicate-text-datasets>.
The library does exact deduplication and can be used to mark duplicates which then have
to be removed from the one-hour chunks (both audio and subtitles).

__WIP__
