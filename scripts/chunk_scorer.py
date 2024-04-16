import json
import sys
import re

from rapidfuzz import fuzz
from sub_preproc.utils.text import normalize_text
from sub_preproc.utils.metrics import calculate_wer, calculate_bleu


def main():

    fn = sys.argv[1]
    with open(fn) as fh:
        d = json.load(fh)

    chunks = d["chunks"]

    for chunk in chunks:
        gt = chunk["text"]
        gt = normalize_text(gt)
        gt_words = gt.split()
        preds = chunk["transcription"]
        for pred in preds:
            model = pred["model"]
            pt = pred["text"]
            pt = normalize_text(pt)
            pt_words = pt.split()
            pred["scores"] = {
                "bleu": calculate_bleu(gt, pt),
                "wer": calculate_wer(gt, pt),
                "first": fuzz.ratio(gt_words[0], pt_words[0]),
                "last": fuzz.ratio(gt_words[-1], pt_words[-1]),
            }


"""
If all automatic transcriptions have low scores: throw away ground truth
If at least one model has good transcription: save ground truth
Throw away if bleu < x

If first and last have "> score", use these examples for training with timestamps

For wav2vec2 finetuning, the bleu_score should be close to 1. And first&last should match. 
"""
