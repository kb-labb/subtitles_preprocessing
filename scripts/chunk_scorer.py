import json
import sys
import re
import argparse
from multiprocessing import Pool

from rapidfuzz import fuzz
from sub_preproc.utils.text import normalize_text
from sub_preproc.utils.metrics import calculate_wer, calculate_bleu


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--json_files",
    type=str,
    help="Path to file containg json filepaths",
    default="files.txt",
    )
    
    parser.add_argument(
    "--num_proc",
    type=int,
    help="Number of processes to use",
    default=16,
    )
    
    parser.add_argument(
    "--recalculate",
    action="store_true",
    help="Recalculate scores",
    )
    
    parser.add_argument(
    "--save_as_new",
    action="store_true",
    help="Save as new file",
    )
    
    return parser.parse_args()

def calculate_scores(file_to_process, recalculate=False, save_as_new=False):
    with open(file_to_process, "r") as fh:
        d = json.load(fh)
    for i, chunk in enumerate(d["chunks"]):      
          
        whisper_scores, wav2vec_scores = None, None
        
        if not recalculate and "filters" in chunk:
            # skip if already processed and not recalculating
            print(f"Skipping: {file_to_process}. Already processed. Use --recalculate to recalculate the scores.")
            break
        
        elif recalculate and "filters" not in chunk:
            # skip if not processed and recalculating
            print(f"Skipping: {file_to_process}, chunk: {i}. No scores to recalculate. Calculate scores first.")
            break  
        
        elif not recalculate and "filters" not in chunk or recalculate and "filters" in chunk:
            if recalculate and "filters" in chunk:
                # recalculate if already processed
                print(f"Recalculating: {file_to_process}, chunk: {i}")
            elif not recalculate and "filters" not in chunk:
                # calculate if not processed
                print(f"Calculaiting: {file_to_process}, chunk: {i}")
            gt = chunk["text"]
            gt = normalize_text(gt)
            gt_words = gt.split()
            preds = chunk["transcription"]
            chunk["filters"] = {"stage1_whisper": False, 
                                "stage2_whisper": False,
                                "stage2_whisper_timestamps": False, 
                                "stage1_wav2vec": False,
                                "silence": False}
            for pred in preds:
                if 'model' in pred:
                    model = pred["model"]
                    pt = pred["text"]
                    pt = normalize_text(pt)
                    pt_words = pt.split()
                    if gt == "" and "wav2vec2" in model.lower() and pt == "":
                        chunk["filters"]["silence"] = True   
                        pred["scores"] = {
                                    "bleu": 0.0,
                                    "wer": 0.0,
                                    "first": 0.0,
                                    "last": 0.0,
                                        }
                    else:
                        if gt != "" and pt != "":
                            pred["scores"] = {
                                    "bleu": calculate_bleu(gt, pt),
                                    "wer": calculate_wer(gt, pt),
                                    "first": fuzz.ratio(gt_words[0], pt_words[0]),
                                    "last": fuzz.ratio(gt_words[-1], pt_words[-1]),
                                        }
                        elif [gt == "" and pt != ""] or [gt != "" and pt == ""]:
                            pred["scores"] = {
                                    "bleu": 0.0,
                                    "wer": 0.0,
                                    "first": 0.0,
                                    "last": 0.0,
                                        }
                    if "wav2vec2" in model.lower():
                        wav2vec_scores = pred["scores"]
                    elif "whisper" in model.lower():
                        whisper_scores = pred["scores"] 
                else:
                    print(f"Transcription missing in {file_to_process}, chunk: {i}")
                    break      

            if whisper_scores != None and wav2vec_scores != None:
                # change the values to the desired thresholds
                if wav2vec_scores["bleu"] > 0.4 or whisper_scores["bleu"] > 0.8:
                    chunk["filters"]["stage1_whisper"] = True
                if whisper_scores["bleu"] > 0.8 and whisper_scores["wer"] < 0.2:
                    chunk["filters"]["stage2_whisper"] = True
                if whisper_scores["bleu"] > 0.8 and whisper_scores["first"] >= 80.0 and whisper_scores["last"] >= 80.0 and whisper_scores["wer"] < 0.2:
                    chunk["filters"]["stage2_whisper_timestamps"] = True
                if [wav2vec_scores["bleu"] > 0.4 or whisper_scores["bleu"] > 0.8] and whisper_scores["first"] >= 80.0 and whisper_scores["last"] >= 80.0 \
                    and wav2vec_scores["first"] >= 80.0 and wav2vec_scores["last"] >= 80.0 and whisper_scores["wer"] < 0.2 and wav2vec_scores["wer"] < 0.2:
                    chunk["filters"]["stage1_wav2vec"] = True    
            else:
                print(f"Scores missing in {file_to_process}, chunk: {i}")
                break    
                                    
    if save_as_new:    
        file_to_save = file_to_process.split(".json")[0]
        with open(file_to_save + '_scores.json', "w") as f:
            json.dump(d, f, ensure_ascii=False, indent=4)
    else:   
        with open(file_to_process, "w") as f:
            json.dump(d, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    
    args = get_args()
    
    json_files = []
    with open(args.json_files) as fh:
        for line in fh:
                json_files.append(line.strip())
    
    with Pool(args.num_proc) as pool:
        results = pool.starmap(calculate_scores, [(file, args.recalculate, args.save_as_new) for file in json_files])
    

"""
If all automatic transcriptions have low scores: throw away ground truth
If at least one model has good transcription: save ground truth
Throw away if bleu < x

If first and last have "> score", use these examples for training with timestamps

For wav2vec2 finetuning, the bleu_score should be close to 1. And first&last should match.
"""
