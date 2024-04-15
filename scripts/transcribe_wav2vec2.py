import argparse
import json
import logging
import os
import datetime

import torch
from sub_preproc.utils.dataset import (
    AudioFileChunkerDataset,
    custom_collate_fn,
    make_transcription_chunks_w2v,
    wav2vec_collate_fn,
)
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM


def get_word_timestamps_hf(word_offsets, time_offset):
    word_timestamps = []
    for word_offset in word_offsets:
        word_offset = [
            {
                "word": w["word"],
                "start_time": round(w["start_offset"] * time_offset, 2),
                "end_time": round(w["end_offset"] * time_offset, 2),
            }
            for w in word_offset
        ]
        word_timestamps.append(word_offset)
    return word_timestamps


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, default="KBLab/wav2vec2-large-voxrex-swedish")
    argparser.add_argument("--gpu_id", type=int, default=0)
    argparser.add_argument("--max_length", type=int, default=185)
    argparser.add_argument(
        "--overwrite_all", action="store_true", help="Overwrite all existing transcriptions."
    )
    argparser.add_argument(
        "--overwrite_model",
        action="store_true",
        help="Overwrite existing transcriptions for the model.",
    )
    argparser.add_argument("--json_files", type=str, required=True)

    return argparser.parse_args()


def main():
    os.makedirs("logs", exist_ok=True)
    now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    logging.basicConfig(
        filename=f"logs/transcribe_w2v-{now}.log",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    args = get_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # read vad json
    json_files = []
    with open(args.json_files) as fh:
        for line in fh:
            json_files.append(line.strip())

    audio_files = []
    vad_dicts = []
    empty_json_files = []
    for json_file in json_files:
        with open(json_file) as f:
            vad_dict = json.load(f)
            if len(vad_dict["chunks"]) == 0:
                # Skip empty or only static audio files
                empty_json_files.append(json_file)
                continue
            # audio_files.append(vad_dict["metadata"]["audio_path"])
            audio_files.append(json_file[:-5] + ".wav")
            vad_dicts.append(vad_dict)

    json_files = [json_file for json_file in json_files if json_file not in empty_json_files]

    model = AutoModelForCTC.from_pretrained(args.model_name, torch_dtype=torch.float16).to(device)

    # processor = Wav2Vec2Processor.from_pretrained(
    #     args.model_name, sample_rate=16000, return_tensors="pt"
    # )
    processor = Wav2Vec2ProcessorWithLM.from_pretrained("/home/robkur/workspace/make_kenlm/voxrex_europarl-5gram", sample_rate=16_000, return_tensors="pt")

    audio_dataset = AudioFileChunkerDataset(
        audio_paths=audio_files, json_paths=json_files, model_name=args.model_name, processor=processor,
    )

    dataloader_datasets = torch.utils.data.DataLoader(
        audio_dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=2,
        shuffle=False,
    )

    TIME_OFFSET = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate

    for dataset_info in tqdm(dataloader_datasets):
        logging.info(f"Transcribing: {dataset_info[0]['json_path']}.")

        if dataset_info[0]["is_transcribed_same_model"]:
            logger.info(f"Already transcribed: {dataset_info[0]['json_path']}.")
            continue  # Skip already transcribed videos

        dataset = dataset_info[0]["dataset"]
        dataloader_mel = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=4,
            collate_fn=wav2vec_collate_fn,
            pin_memory=True,
            pin_memory_device=f"cuda:{args.gpu_id}",
            shuffle=False,
        )

        transcription_texts = []

        for batch in dataloader_mel:
            batch = batch.to(device).half()
            with torch.inference_mode():
                logits = model(batch).logits

            probs = torch.nn.functional.softmax(logits, dim=-1)  # Need for CTC segmentation
            # predicted_ids = torch.argmax(logits, dim=-1)
            # transcription = audio_dataset.processor.batch_decode(
            #     predicted_ids, output_word_offsets=True
            # )
            transcription = audio_dataset.processor.batch_decode(
                logits.cpu().numpy(), output_word_offsets=True
            )

            word_timestamps = get_word_timestamps_hf(
                transcription["word_offsets"], time_offset=TIME_OFFSET
            )

            transcription_chunk = make_transcription_chunks_w2v(
                transcription["text"], word_timestamps=word_timestamps, model_name=args.model_name
            )
            transcription_texts.extend(transcription_chunk)

        # Add transcription to the json file
        sub_dict = dataset.sub_dict
        assert len(sub_dict["chunks"]) == len(transcription_texts)

        for i, chunk in enumerate(sub_dict["chunks"]):
            if args.overwrite_all or "transcription" not in chunk:
                chunk["transcription"] = [transcription_texts[i]]
            elif "transcription" in chunk:
                if args.overwrite_model:
                    for j, transcription in enumerate(chunk["transcription"]):
                        if transcription["model"] == args.model_name:
                            chunk["transcription"][j] = transcription_texts[i]
                else:
                    models = [transcription["model"] for transcription in chunk["transcription"]]
                    # Check if transcription already exists for the model
                    if args.model_name not in models:
                        chunk["transcription"].append(transcription_texts[i])

        # Save the json file
        with open(dataset_info[0]["json_path"], "w") as f:
            json.dump(sub_dict, f, ensure_ascii=False, indent=4)

        logger.info(f"Transcription finished: {dataset_info[0]['json_path']}.")


if __name__ == "__main__":
    main()
