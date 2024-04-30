import argparse
import datetime
import json
import logging
import os

import torch
from sub_preproc.utils.dataset import (
    AudioFileChunkerDataset,
    custom_collate_fn,
    make_transcription_chunks,
)
from sub_preproc.utils.make_chunks import n_non_silent_chunks
from tqdm import tqdm
from transformers import AutoProcessor, WhisperForConditionalGeneration


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, default="openai/whisper-large-v3")
    argparser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "eager", "sdpa"],
        help="""Attention implementation to use. SDPA is default for torch>=2.1.1 in Hugging Face. 
        Otherwise eager is default. Use flash_attention_2 if you have installed the flash-attention package.""",
    )
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
        filename=f"logs/transcribe_whisper-{now}.log",
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
    for line in json_files:
        line = line.split()
        assert 0 < len(line) and len(line) <= 2
        with open(line[0]) as f:
            try:
                vad_dict = json.load(f)
                if n_non_silent_chunks(vad_dict) == 0:
                    # Skip empty or only static audio files
                    empty_json_files.append(line)
                else:
                    # audio_files.append(vad_dict["metadata"]["audio_path"])
                    if len(line) == 2:
                        audio_files.append(line[1])
                    else:
                        audio_files.append(line[0][:-5] + ".wav")
                    vad_dicts.append(vad_dict)
            except json.JSONDecodeError:
                logging.info(f"failed reading json-file {line[0]}")

    json_files = [json_file for json_file in json_files if json_file not in empty_json_files]

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.float16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_name, sample_rate=16_000, return_tensors="pt"
    )

    audio_dataset = AudioFileChunkerDataset(
        audio_paths=audio_files,
        json_paths=json_files,
        model_name=args.model_name,
        processor=processor,
    )

    # Create a torch dataloader
    dataloader_datasets = torch.utils.data.DataLoader(
        audio_dataset,
        batch_size=1,
        num_workers=3,
        collate_fn=custom_collate_fn,
        shuffle=False,
    )

    for dataset_info in tqdm(dataloader_datasets):
        if dataset_info[0]["is_transcribed_same_model"]:
            logger.info(f"Already transcribed: {dataset_info[0]['json_path']}.")
            continue  # Skip already transcribed videos

        dataset = dataset_info[0]["dataset"]
        dataloader_mel = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            pin_memory_device=f"cuda:{args.gpu_id}",
            shuffle=False,
        )

        transcription_texts = []

        for batch in dataloader_mel:
            batch = batch.to(device).half()
            predicted_ids = model.generate(
                batch,
                return_dict_in_generate=True,
                task="transcribe",
                language="sv",
                output_scores=True,
                max_length=args.max_length,
            )
            transcription = audio_dataset.processor.batch_decode(
                predicted_ids["sequences"], skip_special_tokens=True
            )

            transcription_chunk = make_transcription_chunks(transcription, args.model_name)
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
            # json.dump(sub_dict, f, ensure_ascii=False, indent=4)
            json.dump(sub_dict, f, indent=4)

        logger.info(f"Transcription finished: {dataset_info[0]['json_path']}.")


if __name__ == "__main__":
    main()
