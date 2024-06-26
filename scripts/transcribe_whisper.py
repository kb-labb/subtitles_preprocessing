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
    argparser.add_argument("--batch_size", type=int, default=16)
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
    logger.info("Reading json-file list")
    json_files = []
    with open(args.json_files) as fh:
        if args.json_files.endswith(".json"):
            json_files = json.load(fh)
        else:
            for line in fh:
                json_files.append(line.strip())


    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.float16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_name, sample_rate=16_000, return_tensors="pt"
    )

    def my_filter(x):
        if x["duration"] < 20_000:
            return False
        if "language_probs" not in x:
            return False
        if "transcription" in x:
            if args.model_name in x["transcription"]:
                return False
        if max(x["language_probs"]["openai/whisper-large-v3"].items(), key=lambda x: x[1])[0] != "sv":
            return False
        return True


    audio_dataset = AudioFileChunkerDataset(
        json_paths=json_files,
        model_name=args.model_name,
        processor=processor,
        chunks_or_subs="chunks",
        my_filter=my_filter,
        logger=logger,
    )

    # Create a torch dataloader
    dataloader_datasets = torch.utils.data.DataLoader(
        audio_dataset,
        batch_size=1,
        num_workers=4,
        prefetch_factor=4,
        collate_fn=custom_collate_fn,
        shuffle=False,
    )

    logger.info("Iterate over outer dataloader")
    for dataset_info in tqdm(dataloader_datasets):
        try:
            if dataset_info[0]["dataset"] is None:
                logger.info(f"Do nothing for {dataset_info[0]['json_path']}")
                continue

            dataset = dataset_info[0]["dataset"]
            dataloader_mel = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
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
            assert len(list(filter(lambda x: my_filter(x), sub_dict["chunks"]))) == len(transcription_texts)

            for i, chunk in enumerate(filter(lambda x: my_filter(x), sub_dict["chunks"])):
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
        except Exception as e:
            logger.info(f"Transcription failed: {dataset_info[0]['json_path']}. Exception was {e}")


if __name__ == "__main__":
    main()
