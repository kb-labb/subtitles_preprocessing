import argparse
import datetime
import json
import logging
import os

import torch
from sub_preproc.utils.dataset import (
    AudioFileChunkerDataset,
    custom_collate_fn,
    make_transcription_chunks_w2v,
    wav2vec_collate_fn,
)
from sub_preproc.utils.make_chunks import n_non_silent_chunks
from tqdm import tqdm
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
)


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
    argparser.add_argument("--batch_size", type=int, default=16)
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
    logger.info("Reading json-file list")
    json_files = []
    with open(args.json_files) as fh:
        if args.json_files.endswith(".json"):
            json_files = json.load(fh)
        else:
            for line in fh:
                json_files.append(line.strip())

    model = AutoModelForCTC.from_pretrained(args.model_name, torch_dtype=torch.float16).to(device)

    processor = AutoProcessor.from_pretrained(
        args.model_name, sample_rate=16000, return_tensors="pt"
    )
    # processor = Wav2Vec2Processor.from_pretrained(
    #     args.model_name, sample_rate=16000, return_tensors="pt"
    # )
    # processor = Wav2Vec2ProcessorWithLM.from_pretrained(
    #     "/home/robkur/workspace/make_kenlm/voxrex_europarl-5gram",
    #     sample_rate=16_000,
    #     return_tensors="pt",
    # )

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

    dataloader_datasets = torch.utils.data.DataLoader(
        audio_dataset,
        batch_size=1,
        num_workers=4,
        prefetch_factor=4,
        collate_fn=custom_collate_fn,
        shuffle=False,
    )

    TIME_OFFSET = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate

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

                # probs = torch.nn.functional.softmax(logits, dim=-1)  # Need for CTC segmentation
                if type(processor) == Wav2Vec2Processor:
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = audio_dataset.processor.batch_decode(
                        predicted_ids, output_word_offsets=True
                    )
                elif type(processor) == Wav2Vec2ProcessorWithLM:
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
