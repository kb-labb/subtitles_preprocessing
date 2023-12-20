import argparse
import os
import time
import pandas as pd
import csv
from tqdm import tqdm
from pydub import AudioSegment
import concurrent.futures
import logging
from faster_whisper import WhisperModel


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.info("Starting")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="results",
        help="Directory containing subdirectories with audio and srt files.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="chunks",
        help="Directory where the resulting wav file chunks should be saved.",
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=8,
        help="Number of processes to use for multiprocessing.",
    )

    parser.add_argument(
        "--parquet_dir",
        type=str,
        default="subs_preprocessed",
        help="Name of directory where parquet file(s) are stored.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="language_info.csv",
        help="Directory where csv file with info about subs will be saved.",
    )
    return parser.parse_args()


def create_statistics(channel, file, lang, lang_prob, csv_file, output_dir):
    """Create a dataframe with info about available subtitles."""
    save_dir = os.path.join(output_dir, channel, csv_file)
    exists = os.path.exists(save_dir)
    with open(save_dir, mode="a", newline="") as save_file:
        fieldnames = ["channel", "filename", "language", "probability"]
        writer = csv.DictWriter(save_file, fieldnames=fieldnames)

        # Write the header row
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "channel": channel,
                "filename": str(file) + ".wav",
                "language": lang,
                "probability": lang_prob,
            }
        )


# Save processed audio as desired format
def save_processed_audio(audio_segment, output_path):
    audio_segment.export(output_path, format="wav")  # Saving as WAV format


def split_audio(audio_df: pd.DataFrame, data_folder, output_dir, model, csv_file):
    remove_empty = []
    f = audio_df.iloc[0, audio_df.columns.get_loc("program")]
    audio_file_name = audio_df.iloc[0, audio_df.columns.get_loc("audio")]
    audio_path = f"{data_folder}/{f}/{audio_file_name}"
    audio = AudioSegment.from_wav(audio_path)
    os.makedirs(output_dir, exist_ok=True)
    filenames = []
    for i, row in audio_df.iterrows():
        if (row["end_bucket"] - row["start_bucket"]) < 2000:
            continue
        audio_chunk = audio[row["start_bucket"] : row["end_bucket"]]
        os.makedirs(os.path.join(output_dir, row["program"]), exist_ok=True)
        filename = f"{output_dir}/{row['program']}/{row['observation_nr']}.wav"
        output_path = os.path.join(audio_path, filename)
        save_processed_audio(audio_chunk, filename)
        # language detection
        segments, info = model.transcribe(filename, vad_filter=True, beam_size=5)
        if info.language == "sv" and info.language_probability > 0.5:
            print(
                "Detected language '%s' with probability %f"
                % (info.language, info.language_probability)
            )
            audio_chunk.export(filename, format="wav")
            create_statistics(
                row["program"],
                row["observation_nr"],
                info.language,
                info.language_probability,
                csv_file,
            )
            filenames.append(filename)
        else:
            # if swedish not detected, check previous block, if it is swedish, save the audio + the detected language
            prev_filename = (
                f"{output_dir}/{row['program']}/{row['observation_nr']-1}.wav"
            )
            if os.path.exists(prev_filename):
                prev_segments, prev_info = model.transcribe(
                    prev_filename, vad_filter=True, beam_size=5
                )
                if prev_info.language == "sv":
                    audio_chunk.export(filename, format="wav")
                    filenames.append(filename)
                    create_statistics(
                        row["program"],
                        row["observation_nr"],
                        info.language,
                        info.language_probability,
                        csv_file,
                    )
                    print(
                        "Detected language '%s' with probability %f"
                        % (info.language, info.language_probability)
                    )
                else:
                    remove_empty.append(filename)
    for i in remove_empty:
        print("removing !", i)
        os.remove(i)

    return filenames


def main() -> None:
    args = get_args()
    output_dir = args.output
    csv_file = args.csv

    parquet_files = [d for d in os.listdir(args.parquet_dir)]
    sampling_rate = 16000
    model_size = "large-v2"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    metadata_df = pd.DataFrame()
    for parquet_file in tqdm(parquet_files):
        logging.info(f"Processing parquet file {parquet_file}")

        df_subs = pd.read_parquet(args.parquet_dir + "/" + parquet_file)
        df_subs["bucket_filename"] = df_subs["observation_nr"].astype(str) + ".wav"
        # Keep only first obs in each observation_nr group
        df_subs_unique = df_subs.drop_duplicates(
            ["observation_nr", "audio"], keep="first"
        )
        df_subs_unique = df_subs_unique.drop(df_subs_unique.tail(1).index)

        audio_groups = df_subs_unique[
            [
                "program",
                "audio",
                "start_bucket",
                "end_bucket",
                "bucket_filename",
                "observation_nr",
            ]
        ].groupby("audio")
        df_list = [audio_groups.get_group(df) for df in audio_groups.groups]
        logging.info("Splitting audio")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            out_filenames = list(
                tqdm(executor.map(split_audio, df_list), total=len(df_list))
            )
        # drop filenames that didn't pass language detection check by a left join
        out_filenames_new = [
            x.replace(f"{output_dir}/{df_subs['program'][0]}/", "")
            for x in out_filenames[0]
        ]
        out_filenames_df = pd.DataFrame({"bucket_filename": out_filenames_new})
        df_subs_unique = df_subs_unique.merge(
            out_filenames_df.drop_duplicates(),
            on=["bucket_filename"],
            how="left",
            indicator=True,
        )
        df_subs_unique = df_subs_unique[df_subs_unique._merge == "both"]
        if df_subs_unique.empty:
            os.rmdir(f"{output_dir}/{df_subs['program'][0]}")
        else:
            df_subs_unique = df_subs_unique.copy()
            df_subs_unique["bucket_path"] = [
                bucket_path
                for program_paths_list in out_filenames
                for bucket_path in program_paths_list
            ]
            df_subs_unique["bucket_filename"] = df_subs_unique[
                "bucket_filename"
            ].str.slice_replace(0, 0, (df_subs_unique["program"].iloc[0] + "/"))

            result = df_subs_unique[
                ["bucket_filename", "text_timestamps_bucket"]
            ].rename(
                columns={
                    "bucket_filename": "file_name",
                    "text_timestamps_bucket": "transcription",
                }
            )
            result.to_csv(
                f"{output_dir}/{df_subs['program'][0]}/metadata.csv", index=False
            )

    logging.info("Done")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Audio split took {time.time() - start} seconds")
