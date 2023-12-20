import os
import pandas as pd
from datasets import Audio, Dataset, DatasetDict


# Filter out small files
def filter_files_by_size(df):
    base_path = "/Users/rrj/svt/undertext-whisper-dataset/chunks/minio/chunks/"
    valid_files = []
    valid_transcriptions = []
    for idx, row in df.iterrows():
        file_path = os.path.join(base_path, row["file_name"])
        if os.path.getsize(file_path) > 225:
            valid_files.append(file_path)
            valid_transcriptions.append(row["transcription"])
    return valid_files, valid_transcriptions


def main() -> None:
    # Load CSVs
    df_train = pd.read_csv(
        "/Users/rrj/svt/undertext-whisper-dataset/chunks/minio/chunks/train.csv"
    )
    df_validation = pd.read_csv(
        "/Users/rrj/svt/undertext-whisper-dataset/chunks/minio/chunks/validation.csv"
    )
    df_test = pd.read_csv("/Users/rrj/svt/undertext-whisper-dataset/chunks/minio/chunks/test.csv")

    train_files, train_transcriptions = filter_files_by_size(df_train)
    validation_files, validation_transcriptions = filter_files_by_size(df_validation)
    test_files, test_transcriptions = filter_files_by_size(df_test)

    # Create datasets
    train_audio_dataset = Dataset.from_dict(
        {"audio": train_files, "transcription": train_transcriptions}
    ).cast_column("audio", Audio())
    test_audio_dataset = Dataset.from_dict(
        {"audio": test_files, "transcription": test_transcriptions}
    ).cast_column("audio", Audio())
    validation_audio_dataset = Dataset.from_dict(
        {"audio": validation_files, "transcription": validation_transcriptions}
    ).cast_column("audio", Audio())

    dataset = DatasetDict(
        {
            "train": train_audio_dataset,
            "test": test_audio_dataset,
            "validation": validation_audio_dataset,
        }
    )

    dataset.push_to_hub("svt-labs/subsound_v2", private=True)


if __name__ == "__main__":
    main()
