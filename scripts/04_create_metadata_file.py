import pandas as pd
import os
from os import makedirs
import argparse
import glob


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--data",
        type=str,
        default="audio_chunks",
        help="Directory with chunks.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metadata",
        help="Directory where the concatenated metadata file should be saved.",
    )

    return parser.parse_args()

def get_txt_files(base_dir):
    return glob.iglob(f"{base_dir}/**/*.txt", recursive=True)

def get_dir_names(base_dir):
    return glob.glob(f"{base_dir}/**/", recursive=True)

def main() -> None:
    args = get_args()
    chunks_dir = args.data
    output_dir = args.output
    concatenated_metadata = pd.DataFrame()
    source = ''
    if not os.path.exists(output_dir):
        makedirs(output_dir)
    for d in get_dir_names(chunks_dir):
        concatenated_metadata = pd.DataFrame()
        if 'youtube' in d:
            source = 'youtube_'
        ### add other sources of data flags 
        if "stage" in d:
            for file in get_txt_files(d):

                dirname, basename = os.path.split(file)
                stage_filter = os.path.basename(os.path.dirname(file))
                with open(d+'/'+basename, 'r') as f:
                    text = f.read().rstrip()
                metadata = pd.DataFrame({"filename": [d+basename[:-4]+'.wav'] , "transcription": [text]})    
                concatenated_metadata = pd.concat([concatenated_metadata, metadata])
            concatenated_metadata.to_csv(f"{output_dir}/metadata_{source}{stage_filter}.csv", index=False)                                


if __name__ == "__main__":
    main()
