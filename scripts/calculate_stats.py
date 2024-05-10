import json
import argparse
import pandas as pd
import csv 

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--json_files",
    type=str,
    help="Path to file containing json filepaths",
    default="files.txt",
    )
    
    parser.add_argument(
    "--output",
    type=str,
    help="Path to save the file with the results",
    default="stats.csv",
    )
    
    return parser.parse_args()

def main():
    
    args = get_args()
    
    final_results = {
        "stage1_whisper": {"count": 0, "duration": 0}, 
        "stage2_whisper": {"count": 0, "duration": 0},
        "stage2_whisper_timestamps": {"count": 0, "duration": 0}, 
        "stage1_wav2vec": {"count": 0, "duration": 0},
        "silence": {"count": 0, "duration": 0}
    }
    
    fieldnames = ["Filter", "Count", "Aggregated Duration"]

    json_files = []
    with open(args.json_files) as fh:
        for line in fh:
                json_files.append(line.strip())
                
    for file in json_files:
        with open(file, "r") as fh:
            d = json.load(fh)
            for i, chunk in enumerate(d["chunks"]):
                try:
                    print(f"Processing: {file}")
                    filters = chunk["filters"]
                    duration = chunk["duration"]
                    for filter_name, filter_info in final_results.items():
                        if filters[filter_name]:
                            filter_info["count"] += 1
                            filter_info["duration"] += duration
                except Exception as e:
                    print(f"Skipping: {file} - {e}")
                    break
                            
    print(final_results)
    
    with open(args.output, 'w', newline='') as csv_file:  
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for filter_name, filter_info in final_results.items():
            writer.writerow({"Filter": filter_name, "Count": filter_info["count"], "Aggregated Duration": filter_info["duration"]})

if __name__ == "__main__":
    main()
