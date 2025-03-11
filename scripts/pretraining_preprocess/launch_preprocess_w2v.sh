#!/bin/bash

# Example of how to run the script (remember to change the --dataset arg accordingly):
# bash launch_preprocess.sh /leonardo_work/EUHPC_A01_006/data/big_parquets/svt svt1_0*.parquet
# bash launch_preprocess.sh /leonardo_work/EUHPC_A01_006/data/big_parquets/riksdagen_web *.parquet

# Read list of files from a directory and launch preprocess_and_filter.py for each file
DATA_DIR=$1 # E.g. /leonardo_work/EUHPC_A01_006/data/big_parquets/svt
GLOB_PATTERN=$2 # Wildcard pattern, e.g. svt1_01*.parquet or *.parquet
STAGE="stage_wav2vec2"  # which stage of training? stage_wav2vec2
DATASET="rixvox"  # which dataset? svt, smdb, rixvox, youtube, isof, nst or sls
OUTPUT_DIR="/leonardo_scratch/large/userexternal/jsikora0/parquet_wav2vec2" # Change --stats_dir depending on if large, or smaller models

# Destination directory
DEST_DIR="${OUTPUT_DIR}/${STAGE}/$(basename ${DATA_DIR})"

# List source files matching the pattern
SOURCE_PARQUET_PATHS=$(ls ${DATA_DIR}/${GLOB_PATTERN})
SOURCE_FILENAMES=$(echo "$SOURCE_PARQUET_PATHS" | xargs -n 1 basename)

if [ -z "$SOURCE_PARQUET_PATHS" ]; then
    echo "No files found in ${DATA_DIR} matching pattern ${GLOB_PATTERN}."
    exit 1
fi

# Check if destination directory exists and is not empty
if [ -d "$DEST_DIR" ] && [ "$(ls -A $DEST_DIR)" ]; then
    echo "Destination directory ${DEST_DIR} already exists and is not empty."
    echo "Checking for files that have not been processed yet..."
    # Extract filenames in the destination directory
    DEST_FILENAMES=$(ls ${DEST_DIR})

    # Find files in source not present in destination
    REMAINING_FILENAMES=$(comm -23 <(echo "$SOURCE_FILENAMES" | sort) <(echo "$DEST_FILENAMES" | sort))
else
    echo "Destination directory ${DEST_DIR} does not exist or is empty."
    REMAINING_FILENAMES="$SOURCE_PARQUET_PATHS"
fi

source venvs/whisper/bin/activate

echo "Launching preprocess_and_filter_w2v.py for the following files:"
echo "${REMAINING_FILENAMES}"

i=0
MINUTES=0 # Minutes from now to start the first job
BEGIN="now+${MINUTES}minutes"
for PARQUET_FILE in ${REMAINING_FILENAMES}; do
    COMMAND="python preprocess_and_filter_w2v.py \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --parquet_filename $(basename ${PARQUET_FILE}) \
        --dataset ${DATASET} \
        --stage ${STAGE} \
        --cache_dir cache \
        --min_input_length 800 \
        --max_input_length 480000 \
        --stats_dir /leonardo_work/EUHPC_A01_006/data/big_parquets/stats/wav2vec2
        "
    
    # We add X minutes per 20 files processed to BEGIN to avoid all jobs starting at the same time
    if [ $((i % 20)) -eq 0 ]; then
        MINUTES=$((MINUTES+2))
        BEGIN="now+${MINUTES}minutes"
    fi

    srun --partition=boost_usr_prod --nodes=1 \
        --ntasks=1 --cpus-per-task=2 --mem=90GB \
        --gres=gpu:0 --time=0-00:40:00 --qos=normal \
        --account=EUHPC_A01_006 \
        --begin=${BEGIN} \
        bash -c "${COMMAND}" 2>&1 | tee -a logs/$(basename ${PARQUET_FILE}).out &
    
    # Increment counter
    i=$((i+1))
done

# Wait for background jobs to finish before exiting
# wait

# --begin=now+15minutes
# source venvs/whisper/bin/activate
