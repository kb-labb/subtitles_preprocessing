#!/bin/bash

# Example of how to run the script (remember to change the --dataset arg accordingly):
# bash launch_preprocess.sh /leonardo_work/EUHPC_A01_006/data/big_parquets/svt svt1_0*.parquet
# bash launch_preprocess.sh /leonardo_work/EUHPC_A01_006/data/big_parquets/riksdagen_web *.parquet

# Read list of files from a directory and launch preprocess_and_filter.py for each file
DATA_DIR=$1 # E.g. /leonardo_work/EUHPC_A01_006/data/big_parquets/svt
GLOB_PATTERN=$2 # Wildcard pattern, e.g. svt1_01*.parquet or *.parquet
STAGE="stage2"  # which stage of training? stage1, stage2 or stage2_wav2vec2
DATASET="isof"  # which dataset? svt, smdb, rixvox, youtube, isof or sls

# Destination directory
DEST_DIR="/leonardo_scratch/large/userexternal/jsikora0/parquet_stages_alt/${STAGE}/$(basename ${DATA_DIR})" 
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
echo "Launching preprocess_and_filter.py for the following files:"
echo "${REMAINING_FILENAMES}"

for PARQUET_FILE in ${REMAINING_FILENAMES}; do
    COMMAND="python preprocess_and_filter.py \
        --data_dir ${DATA_DIR} \
        --output_dir /leonardo_scratch/large/userexternal/jsikora0/parquet_stages_alt \
        --parquet_filename $(basename ${PARQUET_FILE}) \
        --dataset ${DATASET} \
        --stage ${STAGE} \
        --model_name_or_path openai/whisper-small \
        --cache_dir cache \
        --language sv \
        --task transcribe \
        --sampling_rate 16000 \
        --bpe_dropout 0.0 \
        --apply_spec_augment \
        --mask_time_prob 0.5 \
        --mask_time_length 10 \
        --mask_feature_prob 0.0 \
        --mask_feature_length 10 \
        --min_input_length 8000 \
        --max_input_length 480000 \
        --stats_dir /leonardo_work/EUHPC_A01_006/data/big_parquets/stats/whisper-smaller
        "

    srun --partition=boost_usr_prod --nodes=1 \
        --ntasks=1 --cpus-per-task=2 --mem=80GB \
        --gres=gpu:0 --time=0-00:40:00 --qos=normal \
        --account=EUHPC_A01_006 \
        bash -c "${COMMAND}" 2>&1 | tee -a logs/$(basename ${PARQUET_FILE}).out &
done

# Wait for background jobs to finish before exiting
# wait

# --begin=now+15minutes

# source venvs/whisper/bin/activate
