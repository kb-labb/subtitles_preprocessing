#!/bin/bash

# Example of how to run the script (remember to change the --dataset arg accordingly):
# bash launch_preprocess.sh /leonardo_work/EUHPC_A01_006/data/big_parquets/svt svt1_0*.parquet
# bash launch_preprocess.sh /leonardo_work/EUHPC_A01_006/data/big_parquets/riksdagen_web *.parquet

# Read list of files from a directory and launch preprocess_and_filter.py for each file
DATA_DIR=$1 # E.g. /leonardo_work/EUHPC_A01_006/data/big_parquets/svt
GLOB_PATTERN=$2 # Wildcard pattern, e.g. svt1_01*.parquet
PARQUET_FILES=$(ls ${DATA_DIR}/${GLOB_PATTERN})

# Which parquet files exist in PARQUET_FILES but not in the output directory?
# for PARQUET_FILE in ${PARQUET_FILES}; do
#     PARQUET_FILENAME=$(basename ${PARQUET_FILE})
#     if [ ! -f /leonardo_scratch/large/userexternal/jsikora0/parquet_stages/${PARQUET_FILENAME} ]; then
#         echo "File ${PARQUET_FILENAME} does not exist in the output directory"
#     fi
# done


# source venvs/whisper/bin/activate

echo "Launching preprocess_and_filter.py for the following files:"
echo "${PARQUET_FILES}"

for PARQUET_FILE in ${PARQUET_FILES}; do
    COMMAND="python preprocess_and_filter.py \
        --data_dir ${DATA_DIR} \
        --output_dir /leonardo_scratch/large/userexternal/jsikora0/parquet_stages \
        --parquet_filename $(basename ${PARQUET_FILE}) \
        --dataset youtube \
        --stage stage1 \
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
        --stats_dir /leonardo_work/EUHPC_A01_006/data/big_parquets/stats
        "

    srun --partition=boost_usr_prod --nodes=1 \
        --ntasks=1 --cpus-per-task=2 --mem=70GB \
        --gres=gpu:0 --time=0-00:40:00 --qos=normal \
        --account=EUHPC_A01_006 \
        --output logs/$(basename ${PARQUET_FILE}).out \
        bash -c "${COMMAND}" &
done

# Wait for background jobs to finish before exiting
wait

# --begin=now+15minutes

# source venvs/whisper/bin/activate

# COMMAND="python preprocess_and_filter.py \
#     --data_dir /leonardo_work/EUHPC_A01_006/data/big_parquets/svt \
#     --output_dir /leonardo_scratch/fast/EUHPC_A01_006 \
#     --parquet_filename svt1_0365.parquet \
#     --dataset rixvox \
#     --stage stage1 \
#     --model_name_or_path openai/whisper-small \
#     --cache_dir cache \
#     --language sv \
#     --task transcribe \
#     --sampling_rate 16000 \
#     --bpe_dropout 0.2 \
#     --apply_spec_augment \
#     --mask_time_prob 0.5 \
#     --mask_time_length 10 \
#     --mask_feature_prob 0.0 \
#     --mask_feature_length 10 \
#     --min_input_length 8000 \
#     --max_input_length 480000
#     "

# srun --partition=boost_usr_prod --nodes=1 \
#     --ntasks=1 --cpus-per-task=2 --mem=40GB \
#     --gres=gpu:0 --time=0-02:30:00 --qos=normal \
#     --account=EUHPC_A01_006 \
#     bash -c "${COMMAND}"