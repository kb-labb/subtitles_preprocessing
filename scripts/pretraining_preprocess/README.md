### Create python venv for the project

Load modules:

```bash
module load profile/deeplrn
module load cineca-ai/3.0.1
```

Create a virtual environment:

```bash
python -m venv venvs/whisper
```

### Install dependencies

First, unload the `cineca-ai` module:

```bash
module unload cineca-ai
```

Activate the virtual environment and install the dependencies:

```bash
source venvs/whisper/bin/activate
```

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -r requirements-no-deps.txt
```

#### Download tokenizer, config and feature extractor

Compute nodes don't have access to the internet, so you need to download the tokenizer, config and feature extractor before running the preprocessing script. Run the following script to download and cache the files:

```python
import argparse
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--model_name_or_path",
    type=str,
    default="openai/whisper-small",
    help="""Tiny, base, small and medium have the same preprocessing.
    Large uses different settings for the FeatureExtractor, so need 
    to run the preprocessing separately for large models.""",
)
argparser.add_argument(
    "--cache_dir",
    type=str,
    default="cache",
    help="Directory to cache the downloaded tokenizer/feature extractor/model weights.",
)
args = argparser.parse_args()

# 2. Load pretrained model config, tokenizer, and feature extractor
config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
)
feature_extractor = AutoFeatureExtractor.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
    add_prefix_space=True,
)
```

### Run preprocessing

The preprocessing script is `preprocess_and_filter.py`. The script `launch_preprocessing.sh` helps you launch one SLURM job per parquet shard. 

Example:

```bash
bash launch_preprocess.sh /leonardo_work/EUHPC_A01_006/data/big_parquets/riksdagen_web *.parquet
```

This launches one job for every parquet file found in the folder `/leonardo_work/EUHPC_A01_006/data/big_parquets/riksdagen_web`. You can change the wildcard glob pattern to match the file patterns you want to process. For example, to preprocess only `riksdagen_web_0000.parquet` to `riksdagen_web_0009.parquet`, you can use:

```bash
bash launch_preprocess.sh /leonardo_work/EUHPC_A01_006/data/big_parquets/riksdagen_web riksdagen_web_000*.parquet
```

> [!IMPORTANT] 
> Remember to change the `--dataset` argument in the `launch_preprocess.sh` script to match the dataset you are preprocessing.

### Summarize dataset statistics

Use `summarize_dataset_shard_stats.py` to create summary statistics for the datasets, folders and entire dataset.

It will output three files, named:

* `dataset_stats.json`
* `folder_stats.json`
* `{stage}_total_stats.json`

where `{stage}` is `original`, `stage1`, `stage2` or `stage_wav2vec2`.

### Friendly advice

> [!TIP]
> It may be unwise to launch more than 1000 SLURM jobs at once. In my experience it has handled ~700 jobs at once fine. But when I tried to launch a second set of an entire dataset shards' worth of jobs simultaneously the SLURM scheduler started to refuse some of the jobs.