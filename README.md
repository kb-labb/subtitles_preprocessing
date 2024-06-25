# Pipeline for preprocesing Swedish subtitles

Scripts to process data from filmot.com for videos with Swedish subtitles, and to subsequently scrape the audio and subtitles from Youtube. Convenience functions installable as a package: `yt_sub`. The resulting audio/json files from all preprocessing steps are located on the network drive under `delat/youtube`. 

## Environment

The dependencies are listed in `environment.yml`. To create a conda environment called `whisper`, run:

```bash
conda env create -f environment.yml
``` 

## Installation

Clone this repo and install it as a package `yt_sub` in editable mode with

```bash
pip install -e .
```

### Flash attention 2

For flash attention 2, make sure your Nvidia graphics drivers support a CUDA version `>=11.8` (run `nvidia-smi` to check max CUDA version that drivers support).

You can then install CUDA 11.8 via conda:

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

and flash attention 

```bash
MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

## Scripts

The scripts are located in the `scripts` folder. 

### Preprocess the subtitles from each individual source
This pipeline works for different sources of subtitles paired with audio, such as SVT, smdb, youtube, swedia etc. 
Once subtitles paired with audio is extracted and located in a directo ry, the preprocessing pipeline starts with the following scripts: 
1. Preprocess subtitles and output json
* `python scripts/02_preprocess_sub.py` for smdb
* `python scripts/02_preprocess_svt.py` for SVT
* `python scripts/02_preprocess_swedia.py` for swedia
These scripts call the function make_chunks in https://github.com/kb-labb/subtitles_preprocessing/blob/main/src/sub_preproc/utils/make_chunks.py that returns a json file with metadata and chunks split according to min and max thresholds specified when the function is called.  
2. Create `json_files_xxx.txt` file with a list of all json files created in the previous step, where `xxx` is your data source, such as SVT, smdb etc. Script to modify according to your folder structure/json file name etc:
`find .  -name *.json -exec bash -c 'd=${1%/*}; d=${d##*/}; printf "%s %s\n" "$1" ' Out {} \; >json_files_xxx.txt` 
3. With the `json_files_xxx.txt` created in the previous step as input, run the transcription with Whisper. 
`python scripts/transcribe_whisper.py --json_files files_json_xxx.txt` 
This script transcribes the audio in each chunk and adds transcription information in the json file, with the name of the model used to transcribe. 
4. With the `json_files_xxx.txt` created in the previous step as input, run the transcription with Wav2vec2.0. 
`python scripts/transcribe_wav2vec.py --json_files files_json_xxx.txt` 
This script transcribes the audio in each chunk and adds transcription information in the json file, with the name of the model used to transcribe. 
5. The script `chunk_scorer.py` uses the transcriptions obtained in step 3 and 4 as well as the subtitle string to calculate bleu and wer scores, and match first and last words. The resulting scores are used to calculate four quality criteria for the subtitles, that states how well the subtitle string corresponds to the speech. The four quality criteria are named `stage1_whisper`, `stage2_whisper`, `stage2_whisper_timestamps` and `stage1_wav2vec`, where `stage1_whisper` is the lowest quality and `stage1_wav2vec` is the highest quality, and the name states the type of training the data is suitable for. The scripts adds the quality information under `filter` in the json file. 
`python scripts/chunk_scorer.py --json_files files_json_xxx.txt` 
6. The result of the chunks scoring in step 5 is used to extract the audio data used in the training. There are two alternatives here depending on the training type:
* Wav2vec2.0 training:
use `make_audio_chunks.py` to extract audio chunks (in .wav) and the subtitle (.txt) for each chunk if `stage1_wav2vec` == True. This script creates a new folder with .wav and .txt files for all chunks of all original files. `python scripts/make_audio_chunks.py --json_files files_json_xxx.txt` 

* Whisper training:
use `create_parquet_metadata.py` to create a parquet file with all metadata needed for the whisper training, along with the various quality criteria. This script creates one parquet file for each original audio file, with information on how to split the audio into chunks. `python scripts/create_parquet_metadata.py --json_files files_json_xxx.txt` 
