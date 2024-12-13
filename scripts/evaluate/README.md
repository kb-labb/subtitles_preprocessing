## Evaluation

And example of a simple evaluation script is `eval_fleurs.py`.

### How to load model from checkpoints

The process summarized:

1. You create a directory to store all files required to load your model.
2. Copy the `config.json`, `merges.txt`, `vocab.json`, `tokenizer.json`, `special_tokens_map.json`, `added_tokens.json`, `normalizer.json`, `preprocessor_config.json` files to the directory in step 1.
3. From a checkpoint directory of interest, copy the `model.safensors` file to the directory in step 1. Also copy over the `generation_config.json` file at least once from one of the checkpoint directories.

The base checkpoint folder contains some `json` files and a `merges.txt` file. These should be copied to your model directory.

```
outputs/2024-12-06_medium-stage1
├── added_tokens.json
├── checkpoint-100500
├── checkpoint-101250
├── checkpoint-102000
├── checkpoint-102750
├── checkpoint-103500
├── checkpoint-104250
├── config.json
├── merges.txt
├── normalizer.json
├── preprocessor_config.json
├── runs
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.json
```

The checkpoint directories contain a `model.safensors` file and a `generation_config.json` file. These should be copied to your model directory.

Then you can load the model pointing at your directory as done in `eval_fleurs.py`. 