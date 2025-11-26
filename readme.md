# Text Classification with Hugging Face Transformers

This repository contains a minimal, configurable script for fine-tuning BERT (or any compatible encoder) on text classification datasets from the 🤗 Hub.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Fine-tune BERT on the IMDb sentiment dataset:

```bash
python train.py --dataset_name imdb --model_name bert-base-uncased --output_dir ./outputs/imdb
```

Key options:

- `--dataset_name` / `--dataset_config`: Dataset from the 🤗 Hub (e.g., `imdb`, `ag_news`, or `glue --dataset_config sst2`).
- `--text_column` / `--label_column`: Customize column names if the dataset schema differs.
- `--train_split` / `--eval_split`: Choose which dataset splits to use. If no eval split is provided and none exists, a validation set is created from the training split using `--validation_split_ratio`.
- `--batch_size`, `--learning_rate`, `--epochs`, `--max_length`: Training hyperparameters.
- `--max_train_samples` / `--max_eval_samples`: Subset the dataset for quick experiments.
- `--metric`: Any metric available through the `evaluate` library (default: `accuracy`).
- `--push_to_hub` and `--hub_model_id`: Push the fine-tuned model to the Hugging Face Hub.

Example for GLUE SST-2 with smaller subsets for a quick test run:

```bash
python train.py \
  --dataset_name glue \
  --dataset_config sst2 \
  --text_column sentence \
  --model_name bert-base-uncased \
  --max_train_samples 2000 \
  --max_eval_samples 500 \
  --output_dir ./outputs/glue-sst2
```

The script saves checkpoints and evaluation metrics to the specified `output_dir` and can automatically load the best checkpoint when training completes.
