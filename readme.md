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
- `--mixed_precision`: Enable `fp16` or `bf16` training when your GPU supports it (useful on ROCm/AMD or NVIDIA).
- `--no_cuda`: Force CPU training if you want to avoid GPU usage.
- `--push_to_hub` and `--hub_model_id`: Push the fine-tuned model to the Hugging Face Hub.

### Train on local Slovak JSONL data and auto-assign filters

If your training data lives in a JSON Lines file with Slovak fields `Popis` (text) and `Filter` (label), and you have a separate
JSONL file listing the available filters, you can fine-tune and then classify unlabeled data like this:

```bash
python train.py \
  --train_file data/train.jsonl \
  --eval_file data/val.jsonl \
  --filter_list_file data/filters.jsonl \
  --text_column Popis \
  --label_column Filter \
  --model_name bert-base-multilingual-cased \
  --output_dir ./outputs/filters
```

To predict filters for unlabeled items stored in `data/unlabeled.jsonl`, add:

```bash
python train.py \
  --train_file data/train.jsonl \
  --eval_file data/val.jsonl \
  --filter_list_file data/filters.jsonl \
  --predict_file data/unlabeled.jsonl \
  --text_column Popis \
  --label_column Filter \
  --model_name bert-base-multilingual-cased \
  --output_dir ./outputs/filters
```

Predicted filters (with the original `Popis` text and a confidence score) are written to `./outputs/filters/predictions.jsonl` by default.

## Running on AMD GPUs (ROCm)

PyTorch and the Hugging Face stack run on AMD GPUs via ROCm. Install the ROCm-enabled PyTorch wheels before
installing the rest of the dependencies:

```bash
# Pick the command for your ROCm/PyTorch version from https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements.txt
```

Then fine-tune with mixed precision to speed up training if your GPU supports bfloat16:

```bash
python train.py --dataset_name imdb --model_name bert-base-uncased --mixed_precision bf16 --output_dir ./outputs/imdb
```

If you want to run on CPU instead (or your ROCm install is not detected), use `--no_cuda`.

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
