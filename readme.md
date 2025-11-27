# Slovak Filter Classification with Hugging Face Transformers

This repository fine-tunes BERT (or another encoder) to assign the correct **Filter** to Slovak text descriptions (**Popis**) stored in local JSONL files.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Prepare three JSONL files:

- **Training/Eval data** (`--train_file`, `--eval_file`): each line includes `Popis` (text) and `Filter` (label). Example:

  ```json
  {"Popis": "Výmena brzdových doštičiek vpredu", "Filter": "Brzdy"}
  ```

- **Filter list** (`--filter_list_file`): one line per allowed filter with optional descriptions, e.g.:

  ```json
  {"Filter": "Brzdy", "Description": "Servis brzdového systému"}
  ```

- **Unlabeled data** (`--predict_file`): only `Popis` (and any extra context fields you want to keep in the output).

## Train and evaluate on local JSONL files

Fine-tune a multilingual BERT model on your labeled Slovak data and restrict labels to the provided filter list:

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

Key options for this workflow:

- `--train_file`, `--eval_file`, `--predict_file`: Local JSONL inputs for training, evaluation, and unlabeled prediction.
- `--filter_list_file`: Enforces the allowed filter vocabulary and maps string labels to IDs.
- `--text_column`, `--label_column`: Map your column names (defaults: `text` and `label`).
- `--batch_size`, `--learning_rate`, `--epochs`, `--max_length`: Training hyperparameters.
- `--max_train_samples` / `--max_eval_samples`: Subset the data for quick tests.
- `--metric`: Metric from the `evaluate` library (default: `accuracy`).
- `--mixed_precision`: Enable `fp16` or `bf16` training when your GPU supports it.
- `--no_cuda`: Force CPU training.

## Predict filters for new data

After training, classify unlabeled items and write predictions with confidences to `./outputs/filters/predictions.jsonl`:

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

Each output line keeps the original fields (such as `Popis`) and adds `predicted_filter` plus `confidence`.

## Running on AMD GPUs (ROCm)

Install ROCm-enabled PyTorch wheels before the rest of the dependencies:

```bash
# Pick the command for your ROCm/PyTorch version from https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements.txt
```

Then fine-tune with mixed precision if your GPU supports bfloat16:

```bash
python train.py \
  --train_file data/train.jsonl \
  --eval_file data/val.jsonl \
  --filter_list_file data/filters.jsonl \
  --text_column Popis \
  --label_column Filter \
  --model_name bert-base-multilingual-cased \
  --mixed_precision bf16 \
  --output_dir ./outputs/filters
```

Use `--no_cuda` to force CPU training if ROCm is unavailable.
