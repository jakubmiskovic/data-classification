# Slovak Filter Classification with Hugging Face Transformers

This repository fine-tunes BERT (or another encoder) to assign the correct **filter** to Slovak text descriptions (**Popis**) stored in local JSONL files.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Prepare three JSONL files (sample files are in `data/`):

- **Training/Eval data** (`--train_file`, `--eval_file`): each line includes `Popis` (text) and `filter` (label). Example:

  ```json
  {"Popis": "Výmena brzdových doštičiek vpredu", "filter": "Brzdy"}
  ```

- **Filter list** (`--filter_list_file`): one line per allowed filter with optional descriptions, e.g.:

  ```json
  {"Filter": "Brzdy", "Description": "Servis brzdového systému"}
  ```

- **Unlabeled data** (`--predict_file`): only `Popis` (and any extra context fields you want to keep in the output).

The repository includes starter files so you can run end-to-end immediately:

- `data/filters.jsonl` — 154 allowed filters with short descriptions.
- `data/train.jsonl` — 1,199 labeled Popis/filter examples for training (includes multiline Popis entries).
- `data/val.jsonl` — placeholder (empty); the script will auto-create validation from the training file.
- `data/unlabeled.jsonl` — 499 Popis entries without labels for prediction.

### Multiline descriptions

JSONL requires one object per line. If your Popis text contains line breaks, escape them as `\\n` so the file stays single-lined; the loader and tokenizer will restore the newline characters during training. Example:

```jsonl
{"Popis": "Kontrola brzdových hadičiek\\nZákazník nahlásil únik kvapaliny", "filter": "Brzdy"}
```

The sample `data/train.jsonl` already includes one such multiline entry so you can verify ingestion end-to-end.

## Train and evaluate on local JSONL files

Fine-tune a multilingual BERT model on your labeled Slovak data (or the included sample files) and restrict labels to the provided filter list. By default, the script expects `Popis` and `filter` columns, uses a 20% validation split (created automatically from the training file if you omit `--eval_file`), and selects `bert-base-multilingual-cased`:

```bash
python train.py \
  --train_file data/train.jsonl \
  --filter_list_file data/filters.jsonl \
  --output_dir ./outputs/filters
```

Key options for this workflow:

- `--train_file`, `--eval_file`, `--predict_file`: Local JSONL inputs for training, evaluation, and unlabeled prediction.
- `--filter_list_file`: Enforces the allowed filter vocabulary and maps string labels to IDs.
- `--text_column`, `--label_column`: Map your column names (defaults: `Popis` and `filter`).
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
  --filter_list_file data/filters.jsonl \
  --predict_file data/unlabeled.jsonl \
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
  --filter_list_file data/filters.jsonl \
  --model_name bert-base-multilingual-cased \
  --mixed_precision bf16 \
  --output_dir ./outputs/filters
```

Use `--no_cuda` to force CPU training if ROCm is unavailable.
