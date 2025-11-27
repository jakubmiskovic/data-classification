"""Fine-tune a text classification model using Hugging Face Transformers.

This script demonstrates a minimal end-to-end workflow for sequence
classification with BERT (or any compatible encoder) on datasets from the
🤗 Datasets hub. It includes configurable hyperparameters, evaluation, and
model saving.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from datasets import DatasetDict, load_dataset
from evaluate import load
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class ModelConfig:
    model_name: str
    num_labels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a text classification model.")

    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name from the 🤗 Hub (e.g., 'ag_news').",
    )
    dataset_group.add_argument(
        "--train_file",
        type=str,
        help="Local training file in JSON Lines format (for example with 'Popis' and 'Filter' columns).",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Optional dataset config (e.g., 'sst2' for 'glue').",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Name of the training split in the dataset.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default=None,
        help="Optional name of the evaluation split. If not provided, will try 'validation' then 'test', or create one.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Optional local evaluation file in JSON Lines format (used when --train_file is provided).",
    )
    parser.add_argument(
        "--validation_split_ratio",
        type=float,
        default=0.1,
        help="Fraction of the training split to reserve for validation if no eval split exists.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text feature in the dataset (for Slovak data use 'Popis').",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of the label feature in the dataset (for Slovak data use 'Filter').",
    )
    parser.add_argument(
        "--filter_list_file",
        type=str,
        default=None,
        help="Optional JSON Lines file that lists all filters to enforce a fixed label vocabulary.",
    )
    parser.add_argument(
        "--filter_label_column",
        type=str,
        default="Filter",
        help="Column name inside the filter list file that contains label names.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Base model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device batch size for training and evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for the AdamW optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Where to store model checkpoints and logs.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["fp16", "bf16"],
        default=None,
        help="Enable mixed precision training with fp16 or bf16 (requires hardware support).",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Force training on CPU even if a GPU is available.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit the number of training examples for quick experiments.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limit the number of evaluation samples for quick experiments.",
    )
    parser.add_argument(
        "--predict_file",
        type=str,
        default=None,
        help="Optional JSON Lines file with unlabeled data to auto-assign filters.",
    )
    parser.add_argument(
        "--predictions_output",
        type=str,
        default=None,
        help="Where to save predicted filters for --predict_file (defaults to <output_dir>/predictions.jsonl).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        help="Evaluation metric to compute (any metric available through the 'evaluate' library).",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the fine-tuned model to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Repository name on the hub (required if --push_to_hub is set).",
    )
    return parser.parse_args()

def load_filter_labels(args: argparse.Namespace) -> Optional[list]:
    if not args.filter_list_file:
        return None

    filters = load_dataset("json", data_files={"filters": args.filter_list_file})["filters"]
    if args.filter_label_column not in filters.column_names:
        raise ValueError(
            f"Column '{args.filter_label_column}' not found in filter list file. "
            f"Available columns: {filters.column_names}"
        )

    labels = []
    seen = set()
    for value in filters[args.filter_label_column]:
        if value not in seen:
            labels.append(value)
            seen.add(value)
    return labels


def encode_label_column(
    raw_datasets: DatasetDict, label_column: str, label_names: list, splits_with_labels: set
) -> Dict[str, int]:
    label2id = {name: i for i, name in enumerate(label_names)}

    def map_labels(batch):
        try:
            encoded = [label2id[label] for label in batch[label_column]]
        except KeyError as exc:
            raise ValueError(
                f"Encountered unknown label '{exc.args[0]}' that is not in the provided filter list"
            ) from exc
        return {"labels": encoded}

    for split in splits_with_labels:
        raw_datasets[split] = raw_datasets[split].map(
            map_labels, batched=True, remove_columns=[label_column]
        )

    return label2id


def preprocess_datasets(
    tokenizer,
    raw_datasets: DatasetDict,
    text_column: str,
    max_length: int,
    label_column: str,
):
    def tokenize_function(batch):
        return tokenizer(batch[text_column], truncation=True, max_length=max_length)

    tokenized_splits = {}
    for split, dataset in raw_datasets.items():
        if text_column not in dataset.column_names:
            raise ValueError(
                f"Text column '{text_column}' is missing from split '{split}'. Adjust --text_column."
            )

        columns_to_remove = [
            col
            for col in dataset.column_names
            if col not in {text_column, label_column, "labels"}
        ]
        tokenized = dataset.map(
            tokenize_function, batched=True, remove_columns=columns_to_remove
        )

        if label_column in dataset.column_names:
            tokenized = tokenized.rename_column(label_column, "labels")

        tokenized_splits[split] = tokenized

    return DatasetDict(tokenized_splits)


def validate_columns(
    raw_datasets, text_column: str, label_column: str, split: str, require_label: bool = True
):
    missing_columns = set()
    if text_column not in raw_datasets[split].column_names:
        missing_columns.add(text_column)
    if require_label and label_column not in raw_datasets[split].column_names:
        missing_columns.add(label_column)

    if missing_columns:
        raise ValueError(
            f"Missing columns {missing_columns} in split '{split}'. "
            "Adjust --text_column and --label_column to match the dataset."
        )


def load_raw_datasets(args: argparse.Namespace) -> DatasetDict:
    if not args.dataset_name and not args.train_file:
        raise ValueError("Provide either --dataset_name or --train_file to load data.")

    if args.train_file:
        data_files = {"train": args.train_file}
        if args.eval_file:
            data_files["validation"] = args.eval_file
        if args.predict_file:
            data_files["predict"] = args.predict_file
        raw_datasets = load_dataset("json", data_files=data_files)
    else:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config)
        if args.predict_file:
            predict_ds = load_dataset("json", data_files={"predict": args.predict_file})[
                "predict"
            ]
            raw_datasets = DatasetDict(raw_datasets)
            raw_datasets["predict"] = predict_ds

    return raw_datasets


def infer_eval_split(raw_datasets, args: argparse.Namespace) -> str:
    if args.eval_split:
        return args.eval_split
    if "validation" in raw_datasets:
        return "validation"
    if "test" in raw_datasets:
        return "test"

    split = raw_datasets[args.train_split].train_test_split(
        test_size=args.validation_split_ratio, seed=args.seed
    )
    raw_datasets[args.train_split] = split["train"]
    raw_datasets["validation"] = split["test"]
    return "validation"


def prepare_data(args: argparse.Namespace) -> Tuple:
    raw_datasets = load_raw_datasets(args)

    if args.train_split not in raw_datasets:
        raise ValueError(
            f"Train split '{args.train_split}' not found. Available splits: {list(raw_datasets.keys())}"
        )
    validate_columns(raw_datasets, args.text_column, args.label_column, args.train_split)

    eval_split = infer_eval_split(raw_datasets, args)
    validate_columns(raw_datasets, args.text_column, args.label_column, eval_split)

    if "predict" in raw_datasets:
        validate_columns(
            raw_datasets, args.text_column, args.label_column, "predict", require_label=False
        )

    label_names = load_filter_labels(args)
    splits_with_labels = {
        split
        for split, dataset in raw_datasets.items()
        if args.label_column in dataset.column_names
    }

    label_feature = raw_datasets[args.train_split].features[args.label_column]
    if hasattr(label_feature, "names"):
        model_label_names = label_feature.names
        if label_names and len(label_names) != len(model_label_names):
            raise ValueError(
                "The filter list labels do not match the dataset's label set. "
                f"Dataset labels: {model_label_names}. Filter list labels: {label_names}."
            )
        label_names = label_names or model_label_names
        num_labels = len(label_names)
    else:
        if not label_names:
            label_names = raw_datasets[args.train_split].unique(args.label_column)
        encode_label_column(
            raw_datasets, args.label_column, label_names, splits_with_labels
        )
        num_labels = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = preprocess_datasets(
        tokenizer,
        raw_datasets,
        text_column=args.text_column,
        max_length=args.max_length,
        label_column=args.label_column,
    )

    if args.max_train_samples:
        tokenized["train"] = tokenized["train"].select(range(args.max_train_samples))
    if args.max_eval_samples and eval_split in tokenized:
        tokenized[eval_split] = tokenized[eval_split].select(range(args.max_eval_samples))

    model_config = ModelConfig(model_name=args.model_name, num_labels=num_labels)
    return tokenized, tokenizer, model_config, label_names, eval_split


def build_model(config: ModelConfig, label_names: Optional[list]):
    id2label = {i: name for i, name in enumerate(label_names)} if label_names else None
    label2id = {name: i for i, name in enumerate(label_names)} if label_names else None
    return AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        id2label=id2label,
        label2id=label2id,
    )


def compute_metrics_builder(metric_name: str = "accuracy"):
    metric = load(metric_name)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    return compute_metrics


def save_predictions(
    predict_dataset, predictions, label_names: Optional[list], output_path: Path, text_column: str
):
    logits = predictions.predictions
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    pred_ids = probs.argmax(axis=-1)
    confidences = probs.max(axis=-1)

    output_rows = []
    for text, pred_id, confidence in zip(
        predict_dataset[text_column], pred_ids, confidences
    ):
        label = label_names[pred_id] if label_names else str(pred_id)
        output_rows.append(
            {"predicted_filter": label, text_column: text, "confidence": float(confidence)}
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenized, tokenizer, model_config, label_names, eval_split = prepare_data(args)
    model = build_model(model_config, label_names)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    fp16 = args.mixed_precision == "fp16"
    bf16 = args.mixed_precision == "bf16"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        logging_steps=50,
        fp16=fp16,
        bf16=bf16,
        no_cuda=args.no_cuda,
    )

    compute_metrics = compute_metrics_builder(metric_name=args.metric)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[eval_split],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    trainer.save_model()

    if "predict" in tokenized:
        predictions_output = (
            Path(args.predictions_output)
            if args.predictions_output
            else Path(args.output_dir) / "predictions.jsonl"
        )
        predictions = trainer.predict(tokenized["predict"])
        save_predictions(
            tokenized["predict"],
            predictions,
            label_names,
            predictions_output,
            args.text_column,
        )
        print(f"Saved predicted filters to {predictions_output}")

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
