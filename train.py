"""Fine-tune a text classification model using Hugging Face Transformers.

This script demonstrates a minimal end-to-end workflow for sequence
classification with BERT (or any compatible encoder) on datasets from the
🤗 Datasets hub. It includes configurable hyperparameters, evaluation, and
model saving.
"""
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

from datasets import load_dataset
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
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imdb",
        help="Dataset name from the 🤗 Hub (e.g., 'imdb' or 'ag_news').",
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
        "--validation_split_ratio",
        type=float,
        default=0.1,
        help="Fraction of the training split to reserve for validation if no eval split exists.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text feature in the dataset.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of the label feature in the dataset.",
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
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit the number of training examples for quick experiments.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limit the number of evaluation examples for quick experiments.",
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


def get_label_info(dataset, label_column: str) -> Tuple[int, Optional[list]]:
    label_feature = dataset["train"].features[label_column]
    if hasattr(label_feature, "names"):
        label_names = label_feature.names
        num_labels = len(label_names)
    else:
        label_names = None
        num_labels = dataset["train"].features[label_column].num_classes
    return num_labels, label_names


def preprocess_datasets(
    tokenizer,
    raw_datasets,
    text_column: str,
    max_length: int,
    label_column: str,
    train_split: str,
):
    def tokenize_function(batch):
        return tokenizer(batch[text_column], truncation=True, max_length=max_length)

    # Retain only the text and label columns before tokenization to avoid unused fields.
    columns_to_remove = [
        col
        for col in raw_datasets[train_split].column_names
        if col not in {text_column, label_column}
    ]
    tokenized = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=columns_to_remove
    )
    tokenized = tokenized.rename_column(label_column, "labels")
    return tokenized


def validate_columns(raw_datasets, text_column: str, label_column: str, split: str):
    missing_columns = {
        column
        for column in (text_column, label_column)
        if column not in raw_datasets[split].column_names
    }
    if missing_columns:
        raise ValueError(
            f"Missing columns {missing_columns} in split '{split}'. "
            "Adjust --text_column and --label_column to match the dataset."
        )


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
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config)

    if args.train_split not in raw_datasets:
        raise ValueError(
            f"Train split '{args.train_split}' not found. Available splits: {list(raw_datasets.keys())}"
        )
    validate_columns(raw_datasets, args.text_column, args.label_column, args.train_split)

    eval_split = infer_eval_split(raw_datasets, args)
    validate_columns(raw_datasets, args.text_column, args.label_column, eval_split)

    num_labels, label_names = get_label_info(raw_datasets, args.label_column)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = preprocess_datasets(
        tokenizer,
        raw_datasets,
        text_column=args.text_column,
        max_length=args.max_length,
        label_column=args.label_column,
        train_split=args.train_split,
    )

    if args.max_train_samples:
        tokenized[args.train_split] = tokenized[args.train_split].select(
            range(args.max_train_samples)
        )
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


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenized, tokenizer, model_config, label_names, eval_split = prepare_data(args)
    model = build_model(model_config, label_names)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

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

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
