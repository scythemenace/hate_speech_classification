import datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

dataset = datasets.load_dataset("ucberkeley-dlab/measuring-hate-speech", "default")

# dataset["train"][100]

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_datasets.rename_column("hatespeech", "labels")

tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-uncased", num_labels=2, torch_dtype="auto"
)

training_args = TrainingArguments(output_dir="test_trainer")
