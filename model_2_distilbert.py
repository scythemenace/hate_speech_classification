from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import evaluate  # Add this import
import numpy as np  # Required for np.argmax

# Load the dataset
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")

# Split the dataset
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


# Define tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Tokenize the datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)


# Convert 'hatespeech' to integer labels
def convert_labels(examples):
    examples["labels"] = [int(label) for label in examples["hatespeech"]]
    return examples


tokenized_train = tokenized_train.map(convert_labels, batched=True)
tokenized_test = tokenized_test.map(convert_labels, batched=True)

# Remove unnecessary columns and set format
tokenized_train = tokenized_train.remove_columns(
    [
        col
        for col in tokenized_train.column_names
        if col not in ["input_ids", "attention_mask", "labels"]
    ]
)
tokenized_test = tokenized_test.remove_columns(
    [
        col
        for col in tokenized_test.column_names
        if col not in ["input_ids", "attention_mask", "labels"]
    ]
)

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=3, torch_dtype="auto"
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="test_trainer_2",
    eval_strategy="epoch",
)

# Load the accuracy metric
metric = evaluate.load("accuracy")


# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get the predicted class (0 or 1)
    return metric.compute(predictions=predictions, references=labels)


# Initialize the Trainer with compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
