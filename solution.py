from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

# Load the dataset
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")

# Split the dataset (assuming no test split exists)
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


# Define tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Tokenize the datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)


# Rename 'hatespeech' to 'labels' and convert to integer
def convert_labels(examples):
    # Ensure labels are integers (0 or 1) instead of floats (0.0 or 1.0)
    examples["labels"] = [int(label) for label in examples["hatespeech"]]
    return examples


tokenized_train = tokenized_train.map(convert_labels, batched=True)
tokenized_test = tokenized_test.map(convert_labels, batched=True)

# Remove unnecessary columns and set format for PyTorch
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
    "google-bert/bert-base-uncased", num_labels=2, torch_dtype="auto"
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

# Start training
trainer.train()
