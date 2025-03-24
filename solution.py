# Import necessary libraries
from datasets import load_dataset  # Correct import for dataset loading
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,  # Added for training
)

# Load the dataset
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")

# Check the available splits (this dataset typically has only 'train')
print("Available splits:", dataset.keys())

# If no test split exists, manually split the 'train' set
# Let's assume we split 80% train, 20% test if no predefined test set
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
else:
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

# Rename 'hatespeech' to 'labels' for both splits
tokenized_train = tokenized_train.rename_column("hatespeech", "labels")
tokenized_test = tokenized_test.rename_column("hatespeech", "labels")

# Set format for PyTorch
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-uncased", num_labels=2, torch_dtype="auto"
)

training_args = TrainingArguments(output_dir="test_trainer")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

trainer.train()
