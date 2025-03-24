import datasets

from transformers import AutoTokenizer

dataset = datasets.load_dataset("ucberkeley-dlab/measuring-hate-speech", "default")

# dataset["train"][100]

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
