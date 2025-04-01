from datasets import load_dataset
import numpy as np

# Load dataset
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Discretize hatespeech labels into 0, 1, 2
train_labels = [0 if label < 0.5 else 1 if label <= 1.5 else 2 for label in train_dataset["hatespeech"]]
test_labels = [0 if label < 0.5 else 1 if label <= 1.5 else 2 for label in test_dataset["hatespeech"]]

# Compute sizes and class distributions
train_size = len(train_dataset)
test_size = len(test_dataset)
train_class_counts = np.bincount(train_labels, minlength=3)
test_class_counts = np.bincount(test_labels, minlength=3)

# Compute average text length (in words)
train_text_lengths = [len(text.split()) for text in train_dataset["text"]]
test_text_lengths = [len(text.split()) for text in test_dataset["text"]]
avg_train_length = np.mean(train_text_lengths)
avg_test_length = np.mean(test_text_lengths)

# Print table
print("Dataset Split Sizes and Class Distribution:")
print("| Split  | Size   | Not-Hateful (0) | Neutral (1) | Hateful (2) |")
print("|--------|--------|-----------------|-------------|-------------|")
print(f"| Train  | {train_size:5d} | {train_class_counts[0]:15d} | {train_class_counts[1]:11d} | {train_class_counts[2]:11d} |")
print(f"| Test   | {test_size:5d} | {test_class_counts[0]:15d} | {test_class_counts[1]:11d} | {test_class_counts[2]:11d} |")

# Print other features
print("\nOther Dataset Features:")
print(f"Average text length (words) - Train: {avg_train_length:.2f}, Test: {avg_test_length:.2f}")
print("Languages: Primarily English (based on dataset documentation and inspection)")