import numpy as np
from transformers import pipeline
from datasets import load_dataset
import evaluate

# Load dataset
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Initialize models
pipe_fb = pipeline(model="facebook/bart-large-mnli")
pipe_moritz = pipeline(model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# Define the prompt (editable)
prompt_template = "Classify the text as hateful, not-hateful or neutral: {text}"

# Define labels based on test set
labels = ["hateful", "not-hateful", "neutral"]


# Function to get predictions and scores
def evaluate_model(pipe, prompt_template, dataset):
    preds, scores = [], []
    for text in dataset["text"]:
        prompt = prompt_template.format(text=text)
        result = pipe(prompt, candidate_labels=labels)
        pred = np.argmax(result["scores"])  # 0: hateful, 1: not-hateful, 2: neutral
        preds.append(pred)
        scores.append(result["scores"])
    return preds, scores


# Ground truth labels (binary for hateful vs not, neutral treated as not-hateful)
true_labels = [1 if label >= 1 else 0 for label in test_dataset["hatespeech"]]

# Evaluate on test set
preds_fb, scores_fb = evaluate_model(pipe_fb, prompt_template, test_dataset)
preds_moritz, scores_moritz = evaluate_model(pipe_moritz, prompt_template, test_dataset)

# Convert predictions to binary (hateful=1, not-hateful/neutral=0) for accuracy
binary_preds_fb = [1 if p == 0 else 0 for p in preds_fb]
binary_preds_moritz = [1 if p == 0 else 0 for p in preds_moritz]

# Compute accuracy
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)


print(
    f"BART Test Accuracy: {compute_metrics((binary_preds_fb, true_labels))['accuracy']:.3f}"
)
print(
    f"DeBERTa Test Accuracy: {compute_metrics((binary_preds_moritz, true_labels))['accuracy']:.3f}"
)

# Check prediction differences and score variation
diff_count = sum(
    1 for fb, moritz in zip(binary_preds_fb, binary_preds_moritz) if fb != moritz
)
print(
    f"Number of differing binary predictions: {diff_count} out of {len(binary_preds_fb)}"
)

mean_score_diff = np.mean(
    [abs(fb[0] - moritz[0]) for fb, moritz in zip(scores_fb, scores_moritz)]
)
print(f"Mean difference in 'hateful' scores: {mean_score_diff:.3f}")

# Sample outputs for inspection
print("\nSample results for first 3 examples:")
for i in range(min(3, len(test_dataset))):
    print(f"Text: {test_dataset['text'][i]}")
    print(f"BART scores: {scores_fb[i]} -> Pred: {labels[preds_fb[i]]}")
    print(f"DeBERTa scores: {scores_moritz[i]} -> Pred: {labels[preds_moritz[i]]}")
