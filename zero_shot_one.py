import numpy as np
from transformers import pipeline
from datasets import load_dataset
import evaluate

# Load dataset
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
test_dataset = dataset["test"]

# Initialize models
pipe_fb = pipeline(model="facebook/bart-large-mnli")
pipe_moritz = pipeline(model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# Define the prompt
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

# Ground truth labels (keep as 0, 1, 2)
true_labels = [0 if label == 0 else 1 if label == 1 else 2 for label in test_dataset["hatespeech"]]

# Evaluate on test set
preds_fb, scores_fb = evaluate_model(pipe_fb, prompt_template, test_dataset)
preds_moritz, scores_moritz = evaluate_model(pipe_moritz, prompt_template, test_dataset)

# Map predictions to match true labels (0: not-hateful, 1: neutral, 2: hateful)
# labels = ["hateful", "not-hateful", "neutral"] -> map to [2, 0, 1]
label_mapping = {0: 2, 1: 0, 2: 1}  # hateful -> 2, not-hateful -> 0, neutral -> 1
mapped_preds_fb = [label_mapping[p] for p in preds_fb]
mapped_preds_moritz = [label_mapping[p] for p in preds_moritz]

# Compute accuracy
metric = evaluate.load("accuracy")
accuracy_fb = metric.compute(predictions=mapped_preds_fb, references=true_labels)
accuracy_moritz = metric.compute(predictions=mapped_preds_moritz, references=true_labels)

print(f"BART Test Accuracy (3-class): {accuracy_fb['accuracy']:.3f}")
print(f"DeBERTa Test Accuracy (3-class): {accuracy_moritz['accuracy']:.3f}")

# Check prediction differences and score variation
diff_count = sum(1 for fb, moritz in zip(mapped_preds_fb, mapped_preds_moritz) if fb != moritz)
print(f"Number of differing predictions: {diff_count} out of {len(mapped_preds_fb)}")

mean_score_diff = np.mean([abs(fb[0] - moritz[0]) for fb, moritz in zip(scores_fb, scores_moritz)])
print(f"Mean difference in 'hateful' scores: {mean_score_diff:.3f}")

# Sample outputs for inspection
print("\nSample results for first 3 examples:")
for i in range(min(3, len(test_dataset))):
    print(f"Text: {test_dataset['text'][i]}")
    print(f"BART scores: {scores_fb[i]} -> Pred: {labels[preds_fb[i]]} (Mapped: {mapped_preds_fb[i]})")
    print(f"DeBERTa scores: {scores_moritz[i]} -> Pred: {labels[preds_moritz[i]]} (Mapped: {mapped_preds_moritz[i]})")