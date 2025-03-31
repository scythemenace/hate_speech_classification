import numpy as np
from transformers import pipeline
from datasets import load_dataset
import evaluate

# Load the hate speech dataset
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
test_dataset = dataset["test"]

# Initialize the zero-shot pipelines
pipe_fb = pipeline(model="facebook/bart-large-mnli")
pipe_moritz = pipeline(model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# Define candidate labels (simplified to binary for clarity)
labels = ["This text is hateful", "This text is not hateful"]


# Prediction function for zero-shot pipeline
def predict_with_pipeline(pipe, text):
    result = pipe(text, candidate_labels=labels)
    # Return index of highest score (0 for hateful, 1 for not hateful)
    return np.argmax(result["scores"])


# Generate predictions for both models
predictions_fb = [predict_with_pipeline(pipe_fb, text) for text in test_dataset["text"]]
predictions_moritz = [
    predict_with_pipeline(pipe_moritz, text) for text in test_dataset["text"]
]

# Prepare ground truth labels (binary: 1 for hateful, 0 for not hateful)
true_labels = [1 if label >= 1 else 0 for label in test_dataset["hatespeech"]]

# Load the accuracy metric
metric = evaluate.load("accuracy")


# Define the compute_metrics function (adapted for zero-shot)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Predictions are already class indices (0 or 1), no argmax needed
    return metric.compute(predictions=predictions, references=labels)


# Evaluate both models using compute_metrics
eval_pred_fb = (predictions_fb, true_labels)
eval_pred_moritz = (predictions_moritz, true_labels)

accuracy_fb = compute_metrics(eval_pred_fb)
accuracy_moritz = compute_metrics(eval_pred_moritz)

print(f"BART-large-mnli Accuracy: {accuracy_fb['accuracy']:.3f}")
print(f"DeBERTa-v3-base-mnli Accuracy: {accuracy_moritz['accuracy']:.3f}")

# Optional: Check differing predictions to debug identical accuracies
diff_count = sum(
    1 for fb, moritz in zip(predictions_fb, predictions_moritz) if fb != moritz
)
print(f"Number of differing predictions: {diff_count} out of {len(predictions_fb)}")
