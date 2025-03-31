from transformers import pipeline
from datasets import load_dataset
import evaluate

# Loading the hate speech dataset
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")

if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
test_dataset = dataset["test"]

pipe_fb = pipeline(model="facebook/bart-large-mnli")
pipe_moritz = pipeline(model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

text1 = "I feel that business majors are retarded - Is this statement toxic? Answer yes or no"

labels = ["hateful", "not-hateful", "neutral"]


def predict_with_pipeline(pipe, text):
    prompt_text = f"Is this text hateful: {text}"
    result = pipe(prompt_text, candidate_labels=labels)
    scores = result["scores"]  # [P(yes), P(no)]
    return 1 if scores[0] > scores[1] else 0


predictions_fb = []
predictions_moritz = []

true_labels = [1 if label >= 1 else 0 for label in test_dataset["hatespeech"]]

for text in test_dataset["text"]:
    pred_moritz = predict_with_pipeline(pipe_moritz, text)
    pred_fb = predict_with_pipeline(pipe_fb, text)
    predictions_moritz.append(pred_moritz)
    predictions_fb.append(pred_fb)

diff_count = sum(
    1 for fb, moritz in zip(predictions_fb, predictions_moritz) if fb != moritz
)
print(f"Number of differing predictions: {diff_count} out of {len(predictions_fb)}")

# Evaluate
metric = evaluate.load("accuracy")
accuracy_moritz = metric.compute(predictions=predictions_moritz, references=true_labels)
accuracy_fb = metric.compute(predictions=predictions_fb, references=true_labels)

print(f"RoBERTa-large-mnli Accuracy: {accuracy_moritz['accuracy']:.3f}")
print(f"BART-large-mnli Accuracy: {accuracy_fb['accuracy']:.3f}")
