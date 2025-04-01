from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from datasets import load_dataset
import evaluate

# Loading dataset
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Preparing data with three-class labels from hatespeech column
train_texts = train_dataset["text"]
train_labels = train_dataset["hatespeech"]
test_texts = test_dataset["text"]
test_labels = test_dataset["hatespeech"]

# Converting text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features for efficiency
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Training a logistic regression classifier (multi-class)
classifier = LogisticRegression(max_iter=1000, multi_class='multinomial')
classifier.fit(X_train, train_labels)

# Predicting on test set
bow_predictions = classifier.predict(X_test)

# Evaluating the model
metric = evaluate.load("accuracy")
bow_accuracy = metric.compute(predictions=bow_predictions, references=test_labels)
print(f"BoW (Logistic Regression) Accuracy (3-class): {bow_accuracy['accuracy']:.3f}")

# Majority Class Baseline
# Finding the most common label in the training set
label_counts = np.bincount(train_labels, minlength=3)  # Counts for 0, 1, 2
majority_label = np.argmax(label_counts)  # Most frequent: 0, 1, or 2
majority_predictions = [majority_label] * len(test_labels)
majority_accuracy = metric.compute(predictions=majority_predictions, references=test_labels)
print(f"Majority Class Baseline Accuracy (3-class): {majority_accuracy['accuracy']:.3f}")

# Random Baseline
# Randomly predicting 0, 1, or 2 with equal probability (1/3 each)
np.random.seed(42)
random_predictions = np.random.choice([0, 1, 2], size=len(test_labels), p=[1/3, 1/3, 1/3])
random_accuracy = metric.compute(predictions=random_predictions, references=test_labels)
print(f"Random Baseline Accuracy (3-class): {random_accuracy['accuracy']:.3f}")