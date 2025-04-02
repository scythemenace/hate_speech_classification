[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vQMaLvtr)

# Hate Speech Classification Project

This project performs hate speech classification on the `ucberkeley-dlab/measuring-hate-speech` dataset, categorizing social media comments into three classes: not-hateful (0), neutral (1), and hateful (2). The project includes fine-tuned transformer models, zero-shot models, and baseline classifiers, with results reported in `Report.pdf`.

## Files and Directories

- **Report.pdf**: The final report for this assignment, detailing the task, methodology, results, and analysis.
- **baseline.py**: Computes the Bag-of-Words (BoW) classifier with TF-IDF features, majority class baseline, and random baseline for the three-class task.
- **bert_output.txt**: Contains the accuracy output from `model_1_bert.py`, which fine-tunes the `bert-base-uncased` model.
- **bow_and_baseline.txt**: Contains the accuracy outputs from `baseline.py` for the BoW classifier, majority class baseline, and random baseline.
- **distilbert_output.txt**: Contains the accuracy output from `model_2_distilbert.py`, which fine-tunes the `distilbert-base-uncased` model.
- **model_1_bert.py**: Code to fine-tune the `bert-base-uncased` model for the three-class hate speech classification task.
- **model_2_distilbert.py**: Code to fine-tune the `distilbert-base-uncased` model for the same task.
- **requirements.txt**: Lists the Python libraries required for this project, installable via `pip`.
- **statistics.py**: Prints dataset statistics, including the size of training and test splits and the number of items per class (not-hateful, neutral, hateful) in each split.
- **zero_shot_accuracy.txt**: Contains the accuracy outputs for the two zero-shot models (`facebook/bart-large-mnli` and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`).
- **zero_shot_one.py**: Code that uses the zero-shot models to compute their accuracies on the test set.

## Setup Instructions

1. **Prerequisites**:

   - Ensure Python 3 is installed on your system.
   - A GPU is required to run the fine-tuning (`model_1_bert.py`, `model_2_distilbert.py`) and zero-shot (`zero_shot_one.py`) scripts due to their heavy computational requirements.

2. **Install Dependencies**:

   - Install the required libraries listed in `requirements.txt` by running:
     ```
     pip install -r requirements.txt
     ```
   - This includes libraries like `transformers`, `datasets`, `scikit-learn`, `evaluate`, and `numpy`.

3. **Verify GPU Setup**:
   - To monitor GPU usage while running the fine-tuning or zero-shot scripts, set the `CUDA_LAUNCH_BLOCKING` environment variable:
     ```
     CUDA_LAUNCH_BLOCKING=1 python3 <file_name> &
     ```
   - This ensures proper GPU utilization and helps debug CUDA-related issues.

## How to Run the Code

- **General Instructions**:

  - To run any Python script, use the command:
    ```
    python3 <file_name>
    ```
  - For scripts requiring a GPU (`model_1_bert.py`, `model_2_distilbert.py`, `zero_shot_one.py`), use the GPU-enabled command above.

- **Dataset Statistics**:

  - Run `statistics.py` to get information about the dataset splits and class distribution:
    ```
    python3 statistics.py
    ```

- **Fine-Tuned Models**:

  - Run `model_1_bert.py` to fine-tune `bert-base-uncased` and output accuracy to `bert_output.txt`:
    ```
    CUDA_LAUNCH_BLOCKING=1 python3 model_1_bert.py &
    ```
  - Run `model_2_distilbert.py` to fine-tune `distilbert-base-uncased` and output accuracy to `distilbert_output.txt`:
    ```
    CUDA_LAUNCH_BLOCKING=1 python3 model_2_distilbert.py &
    ```

- **Zero-Shot Models**:

  - Run `zero_shot_one.py` to compute accuracies for the zero-shot models (`facebook/bart-large-mnli` and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) and output to `zero_shot_accuracy.txt`:
    ```
    CUDA_LAUNCH_BLOCKING=1 python3 zero_shot_one.py &
    ```

- **Baselines**:
  - Run `baseline.py` to compute the BoW classifier, majority class baseline, and random baseline, with results output to `bow_and_baseline.txt`:
    ```
    python3 baseline.py
    ```
  - This script does not require a GPU and can run on a CPU.

## Notes

- The fine-tuning and zero-shot scripts are computationally intensive and may take several hours on a GPU (e.g., Tesla V100). Ensure your system has sufficient resources.
- The `statistics.py` script provides crucial dataset information, such as split sizes and class distributions, which are useful for understanding the task’s challenges.
