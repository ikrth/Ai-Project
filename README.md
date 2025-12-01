# TF-IDF + Logistic Regression Baseline (MWAHAHA Humor Generation)

This branch implements **Baseline A** for Assignment 2:
a TF-IDF + Logistic Regression model used to score and rank template-based jokes
for the SemEval 2026 MWAHAHA humor generation task (Subtask A, English).

## Files

- `model_baseline_A.py` – TF-IDF + Logistic Regression model (train + score + save/load).
- `train_baseline_A.py` – trains the classifier on a small joke vs non-joke dataset.
- `generate_baseline_A.py` – interactive joke generator (word pair or headline).
- `evaluate_baseline_A.py` – generates jokes for `task-a-en.tsv` and computes simple metrics.
- `requirements.txt` – Python dependencies for this baseline.

## How to run

```bash
# install dependencies
pip install -r requirements.txt

# train the baseline model
python train_baseline_A.py

# run interactive joke generator
python generate_baseline_A.py

# evaluate on SemEval dev inputs (task-a-en.tsv)
python evaluate_baseline_A.py
