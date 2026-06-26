# Fake News Detection 📰

> NLP classifier that distinguishes **real** news articles from **fake** ones using TF-IDF
> features and classic supervised-learning models.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## Overview

Misinformation spreads fast, and automatically flagging fabricated articles is a classic
text-classification problem. This project builds a full NLP pipeline — text cleaning →
vectorization → model training → manual testing — to classify a news article as **true** or **fake**.

## Dataset

- Two labelled corpora: `True.csv` (genuine articles) and `Fake.csv` (fabricated articles),
  combined and tagged with a binary `class` target.
- A separate `manual_testing.csv` holds held-out samples used to sanity-check predictions on
  unseen text.

## Approach

1. **Text preprocessing** — lower-casing, and removal of punctuation, URLs, and noise using
   `re` and `string`.
2. **Feature extraction** — `TfidfVectorizer` converts cleaned text into TF-IDF feature vectors.
3. **Modelling** — four classifiers are trained and compared:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
4. **Evaluation** — accuracy and a full `classification_report` (precision / recall / F1).
5. **Manual testing** — a helper function runs all trained models on a freshly typed headline /
   article and reports each model's verdict.

## Results

The models score very highly on the held-out test split:

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ≈ 0.99 |
| Decision Tree / Random Forest / Gradient Boosting | ≈ 0.99 – 1.00 |

> **Honest caveat:** the True/Fake corpora come from different sources, so part of this near-perfect
> score reflects stylistic/source artifacts rather than pure "truthfulness." The `manual_testing`
> step is included precisely to probe how the models behave on text outside the training distribution.

## Tech Stack

`Python` · `scikit-learn` · `TF-IDF` · `Pandas` · `NumPy` · `Seaborn` · `Matplotlib`

## How to Run

```bash
git clone https://github.com/Tirth1411/-Fake-News-Detection.git
cd -Fake-News-Detection
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter notebook fake-news-detection.ipynb
```

## Repository Structure

```
fake-news-detection.ipynb     # preprocessing, TF-IDF, model training & manual testing
fake-news-detection/          # True.csv, Fake.csv
manual_testing.csv            # held-out samples for manual prediction checks
```
