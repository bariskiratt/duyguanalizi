# Turkish E-Commerce Review Sentiment Analysis

Three-class sentiment classification for Turkish e-commerce product reviews, built by fine-tuning **BERTurk** (`dbmdz/bert-base-turkish-cased`) with a custom MLP classification head — and benchmarked against a TF-IDF baseline to show what the transformer actually buys you.

Beyond the classifier, the project includes an interpretability layer: **SHAP** attribution to explain *why* the model misclassifies the reviews it gets wrong, and **BERTopic** clustering to surface the recurring themes customers complain about.

## Results

Evaluated on **24,706** held-out reviews:

| Metric | TF-IDF + Logistic Regression | BERTurk + MLP |
| --- | --- | --- |
| Accuracy | 75.3% | **84.9%** |
| Macro F1 | — | **0.841** |
| Weighted F1 | — | 0.850 |

Per class:

| Class | Precision | Recall | F1 | Support | Baseline F1 |
| --- | --- | --- | --- | --- | --- |
| `negatif` | 0.818 | 0.816 | 0.817 | 8,049 | 0.819 |
| `pozitif` | 0.961 | 0.934 | **0.947** | 9,453 | 0.831 |
| `notr` | 0.743 | 0.772 | 0.758 | 7,204 | **0.402** |

**The neutral class is where the transformer earns its cost.** The TF-IDF baseline effectively could not model neutrality — 0.277 precision, meaning nearly three quarters of everything it called neutral was not. Fine-tuning lifts that class from 0.402 to 0.758 F1, and most of the overall accuracy gain comes from there.

The remaining error is concentrated in the same place: `negatif` and `notr` are mutually confusable, exchanging 1,386 and 1,374 samples in the confusion matrix. Reviews that pair a complaint with a concession ("kargo hızlıydı ama ürün beklediğim gibi değil") are the hard case, and they are exactly what SHAP surfaces.

> The two systems were evaluated on different splits — the baseline on 14,327 reviews with the original skewed class distribution (neutral only 9%), the transformer on 24,706 class-balanced reviews. Treat the comparison as directional, not as a controlled A/B.

Artifacts: `artifacts/bert_evaluation/` (metrics, confusion matrix, per-review predictions) and `artifacts/baseline/`.

## Labels

| ID | Label |
| --- | --- |
| 0 | `negatif` |
| 1 | `pozitif` |
| 2 | `notr` |

This ordering lives in `src/configs/bert_hparams.yaml` under `labels:` and is the single source of truth — data preparation, training and evaluation all read it, and `prepare_bert_data.py` writes a copy to `data/processed/label_mapping.json` so every artifact can be traced back to it. Training a binary model means deleting `notr` from that list and setting `num_classes: 2`; no code changes.

## Pipeline

```
raw reviews (parquet)
  └─ create_balanced_dataset.py   class balancing, NaN/empty removal, optional dedup
      └─ prepare_bert_data.py     BERTurk tokenization → HF Dataset on disk
          └─ bert_train.py        fine-tuning with early stopping
              └─ bert_evaluate.py metrics, confusion matrix, per-review predictions
                  ├─ shap_error.py        error attribution on misclassifications
                  └─ topics_bertopics.py  topic clustering over the corpus

src/models/baseline/train_baseline.py   TF-IDF + logistic regression reference point
```

## Model

A BERT backbone with a configurable MLP head (`src/models/bert/bert_mlp_classifier.py`) rather than the standard single linear classifier:

- **Backbone:** `dbmdz/bert-base-turkish-cased`, max sequence length 256
- **Pooling:** mean pooling over token embeddings, masked to ignore padding
- **Head:** one hidden layer of 384 units, GELU activation, dropout 0.25, Xavier initialization
- The head builder supports arbitrary hidden-layer stacks, ReLU/GELU/Tanh, and optional batch norm — batch norm is disabled here because a batch size of 16 is too small for stable statistics

### Training configuration

Defined in `src/configs/bert_hparams.yaml`:

| Parameter | Value | Reason |
| --- | --- | --- |
| Learning rate | 2e-5 | Low LR avoids catastrophic forgetting when fine-tuning a pretrained model |
| Batch size | 16 (eval 32) | |
| Gradient accumulation | 2 | Effective batch of 32 without the memory cost |
| Epochs | 3 | |
| Warmup steps | 1,440 | |
| Weight decay | 0.01 | |
| Mixed precision | fp16 | |
| Early stopping | patience 5, threshold 0.005 on `eval_f1` | |

Probability calibration uses Platt scaling on a 0.2 validation split.

### Baseline

`src/models/baseline/train_baseline.py` — TF-IDF vectorization (unigrams and bigrams) into multinomial logistic regression, with `GridSearchCV` hyperparameter search. It exists to answer "is the transformer worth it?" with a number rather than an assumption.

## Analysis layer

**`src/analysis/encoder.py`** — produces mean-pooled, L2-normalized sentence embeddings. Loads the fine-tuned checkpoint when one exists and falls back to the pretrained base model otherwise, so downstream analysis works before or after training.

**`src/analysis/topics_bertopics.py`** — BERTopic over those embeddings, with UMAP (25 neighbors, 5 components, cosine metric) for dimensionality reduction and HDBSCAN for clustering. Output: `reports/topic_report.html`.

**`src/analysis/shap_error.py`** — runs SHAP over misclassified reviews to identify which tokens drove the wrong prediction, turning aggregate error rates into specific, inspectable failure patterns.

## Repository structure

```
src/
├── data/
│   ├── create_balanced_dataset.py   CLI: class balancing and cleaning
│   └── prepare_bert_data.py         tokenization → HF Dataset
├── models/
│   ├── bert/
│   │   ├── bert_mlp_classifier.py   BERT + MLP head architecture
│   │   ├── bert_train.py            fine-tuning loop
│   │   └── bert_evaluate.py         metrics and prediction dump
│   └── baseline/
│       └── train_baseline.py        TF-IDF + logistic regression baseline
├── analysis/
│   ├── encoder.py                   sentence embeddings
│   ├── topics_bertopics.py          BERTopic clustering
│   └── shap_error.py                SHAP error attribution
└── configs/bert_hparams.yaml        hyperparameters + label vocabulary

configs/schema.json                  expected input data schema
artifacts/bert_evaluation/           transformer metrics and predictions
artifacts/baseline/                  baseline metrics
reports/topic_report.html            generated topic report
```

## Input data schema

Defined in `configs/schema.json`:

| Field | Type |
| --- | --- |
| `review_id` | string |
| `product_id` | string |
| `review_date` | datetime |
| `review_text` | string |
| `star_rating` | int |
| `label` | string |

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended for training. The scripts detect CUDA automatically and fall back to CPU.

## Usage

Run from the repository root — the scripts resolve config and data paths relative to it.

```bash
# 1. Balance and clean the raw reviews
python src/data/create_balanced_dataset.py \
    --input data/raw/reviews.parquet \
    --output_dir data/processed/train_balanced \
    --drop_duplicates

# 2. Tokenize into Hugging Face datasets
python src/data/prepare_bert_data.py

# 3. Fine-tune
python src/models/bert/bert_train.py

# 4. Evaluate
python src/models/bert/bert_evaluate.py

# Optional: baseline reference point
python src/models/baseline/train_baseline.py

# Optional: interpretability
python src/analysis/topics_bertopics.py
python src/analysis/shap_error.py
```

## Notes

- **Datasets are not committed.** `.gitignore` excludes parquet and CSV data, so `data/raw/` and the intermediate `data/processed/*.parquet` files must be supplied to run the pipeline end to end. The tokenized Hugging Face datasets under `data/processed/bert_train` and `bert_val` are checked in.
- **The published metrics predate the config consolidation.** They were produced by a 3-class run whose label mapping now lives in `bert_hparams.yaml`; the evaluation artifacts from that run label their columns `LABEL_0/1/2`, which correspond to the table above.
