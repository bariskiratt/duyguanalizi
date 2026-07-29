# Turkish E-Commerce Review Sentiment Analysis

Three-class sentiment classification for Turkish e-commerce product reviews, built by fine-tuning **BERTurk** (`dbmdz/bert-base-turkish-cased`) with a custom MLP classification head — and benchmarked against a TF-IDF baseline to show what the transformer actually buys you.

Beyond the classifier, the project includes an interpretability layer: **SHAP** attribution to explain *why* the model misclassifies the reviews it gets wrong, and **BERTopic** clustering to surface the recurring themes customers complain about.

**Model:** [huggingface.co/bariskirat/duyguanalizi-berturk](https://huggingface.co/bariskirat/duyguanalizi-berturk)

## Results

Evaluated on **65,000** held-out reviews, after removing the label leak described below:

| Metric | Value |
| --- | --- |
| Accuracy | **0.8224** |
| Macro F1 | **0.8263** |
| Weighted F1 | 0.8214 |

Per class:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| `negatif` | 0.851 | 0.880 | **0.865** | 20,000 |
| `pozitif` | 0.800 | 0.866 | **0.831** | 18,000 |
| `notr` | 0.816 | 0.751 | **0.782** | 27,000 |

Confusion matrix (rows are truth):

| | negatif | pozitif | notr |
| --- | --- | --- | --- |
| **negatif** | 17,596 | 192 | 2,212 |
| **pozitif** | 58 | 15,580 | 2,362 |
| **notr** | 3,013 | 3,706 | 20,281 |

**The model does not make polarity errors.** Negative-as-positive and positive-as-negative
together account for 250 of 65,000 predictions. Every meaningful error runs along the
neutral axis, which is the right failure mode to have: showing a review as undecided is
far less damaging than showing it as the opposite of what it says.

`notr` is the weakest class and that is inherent — neutral reviews sit linguistically
between the other two, and most of them pair a complaint with a concession
("kargo hızlıydı ama ürün beklediğim gibi değil").

A 100-review manual audit is in [`reports/`](reports/): 81/100 correct, and of the 19
errors only **2** are unambiguous model failures. In 9 the label contradicts the text —
`"süper kargo süper ses kalitesi süpersüpersüper"` is labelled neutral. Labels derive
from star ratings rather than the text, so measured accuracy has a ceiling set by label
quality, not by the model.

> **No untouched test split exists.** The pipeline produces only `bert_train` and
> `bert_val`, and checkpoint selection (`load_best_model_at_end`) ran against the same
> validation set, so these numbers are mildly optimistic. Carve out a third split before
> quoting them as a final result.

> **The TF-IDF baseline is not currently comparable.** `artifacts/baseline/` was produced
> on the older, leaky dataset and has not been rerun since. Rerun `train_baseline.py` on
> the current data before putting the two side by side.

### Label leakage: star ratings written into the review text

The source data prepends the star rating to the review body — `"Beş Yıldız Kokusunu ve
yoğunluğunu beğendim..."`, `"Üç Yıldız Güzeldi"`. A quarter of all reviews carry this
prefix and it determines the label almost perfectly: 1–2 stars are negative 100% of the
time, 3 stars neutral 99.7%, 4–5 stars positive 98%+.

For those rows the model was not classifying sentiment, it was reading the answer out of
the input. Splitting an earlier 0.8725 macro F1 run by whether the prefix is present:

| Subset | Accuracy |
| --- | --- |
| With star prefix (25.7%) | 100.0% |
| Without prefix (74.3%) | 81.9% |

Re-scoring those same weights on a prefix-stripped validation set gave 0.7966 — the leak
was worth 7.6 points. `create_balanced_dataset.py` now strips the prefix
(`--keep_star_prefix` opts out); only the pattern at the very start is removed, so
mid-sentence uses like `"10 numara 5 yıldız"` survive as the reviewer's own words. The
raw parquet is not in the repo, so `rebuild_without_star_prefix.py` regenerates the
tokenized datasets from what is committed.

Retraining on clean data scored **0.8263**, three points above what the leaky model
managed once its crutch was taken away.

### Preprocessing parity: punctuation

Training text has almost no punctuation — a comma appears 0.7 times per 10,000
characters against roughly 150 in ordinary Turkish. The model therefore never learned to
ignore it, and punctuation flipped predictions outright: `"Berbat, param çöpe gitti"`
came back **positive with 0.99 confidence**.

`src/inference/predictor.py` normalizes input into the training corpus's shape. Measured
on 100 reviews rewritten the way a person would actually type them:

| Input | Accuracy |
| --- | --- |
| Corpus form | 81/100 |
| User punctuation, normalized | **81/100** |
| User punctuation, raw | 68/100 |

Normalization closes the gap exactly. Without it the model loses 13 points the moment a
real person types into it.

Artifacts: `artifacts/bert_evaluation/` (metrics, confusion matrix, per-review predictions).

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
  └─ create_balanced_dataset.py   class balancing, star-prefix stripping, cleaning
      └─ prepare_bert_data.py     BERTurk tokenization → HF Dataset on disk
          └─ bert_train.py        fine-tuning, MPS/CUDA, resumes from last checkpoint
              ├─ bert_evaluate.py metrics, confusion matrix, per-review predictions
              ├─ predictor.py     single-review inference  →  app.py  (Gradio)
              ├─ shap_error.py         error attribution on misclassifications
              └─ topics_bertopics.py   topic clustering over the corpus

src/data/rebuild_without_star_prefix.py   re-derives the tokenized datasets when the
                                          raw parquet is unavailable
src/models/baseline/train_baseline.py     TF-IDF + logistic regression reference point
```

Deployment lives in [DEPLOY.md](DEPLOY.md); [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb)
runs the training half on a Colab GPU.

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
app.py                               Gradio interface (Hugging Face Spaces entry point)
DEPLOY.md                            train → Hub → Space walkthrough
requirements.txt                     full pinned environment
requirements-serve.txt               inference only, no bertopic/shap/umap

src/
├── data/
│   ├── create_balanced_dataset.py   CLI: balancing, cleaning, star-prefix stripping
│   ├── prepare_bert_data.py         tokenization → HF Dataset
│   └── rebuild_without_star_prefix.py  de-leak already-tokenized datasets
├── models/
│   ├── bert/
│   │   ├── bert_mlp_classifier.py   BERT + MLP head architecture
│   │   ├── bert_train.py            fine-tuning loop
│   │   └── bert_evaluate.py         metrics and prediction dump
│   └── baseline/
│       └── train_baseline.py        TF-IDF + logistic regression baseline
├── inference/
│   └── predictor.py                 serving path: rebuild arch, load weights, normalize
├── analysis/
│   ├── encoder.py                   sentence embeddings
│   ├── topics_bertopics.py          BERTopic clustering
│   └── shap_error.py                SHAP error attribution
└── configs/bert_hparams.yaml        hyperparameters + label vocabulary

notebooks/train_colab.ipynb          GPU training on Colab
configs/schema.json                  expected input data schema
artifacts/bert_mlp_ckpt/best_model/  trained weights (gitignored, ~423 MB)
artifacts/bert_evaluation/           transformer metrics and predictions
reports/                             topic report, manual audit
```

### Loading the trained model

The checkpoint is a plain `nn.Module` state dict, not a `PreTrainedModel`, so no
`config.json` is written and `AutoModelForSequenceClassification.from_pretrained` fails
on it with an unrecognized-`model_type` error. Rebuild the architecture from
`bert_hparams.yaml` and load the weights — `src/inference/predictor.py` does this, and
`pooling_strategy` must match training or the state dict loads cleanly and the model
silently runs on features it was never trained on.

```python
from src.inference.predictor import SentimentPredictor

p = SentimentPredictor()                       # or ckpt_dir=..., or $SENTIMENT_CKPT
p.predict("kargo hızlıydı ama ürün beklediğim gibi değil")
# {'label': 'negatif', 'confidence': 0.61, 'scores': {...}}
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

CUDA, Apple Silicon (MPS) and CPU are all detected automatically. Training on an M4 Pro
takes about 12 hours for 3 epochs over 263k reviews; a T4 is comparable. `fp16` is gated
on CUDA — it is applied but measurably no faster on MPS, so it buys only risk there.

## Usage

Run from the repository root — the scripts resolve config and data paths relative to it.

```bash
# 1. Balance and clean the raw reviews (strips the star-rating prefix)
python src/data/create_balanced_dataset.py \
    --input data/raw/reviews.parquet \
    --output_dir data/processed/train_balanced \
    --drop_duplicates

# 2. Tokenize into Hugging Face datasets
python src/data/prepare_bert_data.py

# 3. Fine-tune. Rerun the same command to resume after an interruption —
#    it picks up the newest checkpoint automatically.
python src/models/bert/bert_train.py

# 4. Evaluate
python src/models/bert/bert_evaluate.py

# 5. Serve
python app.py                        # http://localhost:7860

# Optional: baseline reference point
python src/models/baseline/train_baseline.py

# Optional: interpretability
python src/analysis/topics_bertopics.py
python src/analysis/shap_error.py
```

If you only have the tokenized datasets and not the raw parquet, skip steps 1–2 and run
`python src/data/rebuild_without_star_prefix.py --apply` instead.

## Notes

- **Datasets are not committed.** `.gitignore` excludes parquet and CSV data, so `data/raw/` and the intermediate `data/processed/*.parquet` files must be supplied to run the pipeline end to end. The tokenized Hugging Face datasets under `data/processed/bert_train` and `bert_val` are checked in.
- **Checkpoints are ~1.3 GB each** including optimizer state. `save_total_limit: 2` keeps disk use bounded; without it a 3-epoch run writes 32 of them.
- **`bert_evaluate.py` has a fallback chain** that ends in an untrained base BERT and still prints a success line. If you see `Base BERT model loaded (untrained)` in its output, the metrics it produces are meaningless — fix the checkpoint path instead of reading them.
- **`encoder.py` and `shap_error.py` still load the checkpoint with `from_pretrained`**, which does not work on this architecture. `encoder.py` silently falls back to the untrained base model; `shap_error.py` raises. Both need porting to the loading pattern in `predictor.py` before the interpretability layer can be used.
