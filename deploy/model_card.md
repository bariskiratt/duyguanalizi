---
language: tr
license: mit
library_name: transformers
tags:
  - sentiment-analysis
  - turkish
  - bert
  - e-commerce
base_model: dbmdz/bert-base-turkish-cased
pipeline_tag: text-classification
---

# Turkish E-Commerce Review Sentiment (BERTurk + MLP)

Three-class sentiment classification for Turkish product reviews:
`negatif`, `pozitif`, `notr`. BERTurk backbone with a mean-pooled MLP head,
fine-tuned on 263,292 reviews.

## Results

65,000 held-out reviews:

| Metric | Value |
| --- | --- |
| Accuracy | 0.8224 |
| Macro F1 | 0.8263 |
| Weighted F1 | 0.8214 |

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| `negatif` | 0.851 | 0.880 | 0.865 | 20,000 |
| `pozitif` | 0.800 | 0.866 | 0.831 | 18,000 |
| `notr` | 0.816 | 0.751 | 0.782 | 27,000 |

Polarity errors are near zero: negative-as-positive and positive-as-negative
together account for 250 of 65,000 predictions. Essentially all error runs along
the neutral axis.

## Loading

**This is not a `PreTrainedModel`.** The checkpoint is a plain `nn.Module` state
dict, so there is no `config.json` and
`AutoModelForSequenceClassification.from_pretrained` fails with an unrecognized
`model_type`. Rebuild the architecture and load the weights:

```python
import torch, yaml
from safetensors.torch import load_file
from transformers import AutoTokenizer
from bert_mlp_classifier import BertMLPConfig, BertMLPWithCustomPooling

cfg = yaml.safe_load(open("bert_hparams.yaml"))
mlp = BertMLPConfig(
    hidden_sizes=cfg["mlp"]["hidden_sizes"],
    dropout_rate=cfg["mlp"]["dropout_rate"],
    activation=cfg["mlp"]["activation"],
    use_batch_norm=cfg["mlp"]["use_batch_norm"],
)
model = BertMLPWithCustomPooling(
    model_name=cfg["model"]["name"], num_classes=3, mlp_config=mlp, pooling="mean"
)
model.load_state_dict(load_file("model.safetensors"))
model.eval()
```

`pooling_strategy` must be `mean`. Pooling carries no parameters, so a mismatched
choice loads cleanly and then runs the head on features it was never trained on.

The reference implementation is `src/inference/predictor.py` in the
[project repository](https://github.com/bariskiratt/duyguanalizi).

## Input normalization is required

The training corpus has almost no punctuation — a comma appears 0.7 times per
10,000 characters, against roughly 150 in ordinary Turkish. The model never
learned to ignore it, and punctuation flips predictions outright:

> `"Berbat, param çöpe gitti."` → **pozitif, 0.99**
> `"Berbat param çöpe gitti"` → **negatif, 1.00**

Strip punctuation and collapse whitespace before inference:

```python
import re
def normalize(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()
```

Measured on 100 reviews rewritten the way a person actually types: 68/100 without
normalization, 81/100 with it — the same score the corpus-form text gets.

## Training

| | |
| --- | --- |
| Base | `dbmdz/bert-base-turkish-cased` |
| Head | 1 hidden layer, 384 units, GELU, dropout 0.25 |
| Pooling | masked mean over token embeddings |
| Max length | 256 (p99 of the corpus is 148 tokens) |
| Batch | 16 × 2 gradient accumulation |
| LR | 2e-5 backbone, 5e-5 head |
| Epochs | 3 |

## Limitations

- **`notr` is the weak class** (0.782 F1). Neutral reviews sit linguistically
  between the other two and usually pair a complaint with a concession.
- **Long reviews degrade.** In a 100-review audit, accuracy fell to 56% past 50
  words, where a single review covers several topics with different sentiment.
- **Labels come from star ratings, not text.** A reviewer leaving five stars and
  then complaining decouples the label from the words. In the same audit, 9 of 19
  errors were cases where the label contradicted the review text.
- **No untouched test split.** Checkpoint selection used the same validation set
  these metrics come from, so they are mildly optimistic.
- **Domain is e-commerce product reviews.** Behaviour on other Turkish text
  (news, social media, support tickets) is untested.

## Label order

`["negatif", "pozitif", "notr"]` — list position is the class id. Defined in
`bert_hparams.yaml`, which must ship alongside the weights.
