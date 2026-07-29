# Manual audit — 100 reviews

100 reviews sampled from the validation set, stratified by class (34 negative /
33 positive / 33 neutral), seed 1923. Labels come from the data, not from
judgement. Predictions run through `src/inference/predictor.py`, the same path
the Gradio app uses.

## Scenarios

The same 100 reviews, presented three ways:

| Scenario | Correct | Macro F1 |
| --- | --- | --- |
| A — corpus form (no punctuation) | 81/100 | 0.807 |
| B — user punctuation, normalized | **81/100** | **0.807** |
| C — user punctuation, raw | 68/100 | 0.677 |

B matching A exactly is the point: the normalization in `predictor.py` fully
closes the gap between how the corpus is written and how a person types. Without
it the model loses 13 points on identical content.

## Per class (scenario A)

| Class | Precision | Recall | F1 | n |
| --- | --- | --- | --- | --- |
| `negatif` | 0.848 | 0.824 | 0.836 | 34 |
| `pozitif` | 0.838 | 0.939 | 0.886 | 33 |
| `notr` | 0.733 | 0.667 | 0.698 | 33 |

Confusion matrix (rows are truth):

| | negatif | pozitif | notr |
| --- | --- | --- | --- |
| **negatif** | 28 | 0 | 6 |
| **pozitif** | 0 | 31 | 2 |
| **notr** | 5 | 6 | 22 |

Zero polarity confusion in either direction.

## Where it struggles

By review length:

| Length | Accuracy |
| --- | --- |
| < 10 words | 81% (22/27) |
| 10–25 words | 89% (34/38) |
| 25–50 words | 77% (20/26) |
| 50+ words | 56% (5/9) |

Long reviews are worst — they usually cover several topics at once (shipping
good, product bad, price fair) and collapse badly into one label. Very short
reviews suffer from the opposite problem: too little to go on.

By confidence:

| Confidence | Accuracy |
| --- | --- |
| 0.90–1.00 | 95% (42/44) |
| 0.80–0.90 | 70% (14/20) |
| 0.60–0.80 | 79% (22/28) |
| < 0.60 | 38% (3/8) |

Confidence is usable as a signal: above 0.90 the model is right 19 times in 20,
below 0.60 it is near a coin flip. Surfacing low-confidence predictions as
"undecided" in the UI is justified. The middle bands crossing over is small-sample
noise — 20 and 28 examples respectively.

## The 19 errors

Read individually and categorised. This categorisation is a judgement call, not a
measurement:

| Category | Count |
| --- | --- |
| Label contradicts the text (model arguably right) | 9 |
| Genuinely ambiguous, either label defensible | 8 |
| Model clearly wrong | 2 |

Examples:

- **Label suspect.** `"süper kargo süper ses kalitesi süper 7 1 süper rahatlık
  süper mikrofon süper daha ne diueyim süpersüpersüper"` — labelled `notr`,
  predicted `pozitif` at 0.93.
- **Label suspect.** `"ürünü hiç beğenmedim yorumlarda görüşme esnasında karşı
  tarafa sesin kötü gittiğinden bahsediliyordu..."` — labelled `notr`, predicted
  `negatif` at 0.97.
- **Ambiguous.** `"Hasarlı vida Paketleme oldukça özenli yapılmış malzemede
  herhangi bir eksiklik yoktu Görünüm gayet başarılı..."` — labelled `negatif`,
  predicted `notr` at 0.86. A damaged screw inside an otherwise positive review.
- **Model wrong.** `"ideal beğenerek almıştım beklentilerin üstünde çıktı"` —
  labelled `pozitif`, predicted `notr` at 0.69.

Labels derive from star ratings rather than review text, so a user who leaves
five stars and then complains — or three stars and writes "süper" — decouples the
label from the words. Measured accuracy therefore has a ceiling set by label
quality. Only 2 of 19 errors here are unambiguous model failures.

## Caveats

- **No untouched test split.** The pipeline produces only `bert_train` and
  `bert_val`, and `load_best_model_at_end` selected the checkpoint against the
  same validation set these 100 came from. Treat the numbers as mildly optimistic.
- **100 is a narrow base.** The subgroup breakdowns (9 reviews at 50+ words, 8
  below 0.60 confidence) show directions, not reliable rates.
