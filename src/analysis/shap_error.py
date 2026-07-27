import os, re, json, yaml, numpy as np, pandas as pd, torch, emoji
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import shap

# 0) Config
with open("src/configs/bert_hparams.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# 1) Modeli ve tokenizer'ı yükle (eğitilmiş BERT)
CKPT = "./artifacts/bert_mlp_ckpt/best_model"
assert os.path.exists(CKPT), f"Checkpoint bulunamadı: {CKPT}"
tok = AutoTokenizer.from_pretrained(CKPT)
mdl = AutoModelForSequenceClassification.from_pretrained(CKPT).eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mdl.to(DEVICE)

# 2) Veri: text + gold label (balanced_clean.parquet)
DATA_PATH = "data/processed/balanced_clean.parquet"
assert os.path.exists(DATA_PATH), f"{DATA_PATH} yok"
df = pd.read_parquet(DATA_PATH)
assert {"review_text","label"}.issubset(df.columns), "review_text/label kolonları şart"

# 3) Tahmin fonksiyonu (SHAP bunun etrafında çalışacak)
@torch.no_grad()
def predict_proba_text(texts, max_length=CFG["model"].get("max_length", 256), batch_size=64, show_progress=True, desc="Predict"):
    """Metin listesi → (N, num_labels) olasılık matrisi"""
    all_probs = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=(len(texts)+batch_size-1)//batch_size, desc=desc)
    for i in iterator:
        batch = texts[i:i+batch_size]
        # Güvenli: tüm girdileri stringe çevir
        batch = ["" if x is None else str(x) for x in batch]
        enc = tok(
            batch, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        ).to(DEVICE)
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)

# 4) Küçük bir test (rapor) ve hata çerçevesi
texts = df["review_text"].astype(str).tolist()

# Etiketleri sayıya çevir (karışık tipleri normalize et)
def _map_label(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip().lower()
    if s in {"negatif", "negative", "neg"}:
        return 0
    if s in {"pozitif", "positive", "pos"}:
        return 1
    if s in {"notr", "nötr", "neutral", "neu"}:
        return 2
    return 2  # bilinmeyenleri nötr kabul et

y_true = df["label"].apply(_map_label).to_numpy(dtype=int)

probs  = predict_proba_text(texts, batch_size=64, show_progress=True, desc="Predict (all)")
y_pred = probs.argmax(axis=1)
print(classification_report(y_true, y_pred, labels=[0,1,2], target_names=["negatif","pozitif","notr"], zero_division=0))

# 5) Hataları ayıkla ve güvene göre sırala → 50 örnek seç
conf  = probs.max(axis=1)
errors = pd.DataFrame({
    "review_text": df["review_text"].astype(str).fillna(""),
    "true_label": y_true,
    "pred_label": y_pred,
    "prob_pred": conf
})
errors = errors[errors["true_label"] != errors["pred_label"]].sort_values("prob_pred", ascending=False)
errors = errors.head(50).copy()
errors = errors[errors["review_text"].str.len() > 0]

# 6) SHAP: metin maskesi ile açıklama (yaklaşık)
masker = shap.maskers.Text(r"\W+")
explainer = shap.Explainer(predict_proba_text, masker, output_names=[str(i) for i in range(probs.shape[1])])

# SHAP'i progres bar ile parça parça hesapla
sample_texts = errors["review_text"].astype(str).tolist()
sv_parts = []
for t in tqdm(sample_texts, desc="SHAP explaining"):
    text = "" if t is None else str(t)
    s = explainer([text], max_evals=600)
    sv_parts.append(s[0])

# 7) Her örnek için en baskın 5 token'ı topla
def top_tokens(shap_values, top_k=5):
    vals = np.abs(shap_values.values)
    if vals.ndim == 2:  # class x token
        vals = vals.sum(axis=0)
    idx = np.argsort(-vals)[:top_k]
    toks = np.array(shap_values.data)[idx].tolist()
    return toks

errors["shap_top_tokens"] = [top_tokens(s, top_k=5) for s in sv_parts]

# 8) Hata bucket kuralları (ironi/negasyon/emoji vb.)
IRONY_PATTERNS = [
    r"\(!\)", r"(!)\s*$", r"tabii ki", r"“?harika”?", r"“?mükemmel”?",
    r"teşekkürler", r"sağ ol", r"aynen", r"bravo", r"süper(!)", r"şaka gibi"
]
NEGATION = [r"\bdeğil\b", r"\basla\b", r"\bhiç\b", r"\byok\b", r"\bne yazık ki\b"]
EMOJI_NEG = ["😒","🙃","😑","😤","😠","😡","😭"]

def bucketize(text):
    t = text.lower()
    b = []
    if any(re.search(p, t) for p in IRONY_PATTERNS) or any(e in text for e in EMOJI_NEG):
        b.append("ironi")
    if any(re.search(p, t) for p in NEGATION):
        b.append("negasyon")
    if len(t.split()) <= 3:
        b.append("çok_kısa")
    if any(ch in text for ch in ["?", "!"]):
        b.append("punct")
    if sum(c.isupper() for c in text) >= 5:
        b.append("BÜYÜK_HARF")
    if emoji.emoji_count(text) > 0:
        b.append("emoji")
    return b or ["diğer"]

errors["buckets"] = errors["review_text"].apply(bucketize)

# 9) CSV'ye yaz ve kabul kontrolü
out_csv = "error_samples.csv"
errors.to_csv(out_csv, index=False)
ironi_say = errors["buckets"].apply(lambda bs: "ironi" in bs).sum()
print("Yazıldı →", out_csv, "; İroni örneği:", ironi_say, ("OK" if ironi_say>=5 else "REVİZE"))

# 10) SHAP özet raporu (HTML + JSON)
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

# Top token frekansları
token_counter = Counter()
for toks in errors["shap_top_tokens"]:
    if isinstance(toks, (list, tuple)):
        token_counter.update([str(t) for t in toks if t is not None and str(t).strip() != ""])

top_tokens = token_counter.most_common(50)

# Bucket dağılımı
all_buckets = []
for bs in errors["buckets"]:
    if isinstance(bs, (list, tuple)):
        all_buckets.extend(bs)
bucket_counts = Counter(all_buckets)

# Sınıflandırma raporunu JSON olarak ekle
from sklearn.metrics import classification_report as _cr
class_report = _cr(y_true, y_pred, labels=[0,1,2], target_names=["negatif","pozitif","notr"], zero_division=0, output_dict=True)

summary_json = {
    "total_errors": int(len(errors)),
    "irony_count": int(ironi_say),
    "bucket_counts": dict(bucket_counts),
    "top_tokens": top_tokens,
    "classification_report": class_report,
}

with open(os.path.join(reports_dir, "shap_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary_json, f, ensure_ascii=False, indent=2)

# HTML rapor
def _truncate(x, n=240):
    s = str(x)
    return s if len(s) <= n else s[:n] + "…"

details_df = errors[["review_text","true_label","pred_label","prob_pred","shap_top_tokens","buckets"]].copy()
details_df["review_text"] = details_df["review_text"].apply(lambda s: _truncate(s, 240))

top_tokens_df = pd.DataFrame(top_tokens, columns=["token","count"])
bucket_df = pd.DataFrame(sorted(bucket_counts.items(), key=lambda x: -x[1]), columns=["bucket","count"])

html = []
html.append("<html><head><meta charset='utf-8'><title>SHAP Error Report</title>"
            "<style>body{font-family:Arial, sans-serif; margin:24px;} table{border-collapse:collapse; width:100%; margin:12px 0;}"
            "th,td{border:1px solid #ddd; padding:8px; font-size:13px;} th{background:#f4f4f4; text-align:left;}"
            "h1,h2{margin:8px 0;}</style></head><body>")
html.append("<h1>SHAP Error Report</h1>")
html.append(f"<p><b>Total errors:</b> {len(errors)} | <b>Irony:</b> {ironi_say}</p>")
html.append("<h2>Bucket Distribution</h2>")
html.append(bucket_df.to_html(index=False, escape=True))
html.append("<h2>Top SHAP Tokens</h2>")
html.append(top_tokens_df.to_html(index=False, escape=True))
html.append("<h2>Misclassified Samples (Top 50)</h2>")
html.append(details_df.to_html(index=False, escape=True))
html.append("</body></html>")

with open(os.path.join(reports_dir, "shap_error_report.html"), "w", encoding="utf-8") as f:
    f.write("\n".join(html))

print("Yazıldı →", os.path.join(reports_dir, "shap_summary.json"))
print("Yazıldı →", os.path.join(reports_dir, "shap_error_report.html"))
