import os, json, yaml, time, pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

# 0) Yollar
ART_DIR = "artifacts/topics"
REP_DIR = "reports"
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)

# 1) Config ve veri oku
with open("src/configs/bert_hparams.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DATA_PATH = "data/processed/balanced_clean.parquet"
assert os.path.exists(DATA_PATH), f"{DATA_PATH} bulunamadı"
df = pd.read_parquet(DATA_PATH)
assert "review_text" in df.columns, "Veride 'review_text' kolonu yok"
texts = df["review_text"].astype(str).tolist()

# 2) Proje içi encoder (fine-tuned varsa otomatik kullanır)
from encoder import encode
print("[Progress] 0% → Başladı: metinler yükleniyor…")
emb = encode(
    texts,
    max_length=CFG["model"].get("max_length", 256),
    batch_size=64,
    l2_normalize=True,
    show_progress=True,
    desc="Encoding (BERT)"
)
print("[Progress] 70% → Encoding tamamlandı.")

# 3) UMAP (cosine) + HDBSCAN (euclidean)
umap_model = UMAP(
    n_neighbors=25,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)
hdbscan_model = HDBSCAN(
    min_cluster_size=max(30, len(texts)//100),
    min_samples=5,
    metric="euclidean",
    prediction_data=True
)

# 4) c-TF-IDF için sayma vektörleyici
vectorizer = CountVectorizer(
    ngram_range=(1,2),
    stop_words=None,
    min_df=5
)

# 5) BERTopic modelini kur ve eğit
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    language="turkish",
    top_n_words=10,
    min_topic_size=hdbscan_model.min_cluster_size,
    calculate_probabilities=True,
    verbose=True
)

topics = []
probs = None
# fit_transform tek seferde yaptığı için burada BERT encoding ilerleme çubuğu var;
# BERTopic tarafında yerleşik tqdm yok. Yine de adım adım görünürlük için kendi basit
# progress çıktılarımızı ekleyelim.
print(f"[Progress] 70% → BERTopic fit başlıyor ({len(texts):,} doküman)…")
start = time.time()
topics, probs = topic_model.fit_transform(texts, embeddings=emb)
elapsed = time.time() - start
print(f"[Progress] 95% → Clustering & c-TF-IDF tamamlandı. Süre: {elapsed:.1f}s")

# 6) Raporları üret
info = topic_model.get_topic_info()
fig = topic_model.visualize_barchart(top_n_topics=20)
fig.write_html(os.path.join(REP_DIR, "topic_report.html"))
print("[Progress] 98% → Rapor yazıldı: topic_report.html")

# Modeli kaydet (yeniden yükleme için)
topic_model.save(os.path.join(ART_DIR, "bertopic_model"))
print("[Progress] 99% → Model kaydedildi: artifacts/topics/bertopic_model/")

# 7) topics.json (topic_id, label, KW listesi)
rows = []
for tid in info["Topic"].tolist():
    if tid == -1:
        continue
    words = [w for w, _ in topic_model.get_topic(tid)]
    label = " / ".join(words[:3]) if words else f"Topic {tid}"
    rows.append({"topic_id": int(tid), "label": label, "KW": words})
with open(os.path.join(ART_DIR, "topics.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("[Progress] 100% → topics.json yazıldı")

# 8) Kabul kontrolü
ok = all(len(r["KW"]) >= 3 for r in rows)
print("KABUL (Her topic ≥ 3 KW):", "OK" if ok else "REVİZE")
print("Yazıldı →", os.path.join(REP_DIR, "topic_report.html"), ",", os.path.join(ART_DIR, "topics.json"), ",", os.path.join(ART_DIR, "bertopic_model"))
