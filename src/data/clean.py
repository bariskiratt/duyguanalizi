import pandas as pd
import re
from langdetect import detect
import os

def clean_text(text: str) -> str:
    # URL kaldır
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # E-posta kaldır
    text = re.sub(r"\S+@\S+", "", text)
    # Telefon numarası kaldır
    text = re.sub(r"\+?\d[\d -]{8,}\d", "", text)
    # Emoji & non-ascii karakterleri kaldır
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.strip()

def process_data(
    input_csv: str = "data/raw/sample_data.csv",
    output_parquet: str = "data/processed/clean.parquet",
    sep: str = ";",
    encoding: str = "utf-8-sig"
) -> dict:
    # 1) CSV'yi oku
    df = pd.read_csv(
        input_csv,
        sep=sep,
        encoding=encoding,
        names=["review_text", "label_num"],
        header=0
    )

    # 2) Durum kodlarını string etiketlere çevir
    label_map = {0: "negatif", 1: "pozitif", 2: "notr"}
    df["label"] = df["label_num"].map(label_map)

    # 3) Temizleme & filtreleme
    cleaned_rows = []
    for _, row in df.iterrows():
        text_raw = row["review_text"]

        # Dil tespiti (opsiyonel)
        try:
            if detect(text_raw) != "tr":
                continue
        except:
            continue

        # Metni temizle
        text = clean_text(text_raw)
        if len(text.split()) < 3:
            continue

        cleaned_rows.append({
            "review_text": text,
            "label": row["label"]
        })

    # 4) DataFrame & Parquet kaydet
    clean_df = pd.DataFrame(cleaned_rows)
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    clean_df.to_parquet(output_parquet, index=False)

    # 5) İstatistikleri dön
    stats = {
        "n_reviews": len(clean_df),
        "label_distribution": clean_df["label"].value_counts(normalize=True).to_dict(),
        "avg_length": clean_df["review_text"].str.split().apply(len).mean()
    }
    return stats

if __name__ == "__main__":
    print("Script başladı...")
    stats = process_data(input_csv="data/raw/sample_data.csv")
    print("Temizlik tamamlandı. İstatistikler:", stats)
