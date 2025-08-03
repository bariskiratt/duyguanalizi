import pandas as pd
import re
from langdetect import detect
from datetime import datetime
import json
import os

def load_schema(schema_path="configs/schema.json"):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return schema

def clean_text(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)  # URL kaldır
    text = re.sub(r"\S+@\S+", "", text)           # E-posta kaldır
    text = re.sub(r"\+?\d[\d -]{8,}\d", "", text) # Telefon kaldır
    text = text.encode("ascii", "ignore").decode("ascii") # Emoji kaldır
    return text.strip()

def star_to_label(star):
    if star in [1, 2]:
        return "negatif"
    elif star == 3:
        return "notr"
    elif star in [4, 5]:
        return "pozitif"
    return "bilinmiyor"

def process_data(input_csv="data/raw/reviews_sample.csv", 
                 output_parquet="data/processed/clean.parquet"):
    
    df = pd.read_csv(input_csv)
    
    cleaned_rows = []
    for _, row in df.iterrows():
        try:
            lang = detect(row["review_text"])
            if lang != "tr":  
                continue
        except:
            continue
        
        text = clean_text(row["review_text"])
        if len(text.split()) < 3:
            continue

        cleaned_rows.append({
            "review_id": row["review_id"],
            "product_id": row["product_id"],
            "review_date": pd.to_datetime(row["review_date"]),
            "review_text": text,
            "star_rating": row["star_rating"],
            "label": star_to_label(row["star_rating"])
        })
    
    clean_df = pd.DataFrame(cleaned_rows)
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    clean_df.to_parquet(output_parquet, index=False)
    
    stats = {
        "n_reviews": len(clean_df),
        "label_distribution": clean_df["label"].value_counts(normalize=True).to_dict(),
        "avg_length": clean_df["review_text"].str.split().apply(len).mean()
    }
    return stats

if __name__ == "__main__":
    print("Script başladı...")
    stats = process_data()
    print("Temizlik tamamlandı. İstatistikler:", stats)

