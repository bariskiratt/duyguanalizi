import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import os

def run_eda(input_parquet="data/processed/clean.parquet", output_dir="artifacts/eda"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_parquet(input_parquet)
    print("Toplam yorum:", len(df))
    print("Label dağılımı:\n", df["label"].value_counts())

    # Label dağılım grafiği
    df["label"].value_counts().plot(kind="bar", title="Label Dağılımı")
    plt.savefig(f"{output_dir}/label_distribution.png")
    plt.close()

    # Yorum uzunluğu histogramı
    df["length"] = df["review_text"].str.split().apply(len)
    df["length"].hist(bins=10)
    plt.title("Yorum Uzunluğu Dağılımı")
    plt.xlabel("Kelime sayısı")
    plt.ylabel("Frekans")
    plt.savefig(f"{output_dir}/review_length.png")
    plt.close()

    # Kelime bulutu (pozitif örnek)
    text = " ".join(df[df["label"]=="pozitif"]["review_text"].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    wordcloud.to_file(f"{output_dir}/wordcloud_pozitif.png")

    # Negatif kelime bulutu
    text = " ".join(df[df["label"]=="negatif"]["review_text"].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    wordcloud.to_file(f"{output_dir}/wordcloud_negatif.png")

    print(f"EDA raporu görselleri {output_dir}/ klasörüne kaydedildi.")

if __name__ == "__main__":
    run_eda()
