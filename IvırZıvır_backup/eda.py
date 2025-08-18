import os, pandas as pd, matplotlib.pyplot as plt
from wordcloud import WordCloud

def run_eda(
    input_parquet: str = "data/processed/train_data.parquet",
    output_dir:    str = "artifacts/eda"
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(input_parquet)

    print("Toplam yorum :", len(df))
    print(df["label"].value_counts(normalize=True).round(3))

    # -------- label dağılımı -------------------------------------------
    ax = df["label"].value_counts().plot(kind="bar")
    ax.set_xlabel("Etiket"); ax.set_ylabel("Adet")
    ax.set_title("Label Dağılımı – Train")
    plt.tight_layout(); plt.savefig(f"{output_dir}/label_distribution.png")
    plt.close()

    # -------- uzunluk histogramı ---------------------------------------
    (df["review_text"].str.split().apply(len)
        .hist(bins=30, figsize=(6,3)))
    plt.title("Yorum Kelime Uzunluğu")
    plt.xlabel("Kelime"); plt.ylabel("Frekans")
    plt.tight_layout(); plt.savefig(f"{output_dir}/review_length.png")
    plt.close()

    # -------- kelime bulutları -----------------------------------------
    for lbl in ["pozitif", "negatif", "notr" ]:
        corpus = " ".join(df[df["label"] == lbl]["review_text"].tolist())
        if corpus.strip():
            WordCloud(width=900, height=400, background_color="white")\
                .generate(corpus)\
                .to_file(f"{output_dir}/wordcloud_{lbl}.png")

    print(f"✔ EDA görselleri {output_dir}/ altında.")

if __name__ == "__main__":
    run_eda()
