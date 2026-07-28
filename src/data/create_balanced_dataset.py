import argparse, json, re
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path


# Kaynak veride yildiz puani yorum metninin basina metin olarak eklenmis:
# "Beş Yıldız Kokusunu ve yoğunluğunu beğendim...", "Üç Yıldız Güzeldi".
# Bu dogrudan etiket sizintisidir - orneklerin ~%25'inde cevap girdinin icinde
# yazili ve olculdugunde onek tasiyan orneklerde dogruluk %100, tasimayanlarda
# %82 cikiyor. Model duyguyu degil yildizi okumayi ogreniyor.
#
# Yalnizca metnin EN BASINDAKI kalibi siliyoruz. "10 numara 5 yıldız" gibi cumle
# icindeki kullanimlar kullanicinin kendi ifadesidir, onlar korunur.
#
# IGNORECASE kullanmiyoruz: Python'da 'İ'.lower() birlesik iki karaktere
# donustugu icin Turkce buyuk I ile eslesme kacabiliyor. Varyantlari acikca
# yaziyoruz.
_SAYI = r"(?:[Bb]ir|[Tt]ek|[İIiı]ki|[ÜUüu]ç|[Dd][öo]rt|[Bb]e[şs]|\d+)"
_YILDIZ_ONEKI = re.compile(rf"^\s*{_SAYI}\s+[Yy][ıi]ld[ıi]z\b[\s.,:;!\-]*")


def strip_star_prefix(text) -> str:
    """Yorum metninin basina eklenmis yildiz puani etiketini siler."""
    return _YILDIZ_ONEKI.sub("", str(text), count=1).lstrip()


def parse_args():
    p = argparse.ArgumentParser(description="Create a class-balanced parquet with NaNs dropped first")
    p.add_argument("--input", type=str, default="data/raw/train_filtered.csv", help="Input parquet path")
    p.add_argument("--output_dir", type=str, default="data/processed/train_balanced", help="Output directory")
    p.add_argument("--text_col", type=str, default="review_text", help="Text column name")
    p.add_argument("--label_col", type=str, default="label", help="Label column name")
    p.add_argument("--per_class", type=int, default=None, help="Target samples per class (uniform)")
    p.add_argument("--target_total", type=int, default=None, help="Target total samples (split uniformly across classes)")
    p.add_argument("--class_counts_json", type=str, default=None, help="JSON file mapping label->desired_count (overrides others)")
    p.add_argument("--drop_duplicates", action="store_true", help="Drop duplicates by text column before sampling")
    p.add_argument("--keep_star_prefix", action="store_true",
                   help="Metin basindaki yildiz onekini SILME (etiket sizintisini kasten birakmak icin)")
    p.add_argument("--seed", type=int, default=1923, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.input)
    # 1) Drop NaNs/empties first
    df = df.dropna(subset=[args.text_col, args.label_col])
    df[args.text_col] = df[args.text_col].astype(str).str.strip()

    # 1b) Metnin basindaki yildiz puani onekini sil. Bosaltma isleminden ONCE
    # yapiliyor ki sadece onekten ibaret olan yorumlar bir alttaki uzunluk
    # filtresine takilip dussun.
    if not args.keep_star_prefix:
        oncesi = df[args.text_col].copy()
        df[args.text_col] = df[args.text_col].map(strip_star_prefix)
        degisen = int((oncesi != df[args.text_col]).sum())
        print(f"Yildiz oneki temizlenen satir: {degisen} / {len(df)}")

    df = df[df[args.text_col].str.len() > 0]

    # Normalize label values: strip and lowercase so labels are consistent
    # Convert to string first to avoid issues with numeric labels
    if args.label_col in df.columns:
        df[args.label_col] = df[args.label_col].astype(str).str.strip().str.lower()

    # Optional: drop duplicates by text to reduce near-duplicates before balancing
    if args.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=[args.text_col])
        print(f"Dropped duplicates: {before - len(df)}")

    # Ensure categorical label for consistent ordering
    df[args.label_col] = df[args.label_col].astype("category")
    classes = sorted(df[args.label_col].unique().tolist())
    counts = df[args.label_col].value_counts().to_dict()
    print("Class counts:", counts)

    # 2) Decide desired counts per class
    per_class_targets = {}
    if args.class_counts_json:
        with open(args.class_counts_json, "r", encoding="utf-8") as f:
            desired = json.load(f)
        # Map possibly string labels to category values
        for c in classes:
            key = str(c)
            per_class_targets[c] = int(desired.get(key, 0))
    elif args.per_class is not None:
        for c in classes:
            per_class_targets[c] = int(args.per_class)
    elif args.target_total is not None:
        uniform = max(1, args.target_total // len(classes))
        for c in classes:
            per_class_targets[c] = uniform
    else:
        # Default: uniform max possible given the minority class size
        min_available = min(counts.values())
        for c in classes:
            per_class_targets[c] = min_available

    print("Desired per-class targets:", per_class_targets)

    # 3) Sample per class with caps by availability
    parts = []
    for c in classes:
        cls_df = df[df[args.label_col] == c]
        n_desired = per_class_targets[c]
        n = min(len(cls_df), n_desired)
        if n <= 0:
            continue
        parts.append(cls_df.sample(n, random_state=args.seed))

    if not parts:
        raise SystemExit("No samples selected. Check your targets and input data.")

    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print("Balanced counts:", balanced[args.label_col].value_counts().to_dict())
    print("Total:", len(balanced))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "balanced.parquet"
    balanced.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}, rows: {len(balanced)}")


if __name__ == "__main__":
    main()
