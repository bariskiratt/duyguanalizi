import pandas as pd
from pathlib import Path


# If True, also scan all Parquet files under data/processed/
SCAN_ALL_PARQUETS = True

INPUT_CANDIDATES = [
    "data/processed/train_data.parquet",
    "data/processed/test_data.parquet",
    # User mentioned this name; try it first
    "data/processed/hepsiburada_bert_format_dataset.parquet",
    # Fallback to our converter's default output name if present
    "data/processed/hepsi_bert_format.parquet",
]

ALLOWED_LABELS = {"negatif", "pozitif", "notr"}

# Optional deduplication (set to True to drop exact duplicate texts)
DROP_DUPLICATES = False

# Hepsiburada score→label thresholds
NEG_MAX = 40
POS_MIN = 80


def read_and_normalize(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}

    # Case A: Already in BERT format (review_text + label or similar)
    if any(c in cols for c in ("review_text", "text", "yorum")) and any(
        c in cols for c in ("label", "labels", "sentiment")
    ):
        # Resolve text column
        text_col = None
        for candidate in ("review_text", "text", "yorum"):
            if candidate in cols:
                text_col = cols[candidate]
                break
        # Resolve label column
        label_col = None
        for candidate in ("label", "labels", "sentiment"):
            if candidate in cols:
                label_col = cols[candidate]
                break
        out = df[[text_col, label_col]].rename(columns={text_col: "review_text", label_col: "label"}).copy()

    # Case B: Raw Hepsiburada schema (Baslik, Yorum, Puan)
    elif all(c in cols for c in ("baslik", "yorum", "puan")):
        baslik = cols["baslik"]
        yorum = cols["yorum"]
        puan = cols["puan"]
        tmp = df[[baslik, yorum, puan]].copy()
        tmp["review_text"] = tmp[baslik].fillna("") + " " + tmp[yorum].fillna("")
        tmp["review_text"] = tmp["review_text"].astype(str).str.strip()
        def map_score_to_label(s):
            try:
                if pd.isna(s):
                    return "notr"
                s = float(s)
            except Exception:
                return "notr"
            if s <= NEG_MAX:
                return "negatif"
            if s >= POS_MIN:
                return "pozitif"
            return "notr"
        tmp["label"] = tmp[puan].apply(map_score_to_label)
        out = tmp[["review_text", "label"]]

    else:
        raise ValueError(
            f"{path}: could not map columns to (review_text, label). Columns: {list(df.columns)}"
        )

    # Normalize text
    out["review_text"] = out["review_text"].astype(str).str.strip()

    # Normalize label: handle string or numeric
    if pd.api.types.is_numeric_dtype(out["label"]):
        # Best-effort mapping; adjust if your numeric mapping differs
        num_to_str = {0: "negatif", 1: "pozitif", 2: "notr"}
        out["label"] = out["label"].map(num_to_str)
    else:
        out["label"] = out["label"].astype(str).str.strip().str.lower()

    # Keep only allowed labels
    out = out[out["label"].isin(ALLOWED_LABELS)]

    # Drop empties
    out = out[(out["review_text"].str.len() > 0)]

    return out


def main() -> None:
    print("🔄 Searching inputs...")
    existing = [Path(p) for p in INPUT_CANDIDATES if Path(p).exists()]
    if SCAN_ALL_PARQUETS:
        more = sorted(Path("data/processed").glob("*.parquet"))
        # Add new ones not already in existing
        for p in more:
            if p not in existing:
                existing.append(p)
    if not existing:
        raise SystemExit("No input files found. Expected at least one of: " + ", ".join(INPUT_CANDIDATES))

    parts = []
    for p in existing:
        print(f"📥 Reading {p}...")
        df = read_and_normalize(p)
        print(f"   → {len(df):,} rows after normalization")
        parts.append(df)

    print("🔗 Concatenating parts...")
    merged = pd.concat(parts, ignore_index=True)
    if DROP_DUPLICATES:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["review_text"]).reset_index(drop=True)
        dropped = before - len(merged)
        print(f"   Dropped {dropped:,} duplicates; final {len(merged):,} rows")
    else:
        print(f"   Final {len(merged):,} rows (duplicates kept)")

    print("📊 Label distribution:")
    print(merged["label"].value_counts())

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save to path used by prepare_bert_data.py
    out_path = out_dir / "merged.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"💾 Saved merged dataset to: {out_path}")


if __name__ == "__main__":
    main()


