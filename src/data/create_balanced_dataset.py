import argparse
from pathlib import Path
import pandas as pd


ALLOWED_LABELS = {"negatif", "pozitif", "notr"}


def load_dataframe(input_path: Path, text_col: str = "review_text", label_col: str = "label") -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Expected columns '{text_col}' and '{label_col}' in {input_path}. Got: {list(df.columns)}"
        )

    # Normalize
    df = df[[text_col, label_col]].rename(columns={text_col: "review_text", label_col: "label"}).copy()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    if pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].map({0: "negatif", 1: "pozitif", 2: "notr"})
    else:
        df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Keep only allowed labels and non-empty texts
    df = df[df["label"].isin(ALLOWED_LABELS)]
    df = df[df["review_text"].str.len() > 0]
    return df


def balance_after_dedup(
    df: pd.DataFrame,
    per_class: int | None,
    random_state: int,
) -> pd.DataFrame:
    # Deduplicate by text first
    print("🧹 Dropping duplicate review_text before balancing...")
    before = len(df)
    df = df.drop_duplicates(subset=["review_text"]).reset_index(drop=True)
    print(f"   Removed {before - len(df):,} duplicates; remaining {len(df):,}")

    # Class counts
    counts = df["label"].value_counts().to_dict()
    print("📊 Counts after dedup:", counts)
    if not counts:
        raise ValueError("No rows found after deduplication.")

    min_count = min(counts.values())
    if min_count == 0:
        missing = [c for c, n in counts.items() if n == 0]
        raise ValueError(f"Some classes are empty after dedup: {missing}")

    # Determine per-class target
    target = min_count if per_class is None else min(per_class, min_count)
    print(f"🎯 Target per-class samples: {target}")

    # Sample equal counts per class
    parts: list[pd.DataFrame] = []
    for cls in sorted(counts.keys()):
        cls_df = df[df["label"] == cls]
        if len(cls_df) < target:
            raise ValueError(f"Class '{cls}' has only {len(cls_df)} rows < target {target}")
        parts.append(cls_df.sample(n=target, random_state=random_state))

    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print("✅ Final balanced counts:")
    print(balanced["label"].value_counts())
    print(f"Total: {len(balanced):,}")
    return balanced


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a class-balanced dataset after deduplication.")
    p.add_argument("--input", default="data/processed/merged.parquet", help="Input Parquet path")
    p.add_argument("--output_dir", default="data/processed", help="Output directory")
    p.add_argument("--output_name", default="balanced.parquet", help="Output filename")
    p.add_argument("--per_class", type=int, default=50000, help="Optional per-class cap; defaults to min class size")
    p.add_argument("--random_state", type=int, default=2025, help="Random seed for reproducibility")
    p.add_argument("--text_col", default="review_text", help="Name of text column in input (default: review_text)")
    p.add_argument("--label_col", default="label", help="Name of label column in input (default: label)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Reading: {input_path}")
    df = load_dataframe(input_path, text_col=args.text_col, label_col=args.label_col)
    print(f"   Loaded {len(df):,} rows")

    balanced = balance_after_dedup(df, per_class=args.per_class, random_state=args.random_state)

    out_path = out_dir / args.output_name
    balanced.to_parquet(out_path, index=False)
    print(f"💾 Saved balanced dataset to: {out_path}")


if __name__ == "__main__":
    main()
