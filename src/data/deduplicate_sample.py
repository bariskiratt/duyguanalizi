import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib


def parse_args():
    p = argparse.ArgumentParser(description="Sample-based near-duplicate removal for text data")
    p.add_argument("--input", type=str, default="data/processed/balanced_clean.parquet", help="Input parquet path")
    p.add_argument("--text_col", type=str, default="review_text", help="Text column to deduplicate on")
    p.add_argument("--sample_size", type=int, default=169001, help="Max rows to sample for near-duplicate detection (ignored if --full)")
    p.add_argument("--threshold", type=float, default=0.90, help="Cosine similarity threshold to treat as near-duplicate")
    p.add_argument("--min_df", type=int, default=3, help="TF-IDF min_df")
    p.add_argument("--block_prefix_len", type=int, default=10, help="Bucket by first N chars of norm_for_dupe to limit pairwise cost")
    p.add_argument("--output", type=str, default="data/processed/balanced_clean_dedup_sample.parquet", help="Output parquet path for cleaned sample")
    p.add_argument("--full", action="store_true", help="Process the entire dataset with blocking (no sampling)")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.input)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column not found: {args.text_col}")

    # Normalize for case-insensitive duplicate detection
    df["norm_for_dupe"] = df[args.text_col].astype(str).str.lower().str.strip()

    # Exact duplicates first
    before = len(df)
    df = df.drop_duplicates(subset=["norm_for_dupe"]).reset_index(drop=True)
    print(f"Dropped exact duplicates: {before - len(df)}")

    # Choose working set: full with blocking or sampled subset
    if args.full:
        work = df.reset_index(drop=True)
        print(f"Working on FULL dataset with blocking: {len(work)} rows")
    else:
        n = min(args.sample_size, len(df))
        work = df.sample(n, random_state=42).reset_index(drop=True)
        print(f"Sample size for near-dup detection: {len(work)}")

    # Create blocks by prefix (or hash of prefix) to avoid all-pairs
    prefix_len = max(3, args.block_prefix_len)
    work["_block"] = work["norm_for_dupe"].str[:prefix_len]

    keep_indices = []
    drop_indices = set()
    total_flagged = 0

    for block, grp in work.groupby("_block"):
        doc_n = len(grp)
        if doc_n <= 2:
            keep_indices.extend(grp.index.tolist())
            continue
        # Ensure min_df is valid relative to small blocks
        local_min_df = max(1, min(args.min_df, doc_n))
        try:
            vec = TfidfVectorizer(min_df=local_min_df, ngram_range=(1, 2))
            X = vec.fit_transform(grp["norm_for_dupe"]) 
            S = cosine_similarity(X, dense_output=False)
        except ValueError:
            # Fallback for rare edge-cases
            vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
            X = vec.fit_transform(grp["norm_for_dupe"]) 
            S = cosine_similarity(X, dense_output=False)
        grp_idx = grp.index.to_list()
        for i in range(S.shape[0]):
            global_i = grp_idx[i]
            if global_i in drop_indices:
                continue
            sims = S[i].toarray().ravel()
            dup_idx = np.where(sims > args.threshold)[0]
            for j in dup_idx:
                if j == i:
                    continue
                global_j = grp_idx[j]
                if global_j not in drop_indices:
                    drop_indices.add(global_j)
                    total_flagged += 1
        # Add those not dropped in this block
        keep_indices.extend([idx for idx in grp_idx if idx not in drop_indices])

    cleaned = work.loc[sorted(set(keep_indices))].drop(columns=["_block"]).reset_index(drop=True)
    scope = "FULL" if args.full else "SAMPLE"
    print(f"Near-duplicates flagged in {scope} (blocked): {total_flagged}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out_path, index=False)
    print(f"Saved cleaned {scope.lower()} → {out_path} (rows: {len(cleaned)})")


if __name__ == "__main__":
    main()


