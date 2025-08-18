#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_full.py
-------------
Turkish-friendly, full-corpus text deduplication + near-duplicate mining for review datasets.

Outputs:
- Cleaned parquet (no exact dupes; one exemplar per near-dupe cluster)
- near_dupe_pairs.csv: Auditable list of near-duplicate pairs above threshold
- label_conflicts.csv: Any exact/near-dupe groups with conflicting labels (if --label_col provided)

Run BEFORE train/val/test split to avoid leakage.

Example:
    python clean_full.py \
        --input data/processed/balanced.parquet \
        --text_col review_text \
        --label_col label \
        --output_clean data/processed/clean.parquet \
        --output_pairs data/processed/near_dupe_pairs.csv \
        --output_conflicts data/processed/label_conflicts.csv \
        --threshold 0.92 --block_prefix_len 8

"""
import argparse
from collections import defaultdict, Counter
from dataclasses import dataclass
import sys
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------- Utilities ---------------------------

def norm_text(s: str) -> str:
    """Unicode-aware normalization for Turkish text dedupe.
    - Normalize to NFKC (compatibility forms).
    - Casefold (better than lower() for TR locale).
    - Collapse whitespace.
    """
    s = unicodedata.normalize("NFKC", str(s))
    s = s.casefold()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


class UnionFind:
    """Disjoint-set (Union-Find) for clustering indices across multiple blocks."""
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


@dataclass
class Args:
    input: str
    text_col: str
    label_col: str | None
    threshold: float
    min_df: int
    ngram_min: int
    ngram_max: int
    block_prefix_len: int
    output_clean: str
    output_pairs: str
    output_conflicts: str
    sample_size: int
    keep_strategy: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Full-corpus Turkish-friendly text dedup + near-duplicate cleaning.")
    p.add_argument("--input", type=str, required=True, help="Input parquet path")
    p.add_argument("--text_col", type=str, default="review_text", help="Text column")
    p.add_argument("--label_col", type=str, default=None, help="Optional label column for conflict reporting")
    p.add_argument("--threshold", type=float, default=0.92, help="Cosine similarity threshold (char n-grams)")
    p.add_argument("--min_df", type=int, default=3, help="TF-IDF min_df per block")
    p.add_argument("--ngram_min", type=int, default=3, help="char_wb ngram min")
    p.add_argument("--ngram_max", type=int, default=5, help="char_wb ngram max")
    p.add_argument("--block_prefix_len", type=int, default=8, help="Blocking prefix length (also used for suffix)")
    p.add_argument("--output_clean", type=str, default="data/processed/clean.parquet", help="Output cleaned parquet")
    p.add_argument("--output_pairs", type=str, default="data/processed/near_dupe_pairs.csv", help="Output CSV for flagged near-duplicate pairs")
    p.add_argument("--output_conflicts", type=str, default="data/processed/label_conflicts.csv", help="Output CSV for label conflicts")
    p.add_argument("--sample_size", type=int, default=0, help="Optional sample size for speed (0 = full dataset)")
    p.add_argument("--keep_strategy", type=str, default="first", choices=["first","shortest","longest"],
                   help="Which exemplar to keep per cluster")
    a = p.parse_args()

    return Args(
        input=a.input,
        text_col=a.text_col,
        label_col=a.label_col,
        threshold=a.threshold,
        min_df=a.min_df,
        ngram_min=a.ngram_min,
        ngram_max=a.ngram_max,
        block_prefix_len=max(6, a.block_prefix_len),
        output_clean=a.output_clean,
        output_pairs=a.output_pairs,
        output_conflicts=a.output_conflicts,
        sample_size=max(0, a.sample_size),
        keep_strategy=a.keep_strategy,
    )


def ensure_parent(p: str | Path):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------- Core logic ---------------------------

def drop_exact_duplicates(df: pd.DataFrame, text_col: str, label_col: str | None):
    """Drop exact duplicates on normalized text, report counts and conflicts (if labels available)."""
    before = len(df)
    # Conflicts on exact duplicates
    conflict_records = []
    if label_col and label_col in df.columns:
        grp = df.groupby("norm_for_dupe")
        for key, sub in grp:
            labels = sub[label_col].astype(str).unique().tolist()
            if len(labels) > 1:
                # Multiple labels share exactly the same normalized text
                conflict_records.append({
                    "norm_for_dupe": key,
                    "labels": "|".join(sorted(labels)),
                    "count": len(sub),
                    "example_text": sub[text_col].iloc[0]
                })

    df_dedup = df.drop_duplicates(subset=["norm_for_dupe"]).reset_index(drop=True)
    dropped = before - len(df_dedup)
    return df_dedup, dropped, conflict_records


def build_blocks(df_norm: pd.Series, prefix_len: int):
    """Return two blocking keys: head and tail of whitespace-stripped normalized text."""
    no_space = df_norm.str.replace(r"\s+", "", regex=True)
    block_a = no_space.str[:prefix_len]
    block_b = no_space.str[-prefix_len:]
    return block_a, block_b


def vectorize_and_find_pairs(texts: pd.Series, min_df: int, ngram_min: int, ngram_max: int, threshold: float):
    """Within a block, compute char_wb TF-IDF, sparse cosine, and return list of (local_i, local_j, sim) pairs above threshold."""
    doc_n = len(texts)
    local_min_df = max(1, min(min_df, doc_n))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(ngram_min, ngram_max), min_df=local_min_df)
    X = vec.fit_transform(texts.values.astype("U"))
    # Sparse cosine similarities; beware that this is still O(n^2) within block, but blocks are small.
    S = cosine_similarity(X, dense_output=False)
    pairs = []
    # Iterate only upper triangle
    S = S.tocsr()
    for i in range(S.shape[0]):
        row = S.getrow(i)
        for j, sim in zip(row.indices, row.data):
            if j <= i:
                continue
            if sim >= threshold:
                pairs.append((i, j, float(sim)))
    return pairs


def choose_exemplar(sub_df: pd.DataFrame, keep_strategy: str, text_col: str):
    if keep_strategy == "first":
        return sub_df.index[0]
    if keep_strategy == "shortest":
        lengths = sub_df[text_col].astype(str).map(len)
        return lengths.idxmin()
    if keep_strategy == "longest":
        lengths = sub_df[text_col].astype(str).map(len)
        return lengths.idxmax()
    return sub_df.index[0]


def cluster_and_select(df: pd.DataFrame, pairs_global: list[tuple[int,int,float]], keep_strategy: str, text_col: str):
    """From global index pairs, build clusters with Union-Find, return set of kept indices and per-cluster conflicts if any."""
    uf = UnionFind()
    for i, j, _ in pairs_global:
        uf.union(i, j)

    # Build clusters {root: [indices]}
    clusters = defaultdict(list)
    for idx in set([i for i,_,_ in pairs_global] + [j for _,j,_ in pairs_global]):
        clusters[uf.find(idx)].append(idx)

    keep = set()
    drop = set()
    for root, members in clusters.items():
        sub = df.loc[members]
        keep_idx = choose_exemplar(sub, keep_strategy, text_col)
        keep.add(keep_idx)
        for m in members:
            if m != keep_idx:
                drop.add(m)
    return keep, drop, clusters


def collect_label_conflicts(df: pd.DataFrame, clusters: dict, label_col: str, text_col: str):
    """For each cluster, if multiple labels present, record a conflict row."""
    conflict_rows = []
    for _, members in clusters.items():
        sub = df.loc[members]
        labels = sub[label_col].astype(str).tolist()
        if len(set(labels)) > 1:
            row = {
                "cluster_size": len(members),
                "labels": "|".join(sorted(set(labels))),
                "examples": " ||| ".join(sub[text_col].astype(str).head(3).tolist())
            }
            conflict_rows.append(row)
    return conflict_rows


def main():
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_parquet(in_path)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column not found: {args.text_col}")

    # Optional label
    has_label = args.label_col is not None and args.label_col in df.columns

    # Sample if requested
    if args.sample_size > 0 and args.sample_size < len(df):
        df = df.sample(args.sample_size, random_state=42).reset_index(drop=True)
        print(f"[info] Sampled {len(df)} rows for speed; set --sample_size 0 to run full corpus.")

    # Normalize
    df["norm_for_dupe"] = df[args.text_col].astype(str).map(norm_text)

    # Basic stats pre
    total_before = len(df)
    label_dist_before = None
    if has_label:
        label_dist_before = df[args.label_col].value_counts(normalize=True)

    # 1) Exact duplicate removal
    df_exact, exact_dropped, exact_conflicts = drop_exact_duplicates(df, args.text_col, args.label_col)
    print(f"[step] Dropped exact duplicates: {exact_dropped}")

    # 2) Blocking keys
    block_a, block_b = build_blocks(df_exact["norm_for_dupe"], args.block_prefix_len)
    df_exact = df_exact.copy()
    df_exact["_block_a"] = block_a
    df_exact["_block_b"] = block_b

    # 3) Within each block, find near-duplicate pairs
    # We'll scan both blocking keys, collect pairs in global indices
    seen_pairs = set()
    pairs_global = []  # (global_i, global_j, sim)

    def process_blocks(key_name: str):
        nonlocal seen_pairs, pairs_global
        for block, grp in df_exact.groupby(key_name, sort=False):
            if len(grp) <= 1:
                continue
            pairs = vectorize_and_find_pairs(
                texts=grp["norm_for_dupe"],
                min_df=args.min_df,
                ngram_min=args.ngram_min,
                ngram_max=args.ngram_max,
                threshold=args.threshold
            )
            if not pairs:
                continue
            idx_list = grp.index.to_list()
            for li, lj, sim in pairs:
                gi, gj = idx_list[li], idx_list[lj]
                # Ensure (min,max) uniqueness across both block scans
                key = (gi, gj) if gi < gj else (gj, gi)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs_global.append((key[0], key[1], sim))

    process_blocks("_block_a")
    process_blocks("_block_b")

    print(f"[step] Near-duplicate pairs flagged (threshold={args.threshold}): {len(pairs_global)}")

    # 4) Cluster pairs and choose exemplars
    keep, drop, clusters = set(), set(), {}
    if pairs_global:
        keep, drop, clusters = cluster_and_select(df_exact, pairs_global, args.keep_strategy, args.text_col)
    else:
        keep, drop, clusters = set(), set(), {}

    # Compose final kept indices: all minus drop
    all_idx = set(df_exact.index.tolist())
    final_keep = sorted(list(all_idx - drop))
    cleaned = df_exact.loc[final_keep].drop(columns=["_block_a","_block_b"]).reset_index(drop=True)

    # 5) Label conflicts from clusters (near-duplicates)
    conflicts = []
    if has_label and clusters:
        conflicts = collect_label_conflicts(df_exact, clusters, args.label_col, args.text_col)

    # 6) Save outputs
    out_clean = ensure_parent(args.output_clean)
    cleaned.to_parquet(out_clean, index=False)
    print(f"[save] Cleaned parquet → {out_clean} (rows: {len(cleaned)})")

    out_pairs = ensure_parent(args.output_pairs)
    if pairs_global:
        # Convert to a small CSV with global indices and similarity
        pairs_df = pd.DataFrame(pairs_global, columns=["idx_i","idx_j","similarity"])
        # Add short previews for audit
        pairs_df["text_i"] = df_exact.loc[pairs_df["idx_i"], args.text_col].values
        pairs_df["text_j"] = df_exact.loc[pairs_df["idx_j"], args.text_col].values
        pairs_df.to_csv(out_pairs, index=False)
        print(f"[save] Near-dupe pairs CSV → {out_pairs} (rows: {len(pairs_df)})")
    else:
        # Create an empty file for consistency
        pd.DataFrame(columns=["idx_i","idx_j","similarity","text_i","text_j"]).to_csv(out_pairs, index=False)
        print(f"[save] Near-dupe pairs CSV → {out_pairs} (rows: 0)")

    out_conf = ensure_parent(args.output_conflicts)
    conflicts_all = exact_conflicts + conflicts
    if conflicts_all:
        pd.DataFrame(conflicts_all).to_csv(out_conf, index=False)
        print(f"[save] Label conflicts CSV → {out_conf} (rows: {len(conflicts_all)})")
    else:
        pd.DataFrame(columns=["norm_for_dupe","labels","count","example_text","cluster_size","examples"]).to_csv(out_conf, index=False)
        print(f"[save] Label conflicts CSV → {out_conf} (rows: 0)")

    # 7) Report class distribution drift (if label present)
    if has_label:
        label_dist_after = cleaned[args.label_col].value_counts(normalize=True)
        print("[report] Class share before → after (only labels present in each phase are shown):")
        for lab in sorted(set(label_dist_before.index) | set(label_dist_after.index)):
            b = float(label_dist_before.get(lab, np.nan)) if label_dist_before is not None else np.nan
            a = float(label_dist_after.get(lab, np.nan))
            def fmt(x):
                return "NA" if np.isnan(x) else f"{100*x:.2f}%"
            print(f"  - {lab}: {fmt(b)} → {fmt(a)}")

    # 8) Final sanity
    print(f"[done] Total before: {total_before} | after exact: {len(df_exact)} | final cleaned: {len(cleaned)}")
    print("[note] Run this BEFORE splitting into train/val/test to avoid leakage.")

if __name__ == "__main__":
    main()
