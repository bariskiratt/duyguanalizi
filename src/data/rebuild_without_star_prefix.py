"""Tokenize edilmis dataset'ten yildiz oneki sizintisini temizler.

Normalde bu temizlik create_balanced_dataset.py icinde, ham parquet uzerinde
yapilir. Ama bu repoda ham veri yok (.gitignore parquet'leri disliyor ve
data/raw/ bos), elde yalnizca tokenize edilmis HF dataset'i var. Bu script o
durumda dataset'i yeniden uretir:

    input_ids -> metne cevir -> yildiz onegini sil -> yeniden tokenize et

Cevirme turu bu veri uzerinde kayipsiz dogrulandi (2000 ornekte 2000 tam
eslesme), yani yeniden tokenize etmek orijinal input_ids'i bire bir uretir.

Kullanim:
    python src/data/rebuild_without_star_prefix.py
    python src/data/rebuild_without_star_prefix.py --apply   # eskisini yedekleyip yerine koy
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src" / "data"))

from create_balanced_dataset import strip_star_prefix  # noqa: E402

SPLITS = ["bert_train", "bert_val"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default=str(_REPO_ROOT / "data" / "processed"))
    p.add_argument("--config", default=str(_REPO_ROOT / "src" / "configs" / "bert_hparams.yaml"))
    p.add_argument("--suffix", default="_clean", help="Yeni dataset klasor soneki")
    p.add_argument("--apply", action="store_true",
                   help="Uretilen temiz seti orijinalin yerine koy (eskisi _with_leak olarak yedeklenir)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    max_len = cfg["model"].get("max_length", 256)
    data_dir = Path(args.data_dir)

    for split in SPLITS:
        kaynak = data_dir / split
        hedef = data_dir / f"{split}{args.suffix}"
        print(f"\n=== {split} ===")
        ds = load_from_disk(str(kaynak))
        onceki = len(ds)
        print(f"  giris: {onceki:,} satir")

        def donustur(batch):
            metinler = tok.batch_decode(batch["input_ids"], skip_special_tokens=True)
            temiz = [strip_star_prefix(t) for t in metinler]
            enc = tok(temiz, truncation=True, max_length=max_len)
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": batch["labels"],
                "_degisti": [a != b for a, b in zip(metinler, temiz)],
                "_bos": [len(t.strip()) == 0 for t in temiz],
            }

        ds = ds.map(donustur, batched=True, batch_size=1000,
                    remove_columns=ds.column_names, desc="  onek temizleniyor")

        degisen = sum(ds["_degisti"])
        bos = sum(ds["_bos"])
        # Sadece onekten ibaret olan yorumlar temizlikten sonra bos kaliyor;
        # icerigi olmadigi icin atiyoruz.
        ds = ds.filter(lambda x: not x["_bos"], desc="  bos satirlar atiliyor")
        ds = ds.remove_columns(["_degisti", "_bos"])

        print(f"  onek silinen : {degisen:,} (%{degisen/onceki*100:.1f})")
        print(f"  bosalip atilan: {bos:,}")
        print(f"  cikis: {len(ds):,} satir")

        if hedef.exists():
            shutil.rmtree(hedef)
        ds.save_to_disk(str(hedef))
        print(f"  yazildi: {hedef}")

    if args.apply:
        print("\n=== yerine koyuluyor ===")
        for split in SPLITS:
            asil = data_dir / split
            yedek = data_dir / f"{split}_with_leak"
            yeni = data_dir / f"{split}{args.suffix}"
            if yedek.exists():
                shutil.rmtree(yedek)
            asil.rename(yedek)
            yeni.rename(asil)
            print(f"  {split}: eski -> {yedek.name}, yeni -> {split}")
        print("\nEgitimi yeniden calistirabilirsin. Onceki checkpoint'ler artik")
        print("farkli bir dagilima ait; artifacts/bert_mlp_ckpt icini temizle.")
    else:
        print(f"\nOn izleme bitti. Yerine koymak icin --apply ile calistir.")


if __name__ == "__main__":
    main()
