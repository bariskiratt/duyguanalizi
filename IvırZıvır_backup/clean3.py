import os, csv, re, chardet
from typing import List, Tuple
from pathlib import Path

import pandas as pd
# from langdetect import detect  # Açmak istersen

# ────────────────────────────────────────────────────────────────
class DataProcessor:
    """
    • sample formatı:   <yorum>,<0|1|2>
    • hepsiburada:      Puan,Baslik,Yorum
    İkisini de “review_text, label” (pozitif/negatif/notr) şemasına çevirir.
    """

    LABEL_MAP = {
        0: "negatif", 1: "pozitif", 2: "notr",
        "0": "negatif", "1": "pozitif", "2": "notr",
        "Olumsuz": "negatif", "Olumlu": "pozitif", "Tarafsız": "notr",
    }

    # --- Hepsiburada puan aralığı → 0/1/2 -----------------------
    @staticmethod
    def score_to_label(score: int) -> int:
        if score >= 60:           # 60-100
            return 1              # pozitif
        elif score >= 40:         # 40-59
            return 2              # notr
        else:                     # 0-39
            return 0              # negatif

    # ------------------------------------------------------------
    def __init__(self, min_words: int = 3, lang_filter: bool = False):
        self.min_words = min_words
        self.lang_filter = lang_filter

    # ---------- kodlama ----------------------------------------
    @staticmethod
    def detect_encoding(path: str, sample: int = 16_000) -> str:
        with open(path, "rb") as f:
            return chardet.detect(f.read(sample)).get("encoding", "utf-8") or "utf-8"

    # ---------- csv okuma --------------------------------------
    def load_csv(self, path: str) -> pd.DataFrame:
        enc = self.detect_encoding(path)
        with open(path, "r", encoding=enc, errors="replace") as f:
            first = f.readline()
            f.seek(0)

        # Hepsiburada mı?
        if {"Puan", "Baslik", "Yorum"} <= set(first.split(",")):
            df = pd.read_csv(path, encoding=enc)
            df["label_raw"] = df["Puan"].astype(int).apply(self.score_to_label)
            df["review_text"] = (
                df["Yorum"].astype(str).str.strip()
            )  # Başlık eklemek istersen:  + " " + df["Baslik"].fillna("")
            return df[["review_text", "label_raw"]]

        # Aksi hâlde sample formatı
        #  dinamik ayıraç seç (virgül / noktalı virgül)
        best, best_rows = 0.0, []
        for delim in [",", ";"]:
            with open(path, "r", encoding=enc, errors="replace") as f:
                rdr = csv.reader(f, delimiter=delim, quotechar='"', skipinitialspace=True)
                rows = [r for r in rdr if len(r) >= 2]
            matches = sum(r[-1].strip() in self.LABEL_MAP for r in rows)
            if rows and matches / len(rows) > best:
                best, best_rows = matches / len(rows), rows

        return pd.DataFrame(
            [(r[0].strip(), r[-1].strip()) for r in best_rows],
            columns=["review_text", "label_raw"],
        )

    # ---------- basit temizleme --------------------------------
    @staticmethod
    def clean_txt(t: str) -> str:
        t = re.sub(r"http\S+|www\.\S+", "", t)
        t = re.sub(r"\S+@\S+", "", t)
        t = re.sub(r"\+?\d[\d\s-]{8,}\d", "", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    # ---------- tek dosya --------------------------------------
    def process_file(self, src: str, dst: str) -> int:
        df = self.load_csv(src)
        if df.empty:
            print(f"⚠️  {src} okunamadı/boş")
            return 0

        df["label"] = df["label_raw"].map(self.LABEL_MAP)
        df = df.dropna(subset=["label"])

        cleaned = []
        for txt, lbl in zip(df.review_text, df.label):
            txt = self.clean_txt(str(txt))
            if len(txt.split()) < self.min_words:
                continue
            cleaned.append({"review_text": txt, "label": lbl})

        if not cleaned:
            print(f"⚠️  {src}: filtre sonrası satır kalmadı")
            return 0

        pd.DataFrame(cleaned).to_parquet(dst, index=False)
        print(f"✔ {os.path.basename(dst)}  •  {len(cleaned)} satır")
        return len(cleaned)

    # ---------- çoklu dosya ------------------------------------
    def process_many(
        self,
        mappings: List[Tuple[str, str]],
        raw_dir="data/raw",
        out_dir="data/processed",
    ):
        total = 0
        for src, trg in mappings:
            p_src = Path(raw_dir, src)
            p_dst = Path(out_dir, trg)
            if not p_src.exists():
                print(f"⚠️  {src} yok – atlandı")
                continue
            os.makedirs(p_dst.parent, exist_ok=True)
            total += self.process_file(str(p_src), str(p_dst))
        print(f"🏁 Tamamlandı – toplam {total} satır")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    proc = DataProcessor(min_words=1, lang_filter=False)

    proc.process_many(
        mappings=[
            ("hepsiburada_data.csv", "hepsi_clean.parquet"),   # PUAN–BAŞLIK–YORUM
            ("sample_data2.csv", "train_data.parquet"),   # sample format
            ("sample_data.csv",  "test_data.parquet"),
        ]
    )
