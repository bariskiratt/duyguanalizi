import os
import csv
import re
from typing import List, Tuple
from langdetect import detect
import pandas as pd
import chardet
# from langdetect import detect  # Türkçe dil filtresi kullanacaksan aç

# ────────────────────────────────────────────────────────────────
class DataProcessor:
    """Metin verilerini temizleyip Parquet’e dönüştüren sınıf."""

    # String ve sayısal etiketlerin hepsini tek haritaya topladık
    LABEL_MAPPING = {
        "Olumsuz": "negatif",
        "Olumlu":  "pozitif",
        "Tarafsız": "notr",
        "0": "negatif", "1": "pozitif", "2": "notr",
         0 : "negatif",  1 : "pozitif",  2 : "notr",
    }

    def __init__(self, min_word_count: int = 3, use_language_filter: bool = False):
        self.min_word_count = min_word_count
        self.use_language_filter = use_language_filter

    # ------------------------------------------------------------
    @staticmethod
    def detect_file_encoding(file_path: str, sample_size: int = 10000) -> str:
        """Dosya kodlamasını otomatik tespit et (UTF-16 dâhil)."""
        with open(file_path, "rb") as f:
            raw = f.read(sample_size)
        return chardet.detect(raw).get("encoding", "utf-8") or "utf-8"

    # ------------------------------------------------------------
    def load_csv_safely(self, file_path: str) -> pd.DataFrame:
        encoding = self.detect_file_encoding(file_path)

        best_rows, best_match = [], 0.0
        for delim in [",", ";"]:
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                reader = csv.reader(
                    f, delimiter=delim, quotechar='"', skipinitialspace=True
                )
                next(reader, None)  # başlık
                rows = [row for row in reader if len(row) >= 2]

            if not rows:
                continue

            # Etiket eşleşme yüzdesi
            matches = sum(1 for r in rows if r[-1].strip() in self.LABEL_MAPPING)
            match_ratio = matches / len(rows)

            if match_ratio > best_match:
                best_rows, best_match = rows, match_ratio

        # Sonuç DataFrame
        return pd.DataFrame(
            [(r[0].strip(), r[-1].strip()) for r in best_rows],
            columns=["review_text", "label_raw"],
        )
    # ------------------------------------------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        """Basit metin temizleme (URL, mail, tel, fazla boşluk)."""
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"\+?\d[\d\s-]{8,}\d", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ------------------------------------------------------------
    def is_valid_text(self, text: str) -> bool:
        """Minimum kelime sayısı ve (isteğe bağlı) Türkçe kontrolü."""
        if len(text.split()) < self.min_word_count:
            return False

        if self.use_language_filter:
            try:
                from langdetect import detect
                if detect(text) != "tr":
                    return False
            except Exception:
                pass  # tespit edilemezse kabul et
        return True

    # ------------------------------------------------------------
    def process_single_file(self, src: str, trg: str) -> int:
        print(f" İşleniyor: {os.path.basename(src)}")

        df = self.load_csv_safely(src)
        if df.empty:
            print(f" Dosya boş veya okunamadı: {src}")
            return 0

        df["label"] = df["label_raw"].map(self.LABEL_MAPPING)
        df = df.dropna(subset=["label"])

        processed = []
        for _, row in df.iterrows():
            txt = self.clean_text(row.review_text)
            if self.is_valid_text(txt):
                processed.append({"review_text": txt, "label": row.label})

        if processed:
            out_df = pd.DataFrame(processed)
            os.makedirs(os.path.dirname(trg), exist_ok=True)
            out_df.to_parquet(trg, index=False)
            print(f" {os.path.basename(trg)} • {len(out_df)} satır kaydedildi")
            return len(out_df)

        print(" İşlenecek geçerli satır bulunamadı")
        return 0

    # ------------------------------------------------------------
    def process_multiple_files(
        self,
        file_mappings: List[Tuple[str, str]],
        raw_dir: str = "data/raw",
        out_dir: str = "data/processed",
    ):
        print(" Veri işleme başladı\n")
        total = 0
        for src_file, trg_file in file_mappings:
            src_path = os.path.join(raw_dir, src_file)
            trg_path = os.path.join(out_dir, trg_file)
            if not os.path.exists(src_path):
                print(f"⚠️ Dosya bulunamadı: {src_file} – atlandı\n")
                continue
            total += self.process_single_file(src_path, trg_path)
            print()
        print(f" İşlem tamamlandı • Toplam {total} satır işlendi.")


# ────────────────────────────────────────────────────────────────
def main():
    processor = DataProcessor(min_word_count=3, use_language_filter=False)
    mappings = [
        ("sample2_data.csv", "train_data.parquet"),
        ("sample_data.csv",  "test_data.parquet"),
    ]
    processor.process_multiple_files(mappings)


if __name__ == "__main__":
    main()
