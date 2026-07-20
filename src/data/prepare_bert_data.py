import pandas as pd
import numpy as np
from datasets import Dataset, Features, Value, Sequence
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import gc
import psutil
import torch
import yaml
import json
from pathlib import Path


def _find_label_config():
    """Locate src/configs/bert_hparams.yaml by walking up from this file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src" / "configs" / "bert_hparams.yaml"
        if candidate.exists():
            return candidate
    return Path("src/configs/bert_hparams.yaml")


def load_label_vocab():
    """Ordered label vocabulary - the single source of truth for class ids.

    List position is the class id, so switching between the binary and the
    3-class setup is a config edit, not a code edit. Returns (labels, label2id,
    id2label).
    """
    with open(_find_label_config(), "r", encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f)
    labels = _cfg.get("labels")
    if not labels:
        raise ValueError(
            "bert_hparams.yaml is missing the 'labels' list; it defines the "
            "label -> class id mapping used by preparation, training and evaluation."
        )
    label2id = {name: i for i, name in enumerate(labels)}
    return labels, label2id, {i: name for name, i in label2id.items()}


# --- Sistem Ayarları ---
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def get_memory_usage():
    """Mevcut bellek kullanımını MB cinsinden döndürür"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    """Belleği temizler"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_system_info():
    """Sistem kaynak bilgilerini yazdırır"""
    print("🖥️  Sistem Bilgileri:")
    print(f"   CPU Çekirdekleri: {psutil.cpu_count()}")
    print(f"   Toplam RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print(f"   Kullanılabilir RAM: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    if use_cuda:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

def process_labels(labels):
    """
    String etiketleri (negatif/pozitif) integer (0/1) formatına çevirir.
    BERT eğitimi için kritik adımdır.
    """
    print(f"🔄 Etiketler işleniyor ({len(labels):,} adet)...")
    _, label_map, _ = load_label_vocab()
    processed = []
    
    for x in labels:
        if isinstance(x, str):
            clean_x = x.strip().lower()
            # Bilinmeyen etiket gelirse varsayılan olarak 0 ata veya hata ver
            processed.append(label_map.get(clean_x, 0))
        elif isinstance(x, (int, float)):
            processed.append(int(x))
        else:
            processed.append(0)
            
    return processed

def tokenize_data(texts, tokenizer, desc="Tokenizing"):
    """
    Metinleri tokenize eder - Bellek dostu batch işlemi
    """
    print(f"🔄 {desc}...")
    
    # 1. Metin Temizliği
    cleaned_texts = []
    for text in texts:
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            cleaned_texts.append("") # Boş metinler için yer tutucu
        else:
            cleaned_texts.append(str(text).strip())
    
    # 2. Batch İşlemi
    batch_size = 100 # Hız için artırılabilir, bellek hatası alırsanız 25'e düşürün
    all_input_ids = []
    all_attention_masks = []
    
    total_batches = (len(cleaned_texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(cleaned_texts), batch_size), desc=desc, total=total_batches):
        batch_texts = cleaned_texts[i:i + batch_size]
        
        # Sadece bu batch'i tokenize et
        # Padding='max_length' yerine dinamik padding veya sabit uzunluk kullanıyoruz
        batch_encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length', # Dataset oluştururken tutarlılık için
            max_length=256,       # Bert_train için uygun uzunluk
            return_tensors=None
        )
        
        all_input_ids.extend(batch_encodings['input_ids'])
        all_attention_masks.extend(batch_encodings['attention_mask'])
        
        # Periyodik bellek temizliği
        if i % (batch_size * 10) == 0:
            cleanup_memory()
            
    return {'input_ids': all_input_ids, 'attention_mask': all_attention_masks}

def main():
    try:
        print("🚀 BERT Veri Hazırlama Başlıyor...")
        print("=" * 50)
        print_system_info()
        
        # Veri dizinini bul (script konumundan 2 seviye yukarı)
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent.parent  # src/data -> src -> repo_root (duyguanalizi)
        data_dir = repo_root / "data" / "processed"
        train_path = data_dir / "train.parquet"
        val_path = data_dir / "test.parquet"
        output_dir = data_dir  # Çıktı doğrudan processed altında
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📍 Dosya konumları:")
        print(f"   Train: {train_path}")
        print(f"   Val: {val_path}")
        print(f"   Output: {output_dir}")
        
        # 1. Tokenizer Yükle
        model_name = "dbmdz/bert-base-turkish-cased"
        print(f"\n🔄 Tokenizer yükleniyor ({model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 2. Verisetlerini Yükle
        print("\n📂 Parquet dosyaları okunuyor...")
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)  # test.parquet'i validation olarak kullanıyoruz
        
        print(f"   Eğitim verisi: {len(train_df):,} satır")
        print(f"   Doğrulama verisi: {len(val_df):,} satır")

        # 3. Etiketleri Dönüştür (ÖNEMLİ ADIM)
        vocab, label2id, _ = load_label_vocab()
        print(f"\n🏷️ Etiketler dönüştürülüyor ({label2id})...")

        # Persist the mapping next to the data so evaluation artifacts can always
        # be traced back to which class id meant which sentiment.
        with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
            json.dump({"labels": vocab, "label2id": label2id}, f,
                      ensure_ascii=False, indent=2)
        train_labels = process_labels(train_df['label'].tolist())
        val_labels = process_labels(val_df['label'].tolist())
        
        # 4. Tokenizasyon
        print("\n🔄 Tokenizasyon başlıyor...")
        train_encodings = tokenize_data(train_df['review_text'].tolist(), tokenizer, desc="Train Tokenizing")
        cleanup_memory()
        
        val_encodings = tokenize_data(val_df['review_text'].tolist(), tokenizer, desc="Val Tokenizing")
        cleanup_memory()
        
        # 5. Hugging Face Dataset Oluşturma ve Kaydetme
        print("\n💾 Datasetler diske kaydediliyor...")
        
        # Veri şeması tanımla (Performans ve tip güvenliği için)
        features = Features({
            'input_ids': Sequence(Value('int32')),
            'attention_mask': Sequence(Value('int8')),
            'labels': Value('int64') # Trainer int64 bekler
        })

        # Train Dataset
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        }, features=features)
        
        train_save_path = output_dir / "bert_train"
        train_dataset.save_to_disk(str(train_save_path))
        print(f"✅ Eğitim seti kaydedildi: {train_save_path}")
        
        # Bellek temizliği
        del train_dataset, train_encodings, train_labels
        cleanup_memory()

        # Validation Dataset
        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': val_labels
        }, features=features)
        
        val_save_path = output_dir / "bert_val"
        val_dataset.save_to_disk(str(val_save_path))
        print(f"✅ Doğrulama seti kaydedildi: {val_save_path}")

        print("\n" + "=" * 50)
        print("🎉 Hazırlık Tamamlandı! Artık 'bert_train.py' çalıştırılabilir.")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()