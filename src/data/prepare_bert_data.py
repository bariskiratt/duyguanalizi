import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import gc
import psutil

# Force CPU-only mode to avoid CUDA memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Try to import torch safely
try:
    import torch
    print("✅ PyTorch loaded successfully (CPU mode)")
except Exception as e:
    print(f"⚠️ PyTorch import warning: {e}")
    print("Continuing with transformers tokenizer only...")


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()

def load_data():
    """Veriyi yukle ve hazirla"""
    
    print("🔄 Loading data...")
    df = pd.read_parquet("duyguanalizi/data/processed/hepsiburada_bert_format.parquet")
    print(f"✅ Loaded {len(df):,} samples")
    
    print("🔄 Mapping labels...")
    label_map = {'pozitif': 1, 'negatif': 0, 'notr': 2}
    
    # Create a function for progress_map (it needs a callable, not a dict)
    def map_label(label):
        return label_map.get(label, 2)  # Default to 'notr' (2) if not found
    
    # Add progress bar for label mapping
    tqdm.pandas(desc="Mapping labels")
    df['label'] = df['label'].progress_map(map_label)
    
    print("✅ Label mapping completed")
    print(f"📊 Label distribution:")
    for label, count in df['label'].value_counts().sort_index().items():
        label_name = {0: 'Negatif', 1: 'Pozitif', 2: 'Nötr'}[label]
        print(f"   {label_name}: {count:,}")
    
    return df

def tokenize_data(texts, tokenizer, max_length=128, desc="Tokenizing"):
    """Tokenize texts without immediate tensor conversion - Memory optimized"""
    print(f"🔄 {desc} {len(texts):,} texts...")
    print(f"💾 Memory before tokenization: {get_memory_usage():.1f} MB")
    
    # Use smaller batches to prevent OOM
    batch_size = 100  # Reduced from 1000 to 100
    all_encodings = {'input_ids': [], 'attention_mask': []}
    
    # Create progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        batch_encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors=None  # Don't convert to tensors yet
        )
        
        # Accumulate results
        all_encodings['input_ids'].extend(batch_encodings['input_ids'])
        all_encodings['attention_mask'].extend(batch_encodings['attention_mask'])
        
        # Clean up every 50 batches to free memory
        if i % (batch_size * 50) == 0:
            cleanup_memory()
    
    print(f"✅ {desc} completed!")
    print(f"💾 Memory after tokenization: {get_memory_usage():.1f} MB")
    return all_encodings

def main():
    print("🚀 Starting BERT data preparation...")
    print("=" * 50)
    print(f"💾 Initial memory usage: {get_memory_usage():.1f} MB")
    
    # Step 1: Load data
    df = load_data()
    print(f"💾 Memory after loading: {get_memory_usage():.1f} MB")
    
    # Step 2: Load tokenizer
    print("\n🔄 Loading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    print("✅ Tokenizer loaded successfully!")
    
    # Step 3: Split data
    print(f"\n🔄 Splitting data (80% train, 20% validation)...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"✅ Data split completed:")
    print(f"   Training samples: {len(train_df):,}")
    print(f"   Validation samples: {len(val_df):,}")
    
    # Clean up original dataframe to free memory
    del df
    cleanup_memory()
    print(f"💾 Memory after data split: {get_memory_usage():.1f} MB")

    # Step 4: Tokenize the data
    print("\n" + "=" * 50)
    train_encodings = tokenize_data(
        train_df['review_text'].tolist(), 
        tokenizer, 
        desc="Tokenizing training data"
    )
    
    val_encodings = tokenize_data(
        val_df['review_text'].tolist(), 
        tokenizer, 
        desc="Tokenizing validation data"
    )
    
    # Clean up text data to free memory
    cleanup_memory()
    print(f"💾 Memory after tokenization: {get_memory_usage():.1f} MB")

    # Step 5: Create datasets
    print("\n🔄 Creating Hugging Face datasets...")
    
    print("   Creating training dataset...")
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_df['label'].tolist()
    })
    
    print("   Creating validation dataset...")
    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_df['label'].tolist()
    })
    print("✅ Datasets created successfully!")
    
    # Step 6: Save datasets
    print("\n🔄 Saving datasets...")
    
    # Create directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    
    print("   Saving training dataset...")
    train_dataset.save_to_disk("data/processed/bert_train")
    
    print("   Saving validation dataset...")
    val_dataset.save_to_disk("data/processed/bert_val")
    
    # Clean up encodings to free memory
    del train_encodings, val_encodings
    cleanup_memory()
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 BERT data preparation completed successfully!")
    print(f"📊 Final Summary:")
    print(f"   Training dataset size: {len(train_dataset):,}")
    print(f"   Validation dataset size: {len(val_dataset):,}")
    print(f"   Saved to: data/processed/bert_train & data/processed/bert_val")
    print(f"💾 Final memory usage: {get_memory_usage():.1f} MB")
    print("=" * 50)


if __name__ == "__main__":
    main()