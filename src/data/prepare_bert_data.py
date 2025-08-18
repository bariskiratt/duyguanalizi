import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import gc
import psutil
import time
import torch





use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

if use_cuda:
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()

def print_system_info():
    """Print system resource information"""
    print("🖥️ System Information:")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   Total RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print(f"   Available RAM: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    print(f"   RAM usage: {psutil.virtual_memory().percent:.1f}%")



def tokenize_data(texts, tokenizer, desc="Tokenizing"):
    """Tokenize texts without immediate tensor conversion - Memory optimized"""
    print(f"🔄 {desc} {len(texts):,} texts...")
    print(f"💾 Memory before tokenization: {get_memory_usage():.1f} MB")
    
    # Clean and validate texts before tokenization
    cleaned_texts = []
    for i, text in enumerate(texts):
        if pd.isna(text) or not isinstance(text, str) or len(str(text).strip()) == 0:
            print(f"⚠️ Warning: Invalid text at index {i}: {text}")
            cleaned_texts.append("")  # Use empty string for invalid texts
        else:
            cleaned_texts.append(str(text).strip())
    
    # Use smaller batches to prevent OOM
    batch_size = 25  # Reduced from 100 to 25
    all_encodings = {'input_ids': [], 'attention_mask': []}
    
    total_batches = (len(cleaned_texts) + batch_size - 1) // batch_size
    print(f"📊 Processing {total_batches} batches of size {batch_size}")
    
    # Create progress bar
    for batch_idx, i in enumerate(tqdm(range(0, len(cleaned_texts), batch_size), desc=desc)):
        try:
            batch_texts = cleaned_texts[i:i + batch_size]
            
            # Show progress every 5 batches
            if batch_idx % 5 == 0:
                print(f"   Batch {batch_idx}/{total_batches}, Memory: {get_memory_usage():.1f} MB")
            
            # Filter out empty texts
            valid_batch = [text for text in batch_texts if text.strip()]
            
            if not valid_batch:
                # If all texts in batch are empty, create dummy encodings
                dummy_encoding = tokenizer(
                    [""], 
                    truncation=True,
                    return_tensors=None,
                    max_length=256  # Reduced from 312 to 256
                )
                for _ in range(len(batch_texts)):
                    all_encodings['input_ids'].append(dummy_encoding['input_ids'][0])
                    all_encodings['attention_mask'].append(dummy_encoding['attention_mask'][0])
            else:
                # Tokenize valid batch
                batch_encodings = tokenizer(
                    valid_batch,
                    truncation=True,
                    return_tensors=None,
                    max_length=256  # Reduced from 312 to 256
                )
                
                # Handle mixed valid/invalid texts in batch
                batch_idx_encoding = 0
                for text in batch_texts:
                    if text.strip():
                        all_encodings['input_ids'].append(batch_encodings['input_ids'][batch_idx_encoding])
                        all_encodings['attention_mask'].append(batch_encodings['attention_mask'][batch_idx_encoding])
                        batch_idx_encoding += 1
                    else:
                        # Add dummy encoding for empty text
                        dummy_encoding = tokenizer(
                            [""], 
                            truncation=True,
                            return_tensors=None,
                            max_length=256
                        )
                        all_encodings['input_ids'].append(dummy_encoding['input_ids'][0])
                        all_encodings['attention_mask'].append(dummy_encoding['attention_mask'][0])
            
            # Clean up every 10 batches to free memory
            if batch_idx % 10 == 0:
                cleanup_memory()
                
        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {e}")
            print(f"   Batch texts sample: {batch_texts[:2] if batch_texts else 'None'}")
            raise e
    
    print(f"✅ {desc} completed!")
    print(f"💾 Memory after tokenization: {get_memory_usage():.1f} MB")
    return all_encodings

def main():
    try:
        print("🚀 Starting BERT data preparation...")
        print("=" * 50)
        
        # Print system information
        print_system_info()
        print(f"💾 Initial memory usage: {get_memory_usage():.1f} MB")
    
        
        # Check available memory
        available_memory = psutil.virtual_memory().available / 1024 / 1024
        print(f"💾 Available system memory: {available_memory:.1f} MB")
        
        if available_memory < 1000:  # Less than 1GB
            print("⚠️ Warning: Low memory available!")
        
        # Step 2: Load tokenizer
        print("\n🔄 Loading BERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        print("✅ Tokenizer loaded successfully!")
        
        # Step 3: Load separate train and validation datasets
        print(f"\n🔄 Loading separate train and validation datasets...")

        # Load training data from one file
        train_df = pd.read_parquet("data/processed/train_final.parquet")
        print(f"✅ Training data loaded: {len(train_df):,} samples")

        # Load validation data from another file  
        val_df = pd.read_parquet("data/processed/test.parquet")
        print(f"✅ Validation data loaded: {len(val_df):,} samples")

        print(f"📊 Dataset sizes:")
        print(f"   Training samples: {len(train_df):,}")
        print(f"   Validation samples: {len(val_df):,}")
        

        
        # Test with small subset first
        print("\n🧪 Testing with small subset...")
        test_df = train_df.head(50)  # Only first 50 samples
        test_encodings = tokenize_data(
            test_df['review_text'].tolist(), 
            tokenizer, 
            desc="Testing tokenization"
        )
        print(f"✅ Test completed with {len(test_encodings['input_ids'])} samples")
        
        # Step 4: Tokenize the data
        print("\n" + "=" * 50)
        print("🔄 Starting full tokenization...")
        
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
        
        # Check if we have data after tokenization
        if len(train_encodings['input_ids']) == 0 or len(val_encodings['input_ids']) == 0:
            print("❌ No valid data after tokenization!")
            return
        
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
        print(f"   Saved to: duyguanalizi/data/processed/bert_train & duyguanalizi/data/processed/bert_val")
        print(f"💾 Final memory usage: {get_memory_usage():.1f} MB")
        print("=" * 50)
        
    except MemoryError as e:
        print(f"❌ Memory error: {e}")
        print("💡 Try reducing batch size or processing smaller chunks")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()