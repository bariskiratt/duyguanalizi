import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer


def load_data():
    """Veriyi yukle ve hazirla"""

    df = pd.read_parquet("data/processed/train_data.parquet")

    label_map = {'pozitif': 1, 'negatif': 0, 'notr': 2}
    df['label'] = df['label'].map(label_map)

    return df

def tokenize_data(texts,tokenizer,max_length=128):
    return tokenizer(texts,padding=True,truncation=True,max_length=max_length,return_tensors="pt")

def main():
    df = load_data()
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    train_df,val_df = train_test_split(df,test_size=0.2,random_state=42)

    train_encodings = tokenize_data(train_df['review_text'].tolist(),tokenizer)
    val_encodings = tokenize_data(val_df['review_text'].tolist(),tokenizer)

       # Dataset oluştur
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_df['label'].tolist()
    })
    
    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_df['label'].tolist()
    })
    
    # Kaydet
    train_dataset.save_to_disk("data/processed/bert_train")
    val_dataset.save_to_disk("data/processed/bert_val")




if __name__ == "__main__":
    main()