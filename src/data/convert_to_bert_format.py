"""
Convert hepsiburada_dataset.parquet to BERT training format.

Input format:
- Puan: 0-100 score
- Baslik: review title  
- Yorum: review text

Output format (matching train_data.parquet):
- review_text: combined Baslik + Yorum
- label: "pozitif", "notr", or "negatif"

Score mapping:
- score <= 40 → "negatif" 
- score >= 60 → "pozitif"
- otherwise → "notr"

Usage:
  python -m duyguanalizi.src.data.convert_to_bert_format
"""

import pandas as pd
from pathlib import Path


def convert_to_bert_format(
    input_path: str = "data/processed/hepsi_clean.parquet",
    output_path: str = "data/processed/hepsi_bert_format.parquet",
    neg_max: int = 40,
    pos_min: int = 60,
    notr_score: int= 60
) -> None:
    """Convert hepsiburada dataset to BERT training format."""
    
    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Input shape: {df.shape}")
    print(f"Input columns: {list(df.columns)}")
    
    # Combine Baslik and Yorum into review_text
    print("Combining Baslik and Yorum...")
    df['review_text'] = df['Baslik'].fillna('') + ' ' + df['Yorum'].fillna('')
    df['review_text'] = df['review_text'].str.strip()
    
    # Map scores to Turkish labels
    print("Mapping scores to labels...")
    def map_score_to_label(score):
        if pd.isna(score):
            return "notr"
        score = float(score)
        if score <= neg_max:
            return "negatif"
        elif score > pos_min:
            return "pozitif"
        else:
            return "notr"
    
    df['label'] = df['Puan'].apply(map_score_to_label)
    
    # Select only the required columns
    result_df = df[['review_text', 'label']].copy()
    
    # Show label distribution
    print("\nLabel distribution:")
    print(result_df['label'].value_counts())
    
    # Remove rows with empty review_text
    result_df = result_df[result_df['review_text'].str.len() > 0]
    print(f"\nFinal shape after removing empty reviews: {result_df.shape}")
    
    # Save to parquet
    print(f"\nSaving to {output_path}...")
    result_df.to_parquet(output_path, index=False)
    print("Done!")
    
    # Verify the output format matches train_data
    print("\nVerifying output format...")
    verify_df = pd.read_parquet(output_path)
    print(f"Output columns: {list(verify_df.columns)}")
    print(f"Output labels: {sorted(verify_df['label'].unique())}")
    print(f"Sample output:")
    print(verify_df.head(3))


if __name__ == "__main__":
    convert_to_bert_format()
