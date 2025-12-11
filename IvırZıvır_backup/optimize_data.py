<<<<<<< HEAD
import pandas as pd
import re
import unicodedata
from ftfy import fix_text
from unidecode import unidecode

class TurkishTextCleaner:
    def __init__(self):
        # Common Turkish character corruptions from your error analysis
        self.turkish_fixes = {
            # Missing first letters
            r'\bıldız\b': 'yıldız',
            r'\beklerim\b': 'beklerim', 
            r'\beşekkürler\b': 'teşekkürler',
            r'\bayal ırıklığı\b': 'hayal kırıklığı',
            r'\bört\b': 'dört',
            r'\büper\b': 'süper',
            r'\bütlulukk\b': 'mutluluk',
            r'\bemnuniyet\b': 'memnuniyet',
            r'\biyiki\b': 'iyi ki',
            r'\bukemmel\b': 'mükemmel',
            r'\bükemmel\b': 'mükemmel',
            r'\bumuşacık\b': 'yumuşacık',
            
            # Brand name fixes
            r'\brima\b': 'prima',
            r'\bpirimaa\b': 'prima',
            r'\bpirima\b': 'prima',
            
            # Other common corruptions
            r'\bohada\b': 'daha da',
            r'\blmayin\b': 'almayın',
            r'\bakın\b': 'bakmayın',
            r'\berbat\b': 'berbat',
            r'\nternetten\b': 'internetten',
            r'\bnsana\b': 'insana',
        }
        
    def fix_encoding_issues(self, text):
        """Fix encoding and unicode issues"""
        if not isinstance(text, str):
            text = str(text)
        
        # Fix encoding issues
        text = fix_text(text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Fix common Turkish character issues
        text = text.replace('İ', 'i').replace('I', 'ı')  # Common Turkish casing issues
        
        return text
    
    def fix_turkish_typos(self, text):
        """Fix common Turkish typos and corruptions"""
        for pattern, replacement in self.turkish_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def clean_text(self, text):
        """Complete text cleaning pipeline"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string
        text = str(text).strip()
        
        if len(text) == 0:
            return ''
        
        # Step 1: Fix encoding issues
        text = self.fix_encoding_issues(text)
        
        # Step 2: Fix Turkish typos
        text = self.fix_turkish_typos(text)
        
        # Step 3: Remove URLs, HTML, mentions
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[@#]\w+', ' ', text)
        
        # Step 4: Keep only Turkish letters, numbers, and basic punctuation
        text = re.sub(r'[^a-zA-ZçğıöşüÇĞIıÖŞÜ0-9\s.,!?;:\'-]', ' ', text)
        
        # Step 5: Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

def main():
    """Enhanced data cleaning with better Turkish support"""
    
    print("🔄 Loading data for enhanced cleaning...")
    cleaner = TurkishTextCleaner()
    
    # Load your current data
    input_file = "data/processed/balanced_clean_dedup_sample.parquet"  # Adjust path if needed
    df = pd.read_parquet(input_file)
    
    print(f"📊 Original data: {len(df)} rows")
    print(f"📋 Columns: {df.columns.tolist()}")
    print(f"🏷️ Label distribution:\n{df['label'].value_counts()}")
    
    # Clean the text
    print("\n🧹 Applying enhanced text cleaning...")
    df['review_text_cleaned'] = df['review_text'].apply(cleaner.clean_text)
    
    # Remove empty texts after cleaning
    df = df[df['review_text_cleaned'].str.len() > 0]
    
    # Show some before/after examples
    print("\n📋 Cleaning examples:")
    for i in range(min(5, len(df))):
        original = df.iloc[i]['review_text']
        cleaned = df.iloc[i]['review_text_cleaned']
        if original != cleaned:
            print(f"Before: {original[:100]}...")
            print(f"After:  {cleaned[:100]}...")
            print("---")
    
    # Replace the review_text column
    df['review_text'] = df['review_text_cleaned']
    df = df.drop(columns=['review_text_cleaned'])
    
    # Save cleaned data
    output_file = "data/processed/balanced_clean_enhanced.parquet"
    df[['review_text', 'label']].to_parquet(output_file, index=False)
    
    print(f"\n✅ Enhanced cleaning completed!")
    print(f"📁 Saved to: {output_file}")
    print(f"📊 Final data: {len(df)} rows")
    print(f"🏷️ Final label distribution:\n{df['label'].value_counts()}")

if __name__ == "__main__":
    main()
