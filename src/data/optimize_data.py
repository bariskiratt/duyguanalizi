import pandas as pd
import re, unicodedata
try:
    from ftfy import fix_text
except Exception:
    fix_text = None

def normalize_unicode(text: str) -> str:
    s = str(text)
    if fix_text:
        s = fix_text(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u0069\u0307", "i")  # i + combining dot → i
    s = re.sub(r"\s+", " ", s).strip()
    return s

CORRECTIONS = {
    r"\brima\b": "prima",
    r"\bükemmel\b": "mükemmel",
    r"\bayal ırıklığı\b": "hayal kırıklığı",
}

def repair_typos(text: str) -> str:
    s = normalize_unicode(text)
    for pat, rep in CORRECTIONS.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s

df = pd.read_parquet("data/processed/balanced.parquet")
print(len(df),df.columns.tolist())
print(df['label'].value_counts())

df = df.dropna(subset=['review_text'])
df = df[df['review_text'].str.strip().str.len() > 0]

df['review_text'] = df['review_text'].astype(str)

df['text_norm'] = df['review_text']
# Unicode normalize + dar typo düzeltmeleri (Türkçe harfleri koruyarak)
df['text_norm'] = df['text_norm'].apply(repair_typos)
df['text_norm'] = df['text_norm'].str.replace(r'https?://\S+|www\.\S+', ' ', regex=True)   # URLs
df['text_norm'] = df['text_norm'].str.replace(r'<.*?>', ' ', regex=True)                   # HTML
df['text_norm'] = df['text_norm'].str.replace(r'[@#]\w+', ' ', regex=True)                 # @mention / #hashtag
df['text_norm'] = df['text_norm'].str.replace(r'[^A-Za-zÇĞİÖŞÜçğıöşü\s]', ' ', regex=True)          # keep TR letters (both cases) + spaces
df['text_norm'] = df['text_norm'].str.replace(r'\s+', ' ', regex=True).str.strip()         # collapse spaces (no trailing \)

# Çok kısa metinleri ele (>=3 kelime)
df = df[df['text_norm'].str.split().str.len() >= 3]


# If the same normalized text appears with multiple labels, take the most frequent (mode)
mode_labels = (df.groupby('text_norm')['label'].agg(lambda s: s.mode().iat[0]))
df = (df.drop(columns=['label'])
        .drop_duplicates(subset=['text_norm'])
        .merge(mode_labels.rename('label'), left_on='text_norm', right_index=True, how='left'))

df['review_text'] = df['text_norm']

out_path = "data/processed/balanced_clean.parquet"
df[['review_text', 'label']].to_parquet(out_path, index=False)
print(f"Saved: {out_path}, rows: {len(df)}")