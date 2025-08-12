import pandas as pd

df = pd.read_parquet("data/processed/balanced.parquet")
print(len(df),df.columns.tolist())
print(df['label'].value_counts())

df = df.dropna(subset=['review_text'])
df = df[df['review_text'].str.strip().str.len() > 0]

df['review_text'] = df['review_text'].astype(str)

df['text_norm'] = df['review_text'].str.lower()
df['text_norm'] = df['text_norm'].str.replace(r'https?://\S+|www\.\S+', ' ', regex=True)   # URLs
df['text_norm'] = df['text_norm'].str.replace(r'<.*?>', ' ', regex=True)                   # HTML
df['text_norm'] = df['text_norm'].str.replace(r'[@#]\w+', ' ', regex=True)                 # @mention / #hashtag
df['text_norm'] = df['text_norm'].str.replace(r'[^a-zçğıöşü\s]', ' ', regex=True)          # keep TR letters + spaces
df['text_norm'] = df['text_norm'].str.replace(r'\s+', ' ', regex=True).str.strip()         # collapse spaces (no trailing \)


# If the same normalized text appears with multiple labels, take the most frequent (mode)
mode_labels = (df.groupby('text_norm')['label'].agg(lambda s: s.mode().iat[0]))
df = (df.drop(columns=['label'])
        .drop_duplicates(subset=['text_norm'])
        .merge(mode_labels.rename('label'), left_on='text_norm', right_index=True, how='left'))

df['review_text'] = df['text_norm']

out_path = "data/processed/balanced_clean.parquet"
df[['review_text', 'label']].to_parquet(out_path, index=False)
print(f"Saved: {out_path}, rows: {len(df)}, value counts: {df['label'].value_counts()}")