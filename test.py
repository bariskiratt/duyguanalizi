import pandas as pd
f=pd.read_parquet('data/processed/balanced.parquet')
print(f"Number of rows: {f.shape[0]}")
f.dropna(inplace=True)
f.to_parquet('data/processed/balanced_cleaned.parquet')
print(f"Number of rows: {f.shape[0]}")
print(f"Count of each label: {f['label'].value_counts()}")