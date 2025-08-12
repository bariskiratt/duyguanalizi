import pyarrow.parquet as pq; 
f=pq.ParquetFile('data/processed/merged.parquet')
print(f'merged rows:', f.metadata.num_rows)