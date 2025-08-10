from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path






#Ensures label is categorical(what chganges when its categorical?)
df = pd.read_parquet("data/processed/hepsiburada_bert_format.parquet")
df['label'] = df['label'].astype('category')
print(df['label'].value_counts())

classes = sorted(df['label'].unique().tolist())
counts = df['label'].value_counts().to_dict()

# Calculate the number of samples for each class
target_total = 250000
per_class_target = min(target_total // len(classes), min(counts.values()))
print(classes,counts,per_class_target)

balanced_parts = []
rng = 42
#what is cls?
for cls in classes:
    cls_df = df[df['label'] == cls]
    n = min(len(cls_df),per_class_target)
    balanced_parts.append(cls_df.sample(n,random_state=rng))


balanced = pd.concat(balanced_parts,ignore_index=True)
balanced = balanced.sample(frac=1,random_state=rng).reset_index(drop=True)

print(balanced['label'].value_counts())
print(len(balanced))

balanced = balanced.drop_duplicates(subset=['review_text'])
print(balanced['label'].value_counts())

#what is stratified?

#what is normalize?


#what are parents
out_dir = Path("data/processed/hepsiburada_balanced")
out_dir.mkdir(parents=True,exist_ok=True)
#what is index?
balanced.to_parquet(out_dir / "balanced.parquet",index=False)
