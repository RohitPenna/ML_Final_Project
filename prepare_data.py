"""
Combines Fake.csv and True.csv, shuffles, and splits into
train (70%), validation (15%), and test (15%) sets.
Saves splits to data/splits/.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# Load and label
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 1  # 1 = fake
true["label"] = 0  # 0 = real

# Strip whitespace from column names and string fields
for df in [fake, true]:
    df.columns = df.columns.str.strip()
    df["title"] = df["title"].str.strip()
    df["subject"] = df["subject"].str.strip()
    df["date"] = df["date"].str.strip()

# Combine and shuffle
combined = pd.concat([fake, true], ignore_index=True)
combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Split: 70% train, 15% val, 15% test  (stratified on label)
train, temp = train_test_split(
    combined, test_size=0.30, random_state=SEED, stratify=combined["label"]
)
val, test = train_test_split(
    temp, test_size=0.50, random_state=SEED, stratify=temp["label"]
)

# Save
import os
os.makedirs("data/splits", exist_ok=True)
train.to_csv("data/splits/train.csv", index=False)
val.to_csv("data/splits/val.csv", index=False)
test.to_csv("data/splits/test.csv", index=False)

# Print summary
total = len(combined)
print(f"Total articles : {total:,}")
print(f"  Fake         : {combined['label'].sum():,}  ({combined['label'].mean()*100:.1f}%)")
print(f"  Real         : {(combined['label']==0).sum():,}  ({(combined['label']==0).mean()*100:.1f}%)")
print()
for name, df in [("Train", train), ("Validation", val), ("Test", test)]:
    fake_n = df["label"].sum()
    real_n = (df["label"] == 0).sum()
    print(f"{name:12s}: {len(df):,} rows  |  Fake: {fake_n:,} ({fake_n/len(df)*100:.1f}%)  |  Real: {real_n:,} ({real_n/len(df)*100:.1f}%)")
