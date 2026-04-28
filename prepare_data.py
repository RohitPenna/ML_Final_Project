"""
Combines Fake.csv and True.csv, cleans titles to remove easy shortcuts,
and splits into train (70%), validation (15%), and test (15%) sets.

Cleaning steps applied to remove data-leakage shortcuts:
  1. Drop `subject` and `date` — force models to work on text only
  2. Strip trailing source attributions from titles  (e.g. ": Reuters", ": CNN", ": NYT")
  3. Strip leading format prefixes                   (e.g. "WATCH:", "Factbox:", "UPDATE:")
  4. Strip parenthetical media tags                  (e.g. "(VIDEO)", "(IMAGES)", "(TWEET)")
  5. Keep only `title` and `label` — title-only is a harder signal than full article text

Final columns: title (str), label (int: 0=real, 1=fake)
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# ---------------------------------------------------------------------------
# Source suffixes appended to real Reuters/wire titles after a colon
# ---------------------------------------------------------------------------
SOURCE_SUFFIXES = re.compile(
    r":\s*(reuters|nyt|cnn|bbc|ap|afp|pentagon|white house|report|sources?|"
    r"poll|study|data|officials?|documents?|survey|exclusive)\s*$",
    re.IGNORECASE,
)

# Leading format labels common in both real ("Factbox:") and fake ("WATCH:")
LEADING_LABELS = re.compile(
    r"^\s*(watch|factbox|update|breaking|video|photos?|exclusive)\s*[:\-]\s*",
    re.IGNORECASE,
)

# Parenthetical media/format tags "(VIDEO)", "(IMAGES)", "(TWEET)", "(GRAPHIC IMAGES)" etc.
PAREN_TAGS = re.compile(r"\(\s*(video|images?|photos?|tweet|watch|graphic[^)]*)\s*\)", re.IGNORECASE)


def clean_title(title: str) -> str:
    title = str(title).strip()
    title = PAREN_TAGS.sub("", title)          # remove (VIDEO), (IMAGES) …
    title = LEADING_LABELS.sub("", title)      # remove WATCH:, Factbox: …
    title = SOURCE_SUFFIXES.sub("", title)     # remove : Reuters, : CNN …
    return title.strip()


# ---------------------------------------------------------------------------
# Load & label
# ---------------------------------------------------------------------------
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake.columns = fake.columns.str.strip()
true.columns = true.columns.str.strip()

fake["label"] = 1   # 1 = fake
true["label"] = 0   # 0 = real

# Keep only title + label; drop subject, date, text
for df in [fake, true]:
    df["title"] = df["title"].str.strip().apply(clean_title)

combined = pd.concat(
    [fake[["title", "label"]], true[["title", "label"]]],
    ignore_index=True,
)

# Drop any rows where the title ended up empty after cleaning
before = len(combined)
combined = combined[combined["title"].str.len() > 0].reset_index(drop=True)
dropped = before - len(combined)
if dropped:
    print(f"Dropped {dropped} rows with empty titles after cleaning.")

# Shuffle
combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Stratified split: 70 / 15 / 15
# ---------------------------------------------------------------------------
train, temp = train_test_split(
    combined, test_size=0.30, random_state=SEED, stratify=combined["label"]
)
val, test = train_test_split(
    temp, test_size=0.50, random_state=SEED, stratify=temp["label"]
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs("data/splits", exist_ok=True)
train.to_csv("data/splits/train.csv", index=False)
val.to_csv("data/splits/val.csv",   index=False)
test.to_csv("data/splits/test.csv",  index=False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = len(combined)
print(f"\nColumns kept   : title, label")
print(f"Total articles : {total:,}")
print(f"  Fake (1)     : {combined['label'].sum():,}  ({combined['label'].mean()*100:.1f}%)")
print(f"  Real (0)     : {(combined['label']==0).sum():,}  ({(combined['label']==0).mean()*100:.1f}%)")
print()
for name, df in [("Train", train), ("Validation", val), ("Test", test)]:
    fake_n = int(df["label"].sum())
    real_n = int((df["label"] == 0).sum())
    print(f"{name:12s}: {len(df):,} rows  |  Fake: {fake_n:,} ({fake_n/len(df)*100:.1f}%)  |  Real: {real_n:,} ({real_n/len(df)*100:.1f}%)")

print("\nSample cleaned titles:")
print("  [REAL]", combined[combined["label"]==0]["title"].iloc[0])
print("  [FAKE]", combined[combined["label"]==1]["title"].iloc[0])
