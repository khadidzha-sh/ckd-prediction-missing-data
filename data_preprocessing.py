# K
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

ckd = fetch_ucirepo(id=336)
X = ckd.data.features.copy()
y = ckd.data.targets.copy()

df = X.copy()
df["class"] = y.iloc[:, 0]

# print(df.head())
# print(df.shape)
# print(df.info())
# # see how many values are missing
# print(df.isnull().sum())
# missing_percent = df.isnull().mean() * 100
# print(missing_percent.sort_values(ascending=False))

# standardize missing values & remove whitespace
df = df.replace(r"^\s*\?\s*$", np.nan, regex=True)
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].str.strip()

# make sure numeric features are of numeric type
numeric_cols = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# standardize target labels
df["class"] = df["class"].map({"ckd": 1, "notckd": 0})
print(df["class"].value_counts(dropna=False))

df.to_csv("ckd_cleaned.csv", index=False)