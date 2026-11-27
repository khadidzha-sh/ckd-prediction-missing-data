# K
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pathlib import Path

NUMERIC_COLS = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]

def load_and_clean_data():
    # Load dataset from uci repository
    ckd = fetch_ucirepo(id=336)
    X = ckd.data.features.copy()
    y = ckd.data.targets.copy()

    df = X.copy()
    df["class"] = y.iloc[:, 0]
    # standardize missing values & remove whitespace
    df = df.replace(r"^\s*\?\s*$", np.nan, regex=True)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    
    # make sure numeric features are of numeric type
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # standardize target labels: conver them to binary
    df["class"] = df["class"].map({"ckd": 1, "notckd": 0})
    # print(df["class"].value_counts(dropna=False))

    # separate features and labels
    X = df.drop(columns=["class"])
    y = df["class"]
    
    return X, y


# This function is to transform the data into a format our algorithms (KNN, Logistic Regression, LDA) can process
def create_preprocessing_pipeline(X):
    categorical_cols = [
        col for col in X.columns
        if col not in NUMERIC_COLS + ["class"]
    ]
    
    # preprocessing for numerical features
    numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])

    # preprocessing for categorical features
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore",
                                sparse_output=False))])
    # combine both pipelines
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, categorical_cols)])
    
    return preprocessor

if __name__ == "__main__":
    X, y = load_and_clean_data()

    # save the cleaned dataset
    df = X.copy()
    df["class"] = y

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ckd_cleaned.csv"
   
    df.to_csv(output_path, index=False)

    print(f"Cleaned dataset saved to: {output_path}")