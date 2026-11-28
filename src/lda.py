# Khadidzha - Linear Discriminant Analysis
# use data_preprocessing.py to load and clean the data, and create a preprocessing pipeline

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_validate
from data_preprocessing import load_and_clean_data, create_preprocessing_pipeline

# LDA pipeline
def create_lda_pipeline(X):
    lda_pipeline = Pipeline([("preprocessor", create_preprocessing_pipeline(X)), ("classifier", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])
    return lda_pipeline

# evaluate it with 5-fold
def evaluate_lda(X, y):
    lda_pipeline = create_lda_pipeline(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring=["accuracy", "precision", "recall", "f1"]
    scores = cross_validate(lda_pipeline, X, y, cv=cv, scoring=scoring)

    print("LDA baseline performance:")
    for metric in scoring:
        values = scores[f"test_{metric}"]
        print(f"{metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

if __name__ == "__main__":
    X, y = load_and_clean_data()

    evaluate_lda(X, y)