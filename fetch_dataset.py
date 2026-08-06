"""
Fetches the UCI Student Performance dataset and saves it locally.
Source: Cortez, P. (2008). Student Performance [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd

def fetch_and_save():
    dataset = fetch_ucirepo(id=320)
    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)
    df.to_csv("data/student_performance_raw.csv", index=False)

    print("Dataset shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSaved to data/student_performance_raw.csv")

if __name__ == "__main__":
    fetch_and_save()
