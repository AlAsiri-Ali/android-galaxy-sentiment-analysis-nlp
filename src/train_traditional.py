from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from preprocessing import prepare_binary_dataset


def build_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "svc": SVC(),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/android_galaxy_comments.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/traditional_results.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = prepare_binary_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label"],
        test_size=0.3,
        random_state=42,
        stratify=df["label"],
    )

    vectorizers = {
        "bow": CountVectorizer(),
        "tfidf": TfidfVectorizer(),
    }

    rows = []
    for vec_name, vectorizer in vectorizers.items():
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        for model_name, model in build_models().items():
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            rows.append({
                "model": model_name,
                "representation": vec_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, pos_label="positive"),
                "recall": recall_score(y_test, y_pred, pos_label="positive"),
                "f1_score": f1_score(y_test, y_pred, pos_label="positive"),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
