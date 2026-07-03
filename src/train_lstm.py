from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from preprocessing import prepare_binary_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/android_galaxy_comments.csv"))
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    df = prepare_binary_dataset(pd.read_csv(args.data))
    labels = df["label"].map({"negative": 0, "positive": 1}).values

    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["cleaned_text"])
    sequences = tokenizer.texts_to_sequences(df["cleaned_text"])
    X = pad_sequences(sequences, maxlen=100, padding="post", truncating="post")

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels
    )

    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=100),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_split=0.2, epochs=args.epochs, batch_size=64)
    print(model.evaluate(X_test, y_test, verbose=0))


if __name__ == "__main__":
    main()
