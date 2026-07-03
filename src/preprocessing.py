from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def clean_text(text: str, stop_words: set[str] | None = None, lemmatizer: WordNetLemmatizer | None = None) -> str:
    """Clean and normalize a single text review."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    stop_words = stop_words or set(stopwords.words("english"))
    lemmatizer = lemmatizer or WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words and len(tok) > 1]
    return " ".join(tokens)


def prepare_binary_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove neutral labels and keep positive/negative comments for binary sentiment classification."""
    df = df.copy()
    df = df[df["label"].isin(["positive", "negative"])].reset_index(drop=True)
    df["cleaned_text"] = df["comment_text"].apply(clean_text)
    return df
