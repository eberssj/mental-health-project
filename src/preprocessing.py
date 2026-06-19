import hashlib
import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_structured_data(file_path):
    """
    Loads the new 8,300+ rows one-hot encoded dataset.
    Assumes 'Disease' is the target column and the rest are binary features.
    """
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop(['Disease'], axis=1)
    y = df['Disease']
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y, le

def process_text_data(file_path):
    """Loads and cleans the Reddit text dataset."""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['body', 'category'])
    
    le_nlp = LabelEncoder()
    y = le_nlp.fit_transform(df['category'])
    
    return df['body'], y, le_nlp

def get_tfidf_vectorizer():
    """Optimized TF-IDF Vectorizer for Reddit Text."""
    return TfidfVectorizer(
        max_features=5000, 
        ngram_range=(1, 2), 
        min_df=5,
        dtype='float32'
    )

def _texts_to_list(texts):
    if hasattr(texts, "fillna"):
        return texts.fillna("").astype(str).tolist()
    return ["" if text is None else str(text) for text in texts]

def _model_slug(model_name):
    return re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-").lower()

def get_text_fingerprint(texts):
    """Creates a stable short hash so embedding caches match the exact split."""
    digest = hashlib.sha256()
    for text in _texts_to_list(texts):
        digest.update(text.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]

def get_embedding_cache_path(
    texts,
    split_name,
    model_name=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    cache_dir="exports/embeddings"
):
    """Builds a cache path tied to the model name and current text split."""
    os.makedirs(cache_dir, exist_ok=True)
    slug = _model_slug(model_name)
    fingerprint = get_text_fingerprint(texts)
    return os.path.join(cache_dir, f"{split_name}_{slug}_{fingerprint}.npy")

def get_semantic_embeddings(
    texts,
    model_name=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    cache_path=None,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True
):
    """
    Generates semantic embeddings with a local Hugging Face SentenceTransformer.

    The model is downloaded once from Hugging Face and then loaded from the
    local cache on future runs.
    """
    texts_list = _texts_to_list(texts)

    if cache_path and os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(texts_list):
            return embeddings.astype("float32", copy=False)
        print(f"Ignoring stale embedding cache: {cache_path}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install sentence-transformers to generate semantic embeddings: "
            "pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts_list,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True
    ).astype("float32")

    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, embeddings)

    return embeddings
