import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

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