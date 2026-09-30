import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import space_tokenizer, identity_preprocessor, preprocess

def fit_tfidf(docs, min_df, max_df, ngram_range):
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        tokenizer=space_tokenizer,
        preprocessor=identity_preprocessor,
        token_pattern=None,
        lowercase=False
    )
    X = vectorizer.fit_transform(docs)
    return vectorizer, X

def transform_new(vectorizer, docs_preprocessed):
    return vectorizer.transform(docs_preprocessed)

def preprocess_and_transform(vectorizer, raw_texts, **preprocess_kwargs):
    docs_pre = [preprocess(t, **preprocess_kwargs) for t in raw_texts]
    return vectorizer.transform(docs_pre), docs_pre

def save_vectorizer(vectorizer, path="tfidf.pkl"):
    joblib.dump(vectorizer, path)

def load_vectorizer(path="tfidf.pkl"):
    return joblib.load(path)