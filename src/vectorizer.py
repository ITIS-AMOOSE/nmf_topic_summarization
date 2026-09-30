import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from src.preprocessing import space_tokenizer, identity_preprocessor, preprocess
except (ImportError, ModuleNotFoundError):
    from preprocessing import space_tokenizer, identity_preprocessor, preprocess


def fit_tfidf(docs, min_df=1, max_df=1.0, ngram_range=(1, 2)):
    """
    Khởi tạo và huấn luyện TfidfVectorizer trên danh sách văn bản đã tiền xử lý.

    Tham số:
        docs: Danh sách các văn bản (chuỗi đã qua tiền xử lý tách từ).
        min_df: Tần số tài liệu tối thiểu (mặc định: 1).
        max_df: Tần số tài liệu tối đa (mặc định: 1.0).
        ngram_range: Khoảng n-gram (mặc định: (1, 2)).

    Trả về:
        vectorizer (TfidfVectorizer): Mô hình vectorizer đã huấn luyện.
        X (scipy.sparse.csr_matrix): Ma trận TF-IDF đặc trưng.
    """
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


def transform_tfidf(vectorizer, docs_preprocessed):
    """Biến đổi danh sách văn bản đã tiền xử lý sang ma trận TF-IDF."""
    return vectorizer.transform(docs_preprocessed)


# Alias tương thích
transform_new = transform_tfidf


def preprocess_and_transform(vectorizer, raw_texts, **preprocess_kwargs):
    """Tiền xử lý và transform danh sách văn bản thô."""
    docs_pre = [preprocess(t, **preprocess_kwargs) for t in raw_texts]
    return vectorizer.transform(docs_pre), docs_pre


def save_vectorizer(vectorizer, path="models/tfidf.pkl"):
    """Lưu TfidfVectorizer vào file."""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path="models/tfidf.pkl"):
    """Tải TfidfVectorizer từ file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy vectorizer tại: {path}")
    return joblib.load(path)