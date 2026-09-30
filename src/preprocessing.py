import re
import string
try:
    from underthesea import word_tokenize, sent_tokenize
    HAS_UNDERTHESEA = True
except Exception:
    HAS_UNDERTHESEA = False

VI_STOPWORDS = {
    "và","của","là","với","cho","các","một","những","trong","có","được",
    "đã","như","đối","vì","do","đang","tại","theo","khi","này","ra","từ",
    "đến","anh","chị","em","ông","bà","họ","ta","tôi","mình"
}

URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
EXTRA_WS = re.compile(r'\s+')

def clean_text(text, remove_digits=True, remove_punct=True):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = URL_REGEX.sub(' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)  
    text = text.replace('\xa0', ' ')
    text = EXTRA_WS.sub(' ', text).strip()
    if remove_digits:
        text = re.sub(r'\d+', ' ', text)
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = EXTRA_WS.sub(' ', text).strip()
    return text

def tokenize_vi(text):
    text = text.strip()
    if HAS_UNDERTHESEA:
        toks = word_tokenize(text, format="text")  
        return toks.split()
    text = re.sub(r'([,.;:?!()\"])', r' \1 ', text)
    return [t for t in text.split() if t]

def remove_stopwords(tokens, stopwords=None):
    if stopwords is None:
        stopwords = VI_STOPWORDS
    return [t for t in tokens if t not in stopwords and len(t) > 0]

def preprocess(text, remove_digits=True, remove_punct=True, stopwords=None) -> str:
    c = clean_text(text, remove_digits=remove_digits, remove_punct=remove_punct)
    toks = tokenize_vi(c)
    toks = remove_stopwords(toks, stopwords=stopwords)
    return " ".join(toks)

def split_sentences(text):
    if HAS_UNDERTHESEA:
        return sent_tokenize(text)
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    return [seg.strip() for seg in s if seg.strip()]

def identity_preprocessor(text):
    """Hàm tiền xử lý đồng nhất giữ nguyên chuỗi đã xử lý."""
    return text

def space_tokenizer(text):
    """Tách token theo khoảng trắng cho chuỗi đã tiền xử lý/tách từ."""
    if isinstance(text, str):
        return text.split()
    return list(text)

def load_documents(filepath: str) -> list:
    """Đọc dữ liệu văn bản từ file CSV (hoặc text). Trả về danh sách dict."""
    import pandas as pd
    df = pd.read_csv(filepath)
    if "doc_id" not in df.columns:
        df["doc_id"] = list(range(1, len(df) + 1))
    return df.to_dict(orient="records")