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

def clean_text(text, remove_digits, remove_punct):
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
    text = re.sub(r'([,.;:?!()"])', r' \1 ', text)
    return [t for t in text.split() if t]

def remove_stopwords(tokens, stopwords=None):
    if stopwords is None:
        stopwords = VI_STOPWORDS
    return [t for t in tokens if t not in stopwords and len(t) > 0]

def preprocess(text, remove_digits, remove_punct, stopwords=None) -> str:
    c = clean_text(text, remove_digits=remove_digits, remove_punct=remove_punct)
    toks = tokenize_vi(c)
    toks = remove_stopwords(toks, stopwords=stopwords)
    return " ".join(toks)

def split_sentences(text):
    if HAS_UNDERTHESEA:
        return sent_tokenize(text)
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    return [seg.strip() for seg in s if seg.strip()]