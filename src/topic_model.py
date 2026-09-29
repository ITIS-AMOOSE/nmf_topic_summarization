"""
Module mô hình hóa chủ đề sử dụng NMF (Non-negative Matrix Factorization).
Trách nhiệm: Người 1 (NMF / Topic Modeling).

Cơ sở lý thuyết (Slide 38-41):
- Phân rã ma trận không âm: V ≈ W x H
  + V (n_samples x n_features): Ma trận biểu diễn TF-IDF của tài liệu.
  + W (n_samples x k): Ma trận hệ số kích hoạt tài liệu - chủ đề (Document-Topic).
  + H (k x n_features): Ma trận thành phần chủ đề - từ khóa (Topic-Word / components_).
  + k: Số lượng chủ đề tiềm ẩn (n_components).
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from sklearn.decomposition import NMF


def train_nmf(
    tfidf_matrix: Any,
    n_components: int = 5,
    random_state: int = 42,
    init: str = "nndsvda",
    max_iter: int = 400
) -> NMF:
    """
    Khởi tạo và huấn luyện mô hình NMF trên ma trận đặc trưng TF-IDF.

    Tham số:
        tfidf_matrix: Ma trận TF-IDF không âm (scipy.sparse hoặc numpy array).
        n_components (int): Số lượng chủ đề ẩn k cần trích xuất.
        random_state (int): Hạt giống ngẫu nhiên để tái lập kết quả.
        init (str): Phương pháp khởi tạo ma trận ('nndsvda' cho kết quả ổn định trên TF-IDF).
        max_iter (int): Số vòng lặp tối đa để thuật toán hội tụ.

    Trả về:
        NMF: Đối tượng mô hình NMF đã được huấn luyện.
    """
    model = NMF(
        n_components=n_components,
        random_state=random_state,
        init=init,
        max_iter=max_iter
    )
    model.fit(tfidf_matrix)
    return model


def get_top_words(
    nmf_model: NMF,
    feature_names: List[str],
    topic_idx: int,
    top_n: int = 10
) -> List[str]:
    """
    Trích xuất top N từ khóa quan trọng nhất của một chủ đề cụ thể từ ma trận H (components_).

    Tham số:
        nmf_model (NMF): Mô hình NMF đã huấn luyện.
        feature_names (List[str]): Danh sách tên từ vựng từ TfidfVectorizer.
        topic_idx (int): Chỉ số của chủ đề (0 <= topic_idx < k).
        top_n (int): Số lượng từ khóa tiêu biểu cần lấy.

    Trả về:
        List[str]: Danh sách top N từ khóa đại diện cho chủ đề.
    """
    if topic_idx < 0 or topic_idx >= nmf_model.n_components:
        raise IndexError(f"Topic index {topic_idx} không hợp lệ với n_components={nmf_model.n_components}")

    # Lấy hàng trọng số tương ứng trong ma trận H
    topic_weights = nmf_model.components_[topic_idx]

    # Sắp xếp chỉ số theo trọng số giảm dần
    top_indices = topic_weights.argsort()[::-1][:top_n]

    # Ánh xạ chỉ số sang từ vựng
    return [feature_names[i] for i in top_indices]


def get_topics(
    nmf_model: NMF,
    feature_names: List[str],
    top_n: int = 10
) -> Dict[int, List[str]]:
    """
    Trích xuất danh sách các từ khóa đại diện cho tất cả các chủ đề đã học.

    Tham số:
        nmf_model (NMF): Mô hình NMF đã huấn luyện.
        feature_names (List[str]): Danh sách tên từ vựng từ TfidfVectorizer.
        top_n (int): Số lượng từ đại diện cho mỗi chủ đề.

    Trả về:
        Dict[int, List[str]]: Dictionary {topic_id: [top_words]}.
    """
    topics = {}
    for i in range(nmf_model.n_components):
        topics[i] = get_top_words(nmf_model, feature_names, topic_idx=i, top_n=top_n)
    return topics


def get_document_topics(nmf_model: NMF, tfidf_matrix: Any) -> np.ndarray:
    """
    Chiếu ma trận TF-IDF của các tài liệu sang không gian chủ đề (tìm ma trận W).

    Tham số:
        nmf_model (NMF): Mô hình NMF đã huấn luyện.
        tfidf_matrix: Ma trận TF-IDF của các tài liệu.

    Trả về:
        np.ndarray: Ma trận Document-Topic kích thước (n_docs x k).
    """
    return nmf_model.transform(tfidf_matrix)


def predict_topic(nmf_model: NMF, tfidf_vector: Any) -> Tuple[int, np.ndarray]:
    """
    Dự đoán chủ đề chiếm ưu thế (dominant topic) cho một tài liệu hoặc câu văn mới.

    Tham số:
        nmf_model (NMF): Mô hình NMF đã huấn luyện.
        tfidf_vector: Vector TF-IDF của văn bản/câu mới (kích thước 1 x n_features).

    Trả về:
        Tuple[int, np.ndarray]:
            - int: Chỉ số của chủ đề nổi bật nhất (argmax).
            - np.ndarray: Mảng phân phối trọng số trên toàn bộ k chủ đề.
    """
    topic_distribution = nmf_model.transform(tfidf_vector)[0]
    dominant_topic = int(np.argmax(topic_distribution))
    return dominant_topic, topic_distribution


# Standalone Test nếu chạy trực tiếp file này
if __name__ == "__main__":
    print("--- KIỂM THỬ ĐỘC LẬP MODULE TOPIC_MODEL (NGƯỜI 1) ---")
    
    # Giả lập ma trận TF-IDF 4 văn bản, 6 từ vựng
    dummy_tfidf = np.array([
        [0.8, 0.6, 0.0, 0.0, 0.0, 0.0],  # Doc 1: Công nghệ (AI, robot)
        [0.7, 0.7, 0.1, 0.0, 0.0, 0.0],  # Doc 2: Công nghệ
        [0.0, 0.0, 0.0, 0.9, 0.8, 0.1],  # Doc 3: Thể thao (bóng đá, cầu thủ)
        [0.0, 0.0, 0.0, 0.7, 0.9, 0.2],  # Doc 4: Thể thao
    ])
    vocab = ["trí_tuệ_nhân_tạo", "học_máy", "phần_mềm", "bóng_đá", "cầu_thủ", "sân_cỏ"]

    # 1. Train NMF với k=2 chủ đề
    model = train_nmf(dummy_tfidf, n_components=2, random_state=42)
    print("1. Huấn luyện NMF thành công.")

    # 2. Lấy từ khóa từng topic
    topics = get_topics(model, vocab, top_n=2)
    print("2. Các chủ đề trích xuất được:")
    for tid, words in topics.items():
        print(f"   - Chủ đề {tid}: {words}")

    # 3. Lấy phân phối chủ đề các tài liệu (Ma trận W)
    W = get_document_topics(model, dummy_tfidf)
    print(f"3. Kích thước ma trận W: {W.shape}")

    # 4. Dự đoán văn bản mới (chứa từ 'bóng_đá')
    new_doc_vec = np.array([[0.0, 0.0, 0.0, 1.0, 0.5, 0.0]])
    dominant, dist = predict_topic(model, new_doc_vec)
    print(f"4. Dự đoán văn bản mới thuộc chủ đề: {dominant} (Phân phối: {dist.round(3)})")
    print(">>> TẤT CẢ CÁC HÀM CỦA NGƯỜI 1 HOẠT ĐỘNG CHÍNH XÁC! <<<")
