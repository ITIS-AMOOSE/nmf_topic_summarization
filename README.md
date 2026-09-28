# Hệ thống Mô hình hóa Chủ đề NMF và Tóm tắt Văn bản Trích xuất Tiếng Việt

> **Bài tập lớn môn:** Phân tích và khai phá dữ liệu văn bản  
> **Đề tài:** NMF và tóm tắt văn bản dựa trên chủ đề  
> **Trọng tâm kỹ thuật:** Non-negative Matrix Factorization (NMF) + TF-IDF + Extractive Summarization

---

## 1. Cấu trúc thư mục

```text
nmf_topic_summarization/
├── data/
│   ├── sample_documents.csv     # 10 văn bản mẫu tiếng Việt (Công nghệ, Thể thao, Kinh tế, Giáo dục, Y tế)
│   └── raw/                     # Nơi đặt tập dữ liệu mở rộng (ví dụ UIT-ViNews)
├── models/
│   ├── tfidf.pkl                # Mô hình trích xuất đặc trưng TF-IDF đã huấn luyện
│   └── nmf.pkl                  # Mô hình NMF đã huấn luyện
├── results/                     # Báo cáo đánh giá và bảng kết quả tóm tắt
├── notebooks/                   # Jupyter notebooks dùng để thực nghiệm thăm dò dữ liệu
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Tiền xử lý văn bản tiếng Việt (Người 2)
│   ├── vectorizer.py            # Biểu diễn TF-IDF (Người 2)
│   ├── topic_model.py           # Mô hình NMF & trích xuất chủ đề (Người 1)
│   ├── summarizer.py            # Chấm điểm và trích chọn câu theo chủ đề (Người 3)
│   ├── evaluation.py            # Các độ đo đánh giá Reconstruction error, ROUGE, Tỷ lệ nén (Người 4)
│   └── utils.py                 # Lưu/tải mô hình và tiện ích I/O (Người 4)
├── main.py                      # Điều phối toàn bộ pipeline và giao diện CLI (Người 4)
├── requirements.txt             # Danh sách thư viện phụ thuộc
└── README.md                    # Hướng dẫn cài đặt và sử dụng
```

---

## 2. Phân công trách nhiệm 4 thành viên

| Thành viên | Phụ trách | File mã nguồn | Các hàm / chức năng chính |
| :--- | :--- | :--- | :--- |
| **Người 1** | NMF / Topic Modeling | `src/topic_model.py` | `train_nmf()`, `get_topics()`, `get_top_words()`, `get_document_topics()`, `predict_topic()` |
| **Người 2** | Dataset / Preprocessing / TF-IDF | `src/preprocessing.py`<br>`src/vectorizer.py` | `load_documents()`, `clean_text()`, `tokenize_vi()`, `remove_stopwords()`, `split_sentences()`, `fit_tfidf()`, `transform_tfidf()` |
| **Người 3** | Topic-based Summarization | `src/summarizer.py` | `score_sentences()`, `get_sentence_topics()`, `remove_redundancy()`, `summarize()` |
| **Người 4** | Evaluation / Integration | `src/evaluation.py`<br>`src/utils.py`<br>`main.py` | `reconstruction_error()`, `compression_ratio()`, `rouge_score()`, `evaluate_topics()`, `evaluate_summary()`, CLI integration |

---

## 3. Cài đặt môi trường

Khuyến nghị sử dụng Python 3.9 - 3.11.

```bash
# Di chuyển vào thư mục project
cd nmf_topic_summarization

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
venv\Scripts\activate   # Trên Windows
# source venv/bin/activate # Trên Linux/macOS

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 4. Hướng dẫn chạy thử nghiệm

### Chế độ 1: Huấn luyện và đánh giá trên tập dữ liệu (`--mode train`)
Thực hiện toàn bộ pipeline từ đọc dữ liệu, tiền xử lý, TF-IDF, NMF, trích xuất chủ đề, tạo tóm tắt, đánh giá và lưu mô hình vào `models/`:

```bash
python main.py --mode train --data data/sample_documents.csv --n_topics 5 --num_sentences 2
```

### Chế độ 2: Tóm tắt một văn bản bất kỳ (`--mode summarize`)
Sử dụng mô hình đã lưu tại `models/tfidf.pkl` và `models/nmf.pkl` để phát hiện chủ đề và tóm tắt văn bản nhập từ dòng lệnh:

```bash
python main.py --mode summarize --text "Trí tuệ nhân tạo đang phát triển với tốc độ chóng mặt trên toàn cầu. Các giải pháp tự động hóa giúp giải phóng sức lao động trong sản xuất và dịch vụ. Nhiều quốc gia đang đẩy mạnh ban hành khung pháp lý để quản lý an toàn công nghệ này."
```

---

## 5. Mở rộng với tập dữ liệu UIT-ViNews

Hệ thống được thiết kế module hóa độc lập:
1. Đặt file dữ liệu báo chí `UIT-ViNews.csv` vào thư mục `data/raw/`.
2. Kiểm tra tên cột văn bản trong file (ví dụ: `content` hoặc `text`).
3. Chạy lệnh huấn luyện với đường dẫn dữ liệu mới mà không cần chỉnh sửa code:
   ```bash
   python main.py --mode train --data data/raw/UIT-ViNews.csv --n_topics 10 --num_sentences 3
   ```
