"""
main.py - Điều phối pipeline + CLI (Người 4).

Pipeline: Văn bản -> Tiền xử lý -> TF-IDF -> NMF -> Topic -> Tóm tắt -> Đánh giá

!!! ĐANG DÙNG DỮ LIỆU MOCK !!!
Người 1-3 chưa có code, nên mọi chỗ gọi hàm của họ được thay bằng dữ liệu đầu ra
dựng sẵn ở khối "MOCK DATA" bên dưới. Mỗi bước trong run_train() có comment
`# THẬT:` ghi lời gọi dự kiến. Khi có code thật, chỉ cần thay dòng mock bằng dòng
đó (chữ ký hàm ở comment là dự kiến, hãy đối chiếu lại với code thật).

Số liệu mock CHỈ để chạy thử phần của Người 4. Không đưa vào báo cáo.
"""

import argparse
import os
import re
import sys

import numpy as np

from src.evaluation import (
    build_doc_topic_table,
    build_k_table,
    build_summary_table,
    build_topic_table,
    evaluate_summary,
    evaluate_topics,
    topic_category_crosstab,
)
from src.utils import (
    load_model,
    plot_doc_topic_heatmap,
    plot_k_vs_error,
    save_model,
    save_table,
    save_text,
)

# ===========================================================================
# MOCK DATA - thay bằng đầu ra thật của Người 1, 2, 3
# ===========================================================================

# ---- Người 2: load_documents() + split_sentences() ------------------------
# 10 văn bản, mỗi văn bản đã tách sẵn thành 3 câu (khớp data/sample_documents.csv)
MOCK_CATEGORIES = ["Cong_nghe", "Cong_nghe", "The_thao", "The_thao", "Kinh_te",
                   "Kinh_te", "Giao_duc", "Giao_duc", "Y_te", "Y_te"]

MOCK_SENTENCES = [
    ["Trí tuệ nhân tạo đang làm thay đổi căn bản cách thức vận hành của nhiều ngành công nghiệp hiện đại.",
     "Các mô hình học sâu giúp tự động hóa quá trình xử lý dữ liệu lớn và nâng cao hiệu suất làm việc của doanh nghiệp.",
     "Trong tương lai, AI sẽ tiếp tục đóng vai trò then chốt trong cuộc cách mạng số toàn cầu."],
    ["Thế hệ điện thoại thông minh mới được tích hợp vi xử lý bán dẫn tiến trình 3 nanomet tiên tiến.",
     "Thiết bị này mang lại hiệu năng xử lý đồ họa vượt trội đồng thời tiết kiệm pin đáng kể cho người dùng.",
     "Khách hàng đánh giá rất cao khả năng kết nối mạng vệ tinh khẩn cấp của dòng máy cao cấp này."],
    ["Đội tuyển bóng đá nam quốc gia vừa giành chiến thắng thuyết phục trong trận đấu giao hữu quốc tế.",
     "Các cầu thủ trẻ đã thể hiện lối chơi kỷ luật, gắn kết và tuân thủ chặt chẽ đấu pháp chiến thuật của huấn luyện viên.",
     "Người hâm mộ kỳ vọng đội tuyển sẽ tiếp tục duy trì phong độ ấn tượng này ở các giải đấu khu vực sắp tới."],
    ["Giải chạy marathon quốc tế thu hút hàng nghìn vận động viên chuyên nghiệp và phong trào tham gia tranh tài.",
     "Cung đường chạy đi qua nhiều địa danh lịch sử nổi tiếng tạo nên trải nghiệm văn hóa thể thao vô cùng độc đáo.",
     "Sự kiện góp phần lan tỏa mạnh mẽ tinh thần rèn luyện sức khỏe trong cộng đồng."],
    ["Tăng trưởng kinh tế quý này ghi nhận mức tăng ấn tượng nhờ vào đà phục hồi mạnh mẽ của khu vực sản xuất và xuất khẩu.",
     "Kim ngạch thương mại hai chiều với các đối tác chiến lược tiếp tục thiết lập những kỷ lục mới.",
     "Các chuyên gia kinh tế dự báo chỉ số giá tiêu dùng sẽ duy trì ổn định đến cuối năm."],
    ["Ngân hàng trung ương quyết định điều chỉnh giảm nhẹ lãi suất điều hành nhằm hỗ trợ doanh nghiệp tiếp cận nguồn vốn giá rẻ.",
     "Thị trường bất động sản và chứng khoán phản ứng tích cực với những gói tín dụng ưu đãi mới được công bố.",
     "Dòng tiền đầu tư đang dần quay trở lại các kênh sản xuất kinh doanh thực chất."],
    ["Các trường đại học đang đẩy mạnh chuyển đổi số trong công tác giảng dạy và quản lý sinh viên.",
     "Hệ thống bài giảng điện tử và thư viện số trực tuyến giúp người học chủ động tiếp cận nguồn tri thức phong phú mọi lúc mọi nơi.",
     "Phương pháp học tập kết hợp này nâng cao rõ rệt sự tương tác và khả năng tự nghiên cứu."],
    ["Bộ Giáo dục và Đào tạo công bố phương án đổi mới cấu trúc đề thi tốt nghiệp trung học phổ thông từ năm học mới.",
     "Đề thi mẫu tập trung đánh giá năng lực tư duy logic và kỹ năng giải quyết vấn đề thay vì ghi nhớ máy móc.",
     "Sự điều chỉnh này nhận được nhiều phản hồi tích cực từ phía giáo viên và học sinh trên cả nước."],
    ["Bộ Y tế khuyến cáo người dân chủ động tiêm phòng vaccine đầy đủ để ngăn ngừa các đợt bùng phát dịch bệnh truyền nhiễm.",
     "Các bệnh viện tuyến đầu đã chuẩn bị sẵn sàng thuốc men, vật tư và trang thiết bị hồi sức cấp cứu hiện đại.",
     "Ý thức phòng ngừa và kiểm tra sức khỏe định kỳ đóng vai trò quyết định trong việc bảo vệ cộng đồng."],
    ["Chế độ dinh dưỡng cân bằng và lành mạnh giúp tăng cường hệ miễn dịch tự nhiên của cơ thể.",
     "Bác sĩ khuyên mọi người nên bổ sung nhiều rau xanh, trái cây tươi kết hợp với việc uống đủ nước và ngủ đủ giấc.",
     "Lối sống khoa học này giúp giảm thiểu nguy cơ mắc các bệnh tim mạch và tiểu đường mạn tính."],
]

MOCK_DOC_IDS = list(range(1, 11))
MOCK_DOCUMENTS = [
    {"doc_id": MOCK_DOC_IDS[i], "category": MOCK_CATEGORIES[i], "text": " ".join(MOCK_SENTENCES[i])}
    for i in range(10)
]

# ---- Người 2: TF-IDF -> feature_names + X ---------------------------------
# ---- Người 1: NMF -> topics (top words), H, W ------------------------------
# Mỗi topic: (từ, trọng số trong H). 5 topic x 6 từ = 30 từ.
MOCK_TOPIC_WORDS = {
    0: [("trí_tuệ_nhân_tạo", 0.92), ("học_sâu", 0.85), ("dữ_liệu", 0.78),
        ("điện_thoại", 0.74), ("vi_xử_lý", 0.70), ("hiệu_năng", 0.66)],
    1: [("bóng_đá", 0.93), ("đội_tuyển", 0.88), ("cầu_thủ", 0.80),
        ("marathon", 0.76), ("vận_động_viên", 0.71), ("huấn_luyện_viên", 0.64)],
    2: [("kinh_tế", 0.91), ("lãi_suất", 0.86), ("ngân_hàng", 0.79),
        ("xuất_khẩu", 0.73), ("doanh_nghiệp", 0.69), ("tăng_trưởng", 0.65)],
    3: [("đại_học", 0.90), ("sinh_viên", 0.84), ("giáo_dục", 0.81),
        ("đề_thi", 0.75), ("học_sinh", 0.70), ("bài_giảng", 0.63)],
    4: [("vaccine", 0.94), ("bệnh_viện", 0.87), ("dinh_dưỡng", 0.80),
        ("sức_khỏe", 0.76), ("bác_sĩ", 0.72), ("dịch_bệnh", 0.66)],
}
MOCK_TOP_WORDS = {k: [w for w, _ in v] for k, v in MOCK_TOPIC_WORDS.items()}   # kiểu get_top_words()
MOCK_FEATURE_NAMES = [w for v in MOCK_TOP_WORDS.values() for w in v]

MOCK_H = np.zeros((5, 30))                    # topic x word
for _k, _items in MOCK_TOPIC_WORDS.items():
    for _j, (_, _weight) in enumerate(_items):
        MOCK_H[_k, _k * 6 + _j] = _weight

MOCK_W = np.array([                           # document x topic (mỗi hàng 1 văn bản)
    [0.85, 0.00, 0.05, 0.03, 0.00],
    [0.80, 0.02, 0.00, 0.00, 0.04],
    [0.02, 0.83, 0.00, 0.00, 0.05],
    [0.00, 0.78, 0.03, 0.00, 0.10],
    [0.03, 0.00, 0.84, 0.00, 0.00],
    [0.00, 0.00, 0.80, 0.06, 0.00],
    [0.10, 0.00, 0.00, 0.79, 0.00],
    [0.00, 0.00, 0.02, 0.86, 0.00],
    [0.00, 0.03, 0.00, 0.00, 0.88],
    [0.00, 0.00, 0.00, 0.02, 0.82],
])

# X giả ≈ W@H + nhiễu cố định (seed) để reconstruction error có ý nghĩa
_rng = np.random.RandomState(42)
MOCK_X = MOCK_W @ MOCK_H + np.abs(_rng.normal(0, 0.03, (10, 30)))

MOCK_MODEL_ERROR = 0.5106         # giả lập model.reconstruction_err_ (khớp X, W, H mock)
MOCK_K_ERRORS = {2: 0.9312, 3: 0.7645, 4: 0.6187, 5: 0.5106, 6: 0.4473}   # giả lập kết quả thử nhiều k

# ---- Người 3: summarize() ---------------------------------------------------
MOCK_DOC_TOPIC_IDS = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
MOCK_SENTENCE_SCORES = [
    [0.71, 0.64, 0.38], [0.66, 0.41, 0.58], [0.62, 0.49, 0.55], [0.57, 0.60, 0.33], [0.69, 0.35, 0.52],
    [0.63, 0.58, 0.29], [0.54, 0.67, 0.44], [0.72, 0.51, 0.47], [0.61, 0.36, 0.59], [0.48, 0.66, 0.57],
]
MOCK_SELECTED_INDICES = [  # top-2 theo điểm, đã sắp lại theo thứ tự câu gốc
    [0, 1], [0, 2], [0, 2], [0, 1], [0, 2], [0, 1], [0, 1], [0, 1], [0, 2], [1, 2],
]
# Reference summary giả (thực tế nhóm phải tự tạo và ghi rõ cách tạo trong báo cáo)
MOCK_REFERENCE_INDICES = [
    [0, 2], [0, 1], [0, 2], [0, 1], [0, 2], [0, 1], [0, 2], [0, 1], [0, 1], [1, 2],
]


def _mock_summarize_results():
    """Giả lập danh sách kết quả summarize() cho từng văn bản trong tập train."""
    results = []
    for i, sents in enumerate(MOCK_SENTENCES):
        topic_id = MOCK_DOC_TOPIC_IDS[i]
        idx = MOCK_SELECTED_INDICES[i]
        results.append({
            "original": " ".join(sents),
            "summary": " ".join(sents[j] for j in idx),
            "topic_id": topic_id,
            "keywords": MOCK_TOP_WORDS[topic_id][:5],
            "sentence_scores": MOCK_SENTENCE_SCORES[i],
            "selected_indices": idx,
        })
    return results


def _mock_summarize_new_text(text, num_sentences):
    """Giả lập summarize(text, ...) cho văn bản mới: luôn báo topic 0, lấy N câu đầu."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    idx = list(range(min(num_sentences, len(sents))))
    return {
        "original": text,
        "summary": " ".join(sents[j] for j in idx),
        "topic_id": 0,
        "keywords": MOCK_TOP_WORDS[0][:5],
        "sentence_scores": [round(0.70 - 0.08 * i, 2) for i in range(len(sents))],
        "selected_indices": idx,
    }


# ===========================================================================
# PIPELINE
# ===========================================================================
def _banner(note):
    print("=" * 70)
    print("[MOCK] Đang dùng dữ liệu giả thay cho module của Người 1-3.")
    print(f"       {note}")
    print("=" * 70)


def _format_example(res, reference, metrics):
    lines = [
        "VÍ DỤ ĐẦY ĐỦ: Input -> Topic -> Summary",
        f"Input      : {res['original']}",
        f"Topic      : {res['topic_id']}  (từ khóa: {', '.join(res['keywords'])})",
        f"Điểm câu   : {res['sentence_scores']}",
        f"Câu được chọn (chỉ số): {res['selected_indices']}",
        f"Summary    : {res['summary']}",
        f"Độ dài gốc / tóm tắt : {metrics['original_words']} / {metrics['summary_words']} từ",
        f"Compression ratio    : {metrics['compression_ratio']:.4f}",
    ]
    if reference and "rouge" in metrics:
        r = metrics["rouge"]
        lines.append(
            f"ROUGE-1/2/L (F1)     : {r['rouge1']['f1']:.4f} / {r['rouge2']['f1']:.4f} / {r['rougeL']['f1']:.4f}"
        )
    return "\n".join(lines)


def run_train(args):
    _banner("Tham số --data / --n_topics / --num_sentences chưa có tác dụng với phần mock.")
    os.makedirs(args.results_dir, exist_ok=True)

    # --- Người 2: đọc dữ liệu ---------------------------------------------
    # THẬT: documents = load_documents(args.data)
    documents = MOCK_DOCUMENTS
    doc_ids = [d["doc_id"] for d in documents]
    categories = [d["category"] for d in documents]  # chỉ để đối chiếu, KHÔNG đưa vào NMF

    # --- Người 2: tiền xử lý + TF-IDF -------------------------------------
    # THẬT: X, tfidf, feature_names = fit_tfidf([preprocess(d["text"]) for d in documents])
    X, feature_names = MOCK_X, MOCK_FEATURE_NAMES
    tfidf = {"mock": True, "feature_names": feature_names}

    # --- Người 1: NMF -----------------------------------------------------
    # THẬT: nmf, W, H = train_nmf(X, n_topics=args.n_topics)
    #       top_words = get_top_words(H, feature_names, n=6)
    #       model_error = nmf.reconstruction_err_
    W, H, top_words = MOCK_W, MOCK_H, MOCK_TOP_WORDS
    model_error = MOCK_MODEL_ERROR
    nmf = {"mock": True, "H": H, "feature_names": feature_names}
    # THẬT: k_errors = {k: train_nmf(X, k)[0].reconstruction_err_ for k in range(2, 7)}
    k_errors = MOCK_K_ERRORS

    # --- Người 3: tóm tắt từng văn bản ------------------------------------
    # THẬT: results = [summarize(d["text"], tfidf, nmf, args.num_sentences) for d in documents]
    results = _mock_summarize_results()
    references = [" ".join(MOCK_SENTENCES[i][j] for j in MOCK_REFERENCE_INDICES[i]) for i in range(10)]

    # --- Người 4: đánh giá ------------------------------------------------
    topic_metrics = evaluate_topics(X, W, H, model_error=model_error)
    summary_table = build_summary_table(results, references, doc_ids, categories)
    topic_table = build_topic_table(top_words)
    doc_topic_table = build_doc_topic_table(W, doc_ids, categories)
    k_table = build_k_table(k_errors)
    crosstab = topic_category_crosstab(doc_topic_table["dominant_topic"], categories)

    # --- Lưu kết quả ------------------------------------------------------
    r = args.results_dir
    save_table(k_table, os.path.join(r, "k_reconstruction_error.csv"))
    save_table(topic_table, os.path.join(r, "topic_keywords.csv"))
    save_table(doc_topic_table, os.path.join(r, "document_topics.csv"))
    save_table(summary_table, os.path.join(r, "summary_metrics.csv"))
    save_table(crosstab.reset_index(), os.path.join(r, "topic_vs_category.csv"))
    plot_k_vs_error(k_table, os.path.join(r, "k_vs_error.png"))
    plot_doc_topic_heatmap(W, doc_ids, os.path.join(r, "doc_topic_heatmap.png"))

    example = _format_example(results[0], references[0], evaluate_summary(
        results[0]["original"], results[0]["summary"], references[0]))
    save_text(example, os.path.join(r, "example.txt"))

    save_model(tfidf, os.path.join(args.models_dir, "tfidf.pkl"))
    save_model(nmf, os.path.join(args.models_dir, "nmf.pkl"))

    # --- In ra màn hình ---------------------------------------------------
    print("\n[1] Chỉ số mô hình chủ đề")
    for key, val in topic_metrics.items():
        print(f"    {key}: {val}")
    print("\n[2] Bảng k - reconstruction error\n", k_table.to_string(index=False))
    print("\n[3] Top keywords theo topic\n", topic_table.to_string(index=False))
    print("\n[4] Topic chiếm ưu thế x nhãn tham chiếu\n", crosstab.to_string())
    print("\n[5] Chỉ số tóm tắt\n", summary_table.to_string(index=False))
    print("\n[6]", example)
    print(f"\nĐã lưu bảng/biểu đồ vào '{r}/' và mô hình vào '{args.models_dir}/'.")


def run_summarize(args):
    _banner("Topic/từ khóa/điểm câu là giá trị giả cố định; chỉ số câu chọn = N câu đầu.")
    if not args.text:
        sys.exit("Thiếu --text khi dùng --mode summarize.")

    # THẬT: tfidf, nmf = load_model(.../tfidf.pkl), load_model(.../nmf.pkl)
    #       result = summarize(args.text, tfidf, nmf, args.num_sentences)
    load_model(os.path.join(args.models_dir, "tfidf.pkl"))   # kiểm tra đã train chưa
    load_model(os.path.join(args.models_dir, "nmf.pkl"))
    result = _mock_summarize_new_text(args.text, args.num_sentences)

    metrics = evaluate_summary(result["original"], result["summary"])  # văn bản mới: không có reference
    print(f"Topic phát hiện : {result['topic_id']}  (từ khóa: {', '.join(result['keywords'])})")
    print(f"Điểm từng câu   : {result['sentence_scores']}")
    print(f"Câu được chọn   : {result['selected_indices']}")
    print(f"Tóm tắt         : {result['summary']}")
    print(f"Độ dài gốc/tóm tắt: {metrics['original_words']}/{metrics['summary_words']} từ, "
          f"compression ratio = {metrics['compression_ratio']:.4f}")


def main():
    # Windows console mặc định không phải UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description="NMF topic modeling + tóm tắt trích xuất tiếng Việt")
    p.add_argument("--mode", required=True, choices=["train", "summarize"])
    p.add_argument("--data", default="data/sample_documents.csv")
    p.add_argument("--n_topics", type=int, default=5)
    p.add_argument("--num_sentences", type=int, default=2)
    p.add_argument("--text", default=None)
    p.add_argument("--models_dir", default="models")
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    if args.mode == "train":
        run_train(args)
    else:
        run_summarize(args)


if __name__ == "__main__":
    main()
