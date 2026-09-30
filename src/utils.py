"""
utils.py - Lưu/tải mô hình, ghi bảng kết quả, vẽ biểu đồ (Người 4).
"""

import os
import pickle

import numpy as np


def ensure_dir(path):
    """Tạo thư mục (và thư mục cha) nếu chưa có. Trả về chính path."""
    if path:
        os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Mô hình
# ---------------------------------------------------------------------------
def save_model(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy mô hình: {path}. Hãy chạy `python main.py --mode train ...` trước."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Bảng / văn bản
# ---------------------------------------------------------------------------
def save_table(df, path):
    """Lưu DataFrame ra CSV. utf-8-sig để Excel mở không lỗi font tiếng Việt."""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_text(text, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Biểu đồ (matplotlib import muộn để phần còn lại chạy được khi chưa cài)
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")  # ghi ra file, không cần cửa sổ
    import matplotlib.pyplot as plt
    return plt


def plot_k_vs_error(k_table, path):
    """Biểu đồ k - reconstruction error."""
    plt = _plt()
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_table["k"], k_table["reconstruction_error"], marker="o")
    for k, e in zip(k_table["k"], k_table["reconstruction_error"]):
        ax.annotate(f"{e:.3f}", (k, e), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xlabel("Số chủ đề k")
    ax.set_ylabel("Reconstruction error ||X - WH||_F")
    ax.set_title("Sai số tái tạo theo số chủ đề")
    ax.set_xticks(list(k_table["k"]))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_doc_topic_heatmap(W, doc_ids, path):
    """Heatmap ma trận document-topic W."""
    plt = _plt()
    ensure_dir(os.path.dirname(path))
    W = np.asarray(W)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(W, aspect="auto", cmap="Blues")
    ax.set_xticks(range(W.shape[1]))
    ax.set_xticklabels([f"Topic {k}" for k in range(W.shape[1])])
    ax.set_yticks(range(W.shape[0]))
    ax.set_yticklabels([f"Doc {d}" for d in doc_ids])
    ax.set_title("Phân bố chủ đề của từng văn bản (W)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
