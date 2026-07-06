import time
from pathlib import Path

import streamlit as st
from PIL import Image

from recommender import FashionHybridRecommender

BASE_DIR = Path(__file__).resolve().parent
FAISS_ARTIFACT_DIR = BASE_DIR / "results" / "faiss-index"
CF_ARTIFACT_DIR = BASE_DIR / "results" / "cf-hybrid-fusion"
IMAGE_ROOT = BASE_DIR / "results" / "dataset-preprocessing" / "images"

CLIP_CHECKPOINT_PATH = BASE_DIR / "results" / "fashion-dataset" / "checkpoints" / "fashion_clip_best.pt"
CF_MODEL_PATH = CF_ARTIFACT_DIR / "cf_model.pkl"
INTERACTIONS_PATH = FAISS_ARTIFACT_DIR / "interactions.csv"

N_COLS = 5  # so anh moi hang trong luoi ket qua
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]


@st.cache_resource(show_spinner="Dang tai model (CLIP + FAISS + CF)...")
def load_recommender() -> FashionHybridRecommender:
    return FashionHybridRecommender(
        artifact_dir=FAISS_ARTIFACT_DIR,
        cf_model_path=CF_MODEL_PATH,
        clip_checkpoint_path=CLIP_CHECKPOINT_PATH,
        interactions_path=INTERACTIONS_PATH,
    )


def render_sidebar(rec: FashionHybridRecommender):
    """Ve toan bo control o sidebar va tra ve dict tham so tim kiem."""
    st.sidebar.header("Tim kiem")

    mode = st.sidebar.radio(
        "Che do tim kiem",
        options=["Hinh anh", "Van ban", "Ket hop"],
        index=1,
    )

    user_ids = sorted(rec.user_mapping.keys())
    user_choice = st.sidebar.selectbox(
        "User ID (ca nhan hoa)",
        options=["Khach (Cold-start)"] + user_ids,
    )
    user_id = None if user_choice == "Khach (Cold-start)" else int(user_choice)

    top_k = st.sidebar.slider("So ket qua (top_k)", min_value=5, max_value=20, value=10, step=1)

    alpha_h = 0.6
    if user_id is not None:
        alpha_h = st.sidebar.slider(
            "Trong so CLIP vs CF (alpha_h)",
            min_value=0.0, max_value=1.0, value=0.6, step=0.05,
            help="1.0 = chi dung do tuong dong CLIP, 0.0 = chi dung goi y theo lich su (CF).",
        )
    else:
        st.sidebar.caption("Khach -> tu dong dung pure CLIP (cold-start fallback).")

    uploaded_image = None
    query_text = ""
    alpha_img_txt = None  # None -> dung alpha mac dinh luu trong index_config.json

    if mode == "Hinh anh":
        uploaded_image = st.sidebar.file_uploader("Tai anh san pham", type=SUPPORTED_IMAGE_TYPES)
    elif mode == "Van ban":
        query_text = st.sidebar.text_input(
            "Mo ta san pham (tieng Anh)",
            placeholder="vd: black floral print dress",
        )
    else:  # Ket hop
        uploaded_image = st.sidebar.file_uploader("Tai anh san pham", type=SUPPORTED_IMAGE_TYPES)
        query_text = st.sidebar.text_input(
            "Mo ta bo sung (tieng Anh)",
            placeholder="vd: black floral print dress",
        )
        alpha_img_txt = st.sidebar.slider(
            "Trong so Anh vs Van ban (alpha)",
            min_value=0.0, max_value=1.0, value=0.6, step=0.05,
            help="1.0 = chi dung anh, 0.0 = chi dung van ban.",
        )

    search_clicked = st.sidebar.button("Tim kiem", type="primary", use_container_width=True)

    return {
        "mode": mode,
        "user_id": user_id,
        "top_k": top_k,
        "alpha_h": alpha_h,
        "alpha_img_txt": alpha_img_txt,
        "uploaded_image": uploaded_image,
        "query_text": query_text,
        "search_clicked": search_clicked,
    }


def validate_query(mode, image_pil, query_text):
    if mode == "Hinh anh" and image_pil is None:
        return "Vui long tai len mot anh."
    if mode == "Van ban" and not query_text.strip():
        return "Vui long nhap mo ta van ban."
    if mode == "Ket hop" and image_pil is None and not query_text.strip():
        return "Vui long cung cap it nhat anh hoac van ban."
    return None


def reason_tag(row):
    """Sinh nhan giai thich ngan gon cho tung ket qua (explainability co ban cho Tuan 8)."""
    if row.has_cf:
        return "Nguoi dung tuong tu thich (CF)"
    return "Tuong dong noi dung (CLIP)"


def display_grid(rec: FashionHybridRecommender, results, n_cols=N_COLS):
    rows = list(results.itertuples(index=False))
    for start in range(0, len(rows), n_cols):
        cols = st.columns(n_cols)
        for col, row in zip(cols, rows[start:start + n_cols]):
            with col:
                match = rec.df_meta.loc[rec.df_meta["item_id"] == row.item_id, "image_path"]
                img_path = match.iloc[0] if not match.empty else None

                img_path = Path(img_path) if img_path else None
                if img_path and img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                elif img_path and (IMAGE_ROOT / img_path.name).exists():
                    st.image(str(IMAGE_ROOT / img_path.name), use_container_width=True)
                else:
                    st.caption("Khong co anh")

                score_pct = row.s_H * 100
                st.markdown(f"**#{row.rank} - {score_pct:.1f}%**")
                st.caption(f"{row.category} / {row.subcategory}")
                st.caption(str(row.description)[:60])
                st.caption(reason_tag(row))


def main():
    st.set_page_config(page_title="Fashion Recommender", layout="wide")
    st.title("He thong Goi y Thoi trang Da phuong thuc")
    st.caption("Fine-tuned CLIP + FAISS + Collaborative Filtering (SVD) - Hybrid Search")

    try:
        rec = load_recommender()
    except Exception as exc:  # noqa: BLE001 - hien loi ro rang cho nguoi dung app
        st.error(f"Khong load duoc model/artifacts: {exc}")
        st.info(
            "Kiem tra lai FAISS_ARTIFACT_DIR / CLIP_CHECKPOINT_PATH / CF_MODEL_PATH "
            "o dau file app.py cho dung voi may ban."
        )
        st.stop()

    params = render_sidebar(rec)

    if not params["search_clicked"]:
        st.info("Chon che do tim kiem o sidebar roi bam **Tim kiem**.")
        return

    query_image_pil = None
    if params["uploaded_image"] is not None:
        query_image_pil = Image.open(params["uploaded_image"]).convert("RGB")

    error_msg = validate_query(params["mode"], query_image_pil, params["query_text"])
    if error_msg:
        st.warning(error_msg)
        return

    if query_image_pil is not None:
        st.sidebar.image(query_image_pil, caption="Anh query", use_container_width=True)

    t0 = time.time()
    try:
        results = rec.hybrid_recommend(
            query_image=query_image_pil,
            query_text=params["query_text"].strip() or None,
            user_id=params["user_id"],
            top_k=params["top_k"],
            alpha_h=params["alpha_h"],
            alpha=params["alpha_img_txt"],
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Loi khi tim kiem: {exc}")
        return
    elapsed_ms = (time.time() - t0) * 1000

    st.caption(f"Thoi gian phan hoi: {elapsed_ms:.0f} ms | {len(results)} ket qua")

    if results.empty:
        st.warning("Khong tim thay ket qua phu hop.")
        return

    display_grid(rec, results)


if __name__ == "__main__":
    main()
