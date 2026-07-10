import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


class FashionHybridRecommender:
    def __init__(
        self,
        artifact_dir,
        cf_model_path=None,
        clip_checkpoint_path=None,
        interactions_path=None,
        device=None,
        clip_model_name="ViT-B/32",
    ):
        self.artifact_dir = Path(artifact_dir)
        self.cf_model_path = Path(cf_model_path) if cf_model_path else self.artifact_dir / "cf_model.pkl"
        self.clip_checkpoint_path = Path(clip_checkpoint_path) if clip_checkpoint_path else None
        self.interactions_path = Path(interactions_path) if interactions_path else self.artifact_dir / "interactions.csv"
        self.clip_model_name = clip_model_name
        self.device = device

        self._load_faiss_artifacts()
        self._load_cf_model()
        self._load_interactions()
        self._load_clip_model()

    def _load_faiss_artifacts(self):
        import faiss

        required = [
            "fashion.index",
            "metadata_indexed.csv",
            "item_ids.npy",
            "index_config.json",
        ]
        for fname in required:
            path = self.artifact_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"Missing required artifact: {path}")

        self.faiss_index = faiss.read_index(str(self.artifact_dir / "fashion.index"))
        self.df_meta = pd.read_csv(self.artifact_dir / "metadata_indexed.csv")
        self.df_meta["item_id"] = self.df_meta["item_id"].astype(int)

        self.item_ids_arr = np.load(self.artifact_dir / "item_ids.npy")
        if not (len(self.item_ids_arr) == self.faiss_index.ntotal == len(self.df_meta)):
            raise ValueError("item_ids, FAISS index, and metadata lengths must match.")

        self.faiss_idx_to_item_id = self.item_ids_arr.astype(int)
        self.item_id_to_faiss_idx = {int(iid): idx for idx, iid in enumerate(self.item_ids_arr)}

        with open(self.artifact_dir / "index_config.json", "r", encoding="utf-8") as f:
            self.index_config = json.load(f)
        self.alpha_index = self.index_config.get("alpha", 0.6)

    def _load_cf_model(self):
        if not self.cf_model_path.exists():
            raise FileNotFoundError(f"Missing CF model: {self.cf_model_path}")

        with open(self.cf_model_path, "rb") as f:
            payload = pickle.load(f)

        self.U_CF = payload["U"]
        self.V_CF = payload["V"]
        self.svd = payload.get("svd")
        self.user_mapping = payload["user_mapping"]
        self.item_mapping = payload["item_mapping"]
        self.inv_user_mapping = payload.get("inv_user_mapping", {v: k for k, v in self.user_mapping.items()})
        self.inv_item_mapping = payload.get("inv_item_mapping", {v: k for k, v in self.item_mapping.items()})
        self.n_components = payload.get("n_components", self.U_CF.shape[1])
        self.n_users = payload.get("n_users", self.U_CF.shape[0])
        self.n_items = payload.get("n_items", self.V_CF.shape[0])

    def _load_interactions(self):
        if self.interactions_path.exists():
            self.df_inter = pd.read_csv(self.interactions_path)
            self.df_inter["user_id"] = self.df_inter["user_id"].astype(int)
            self.df_inter["item_id"] = self.df_inter["item_id"].astype(int)
        else:
            self.df_inter = pd.DataFrame(columns=["user_id", "item_id", "rating"])

    def _load_clip_model(self):
        import torch
        import torch.nn.functional as F
        import clip

        self.torch = torch
        self.F = F
        self.clip = clip

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.clip_checkpoint_path is None:
            raise ValueError("clip_checkpoint_path is required to match the FAISS index embedding space.")
        if not self.clip_checkpoint_path.exists():
            raise FileNotFoundError(f"Missing fine-tuned CLIP checkpoint: {self.clip_checkpoint_path}")

        self.clip_model, self.clip_preprocess = clip.load(
            self.clip_model_name,
            device=self.device,
            jit=False,
        )
        self.clip_model.eval()

        try:
            ckpt = torch.load(self.clip_checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.clip_checkpoint_path, map_location=self.device)

        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.clip_model.load_state_dict(state_dict)
        self.clip_model.eval()

    @staticmethod
    def _minmax_norm(arr, eps=1e-9):
        lo, hi = arr.min(), arr.max()
        if hi - lo < eps:
            return np.zeros_like(arr, dtype=np.float32), True
        return ((arr - lo) / (hi - lo)).astype(np.float32), False

    @staticmethod
    def _empty_hybrid_result():
        return pd.DataFrame(columns=[
            "rank", "item_id", "faiss_rank", "s_sem", "s_sem_norm",
            "cf_score", "c_ui_norm", "s_H", "has_cf",
            "category", "subcategory", "description",
        ])

    def _seen_item_ids(self, user_id):
        if self.df_inter.empty:
            return set()
        return set(self.df_inter.loc[self.df_inter["user_id"].eq(user_id), "item_id"].astype(int))

    def cf_recommend(self, user_id, top_k=30):
        if user_id not in self.user_mapping or top_k <= 0:
            return []

        u_idx = self.user_mapping[user_id]
        p_u = self.U_CF[u_idx]
        scores = (self.V_CF @ p_u).astype(np.float64)

        seen_item_ids = self._seen_item_ids(user_id)
        if seen_item_ids:
            seen_indices = [self.item_mapping[iid] for iid in seen_item_ids if iid in self.item_mapping]
            scores[seen_indices] = -np.inf

        valid_count = int(np.isfinite(scores).sum())
        if valid_count == 0:
            return []

        top_k_eff = min(top_k, valid_count)
        top_indices = np.argpartition(scores, -top_k_eff)[-top_k_eff:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            {"item_id": self.inv_item_mapping[idx], "cf_score": float(scores[idx])}
            for idx in top_indices
        ]

    def encode_text_query(self, text):
        with self.torch.no_grad():
            tokens = self.clip.tokenize([text], truncate=True).to(self.device)
            feat = self.clip_model.encode_text(tokens)
            feat = self.F.normalize(feat.float(), dim=-1)
        return feat.cpu().numpy().astype("float32")

    def encode_image_query(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        with self.torch.no_grad():
            img_t = self.clip_preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
            feat = self.clip_model.encode_image(img_t)
            feat = self.F.normalize(feat.float(), dim=-1)
        return feat.cpu().numpy().astype("float32")

    def make_query_vector(self, image=None, text=None, alpha=None):
        if image is None and text is None:
            raise ValueError("At least one of image or text is required.")

        alpha = self.alpha_index if alpha is None else alpha

        if image is not None and text is not None:
            img_vec = self.encode_image_query(image)
            txt_vec = self.encode_text_query(text)
            combined = alpha * img_vec + (1 - alpha) * txt_vec
            norm = np.linalg.norm(combined, axis=1, keepdims=True)
            return (combined / np.where(norm > 0, norm, 1.0)).astype("float32")

        if image is not None:
            return self.encode_image_query(image)

        return self.encode_text_query(text)

    def hybrid_recommend(self, query_image=None, query_text=None, user_id=None, top_k=10, alpha_h=0.6, alpha=None):
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        is_cold_start = (user_id is None) or (user_id not in self.user_mapping)
        if is_cold_start:
            alpha_h = 1.0

        # alpha: trong so image vs text khi ket hop ca hai (None -> dung alpha_index mac dinh
        # duoc luu trong index_config.json luc build FAISS index).
        query_vec = self.make_query_vector(image=query_image, text=query_text, alpha=alpha)
        seen_item_ids = set() if is_cold_start else self._seen_item_ids(user_id)

        search_k = min(max(top_k * 3, top_k), self.faiss_index.ntotal)
        cand_item_ids = []
        sem_scores = np.array([], dtype=np.float64)
        faiss_idxs = np.array([], dtype=np.int64)

        while True:
            raw_scores, raw_indices = self.faiss_index.search(query_vec, search_k)
            current_scores = raw_scores[0].astype(np.float64)
            current_faiss_idxs = raw_indices[0]

            valid_mask = current_faiss_idxs >= 0
            current_scores = current_scores[valid_mask]
            current_faiss_idxs = current_faiss_idxs[valid_mask]

            if len(current_faiss_idxs) == 0:
                return self._empty_hybrid_result()

            current_item_ids = self.faiss_idx_to_item_id[current_faiss_idxs].astype(int).tolist()

            if seen_item_ids:
                keep_mask = np.array([iid not in seen_item_ids for iid in current_item_ids], dtype=bool)
                current_scores = current_scores[keep_mask]
                current_faiss_idxs = current_faiss_idxs[keep_mask]
                current_item_ids = [iid for iid, keep in zip(current_item_ids, keep_mask) if keep]

            cand_item_ids = current_item_ids
            sem_scores = current_scores
            faiss_idxs = current_faiss_idxs

            if len(cand_item_ids) >= top_k or search_k >= self.faiss_index.ntotal:
                break

            search_k = min(search_k * 2, self.faiss_index.ntotal)

        m = len(cand_item_ids)
        if m == 0:
            return self._empty_hybrid_result()

        cf_scores_raw = np.zeros(m, dtype=np.float64)
        has_cf_flag = np.zeros(m, dtype=bool)

        if not is_cold_start:
            u_idx = self.user_mapping[user_id]
            p_u = self.U_CF[u_idx].astype(np.float64)

            for pos, iid in enumerate(cand_item_ids):
                if iid in self.item_mapping:
                    i_idx = self.item_mapping[iid]
                    q_i = self.V_CF[i_idx].astype(np.float64)
                    cf_scores_raw[pos] = float(p_u @ q_i)
                    has_cf_flag[pos] = True

        s_sem_norm, sem_trivial = self._minmax_norm(sem_scores)

        c_ui_norm = np.zeros(m, dtype=np.float32)
        if has_cf_flag.any():
            cf_norm_subset, cf_trivial = self._minmax_norm(cf_scores_raw[has_cf_flag])
            c_ui_norm[has_cf_flag] = cf_norm_subset
        else:
            cf_trivial = True

        if sem_trivial and cf_trivial:
            s_sem_norm = (sem_scores / max(sem_scores.max(), 1e-9)).astype(np.float32)

        s_H = np.where(
            has_cf_flag,
            alpha_h * s_sem_norm + (1.0 - alpha_h) * c_ui_norm,
            s_sem_norm,
        ).astype(np.float32)

        ranked_pos = np.argsort(s_H)[::-1][:top_k]
        result_rows = []
        for rank, pos in enumerate(ranked_pos):
            result_rows.append({
                "rank": rank + 1,
                "item_id": cand_item_ids[pos],
                "faiss_rank": int(pos + 1),
                "s_sem": float(sem_scores[pos]),
                "s_sem_norm": float(s_sem_norm[pos]),
                "cf_score": float(cf_scores_raw[pos]),
                "c_ui_norm": float(c_ui_norm[pos]),
                "s_H": float(s_H[pos]),
                "has_cf": bool(has_cf_flag[pos]),
            })

        df_result = pd.DataFrame(result_rows)
        return df_result.merge(
            self.df_meta[["item_id", "category", "subcategory", "description"]].astype({"item_id": int}),
            on="item_id",
            how="left",
        )
