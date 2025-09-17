# app.py — Instacart Recommender demo (robust; histories + weight sliders + SHAP)
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
import joblib
import streamlit as st

# Optional deps for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

# --------------------------------
# Streamlit page setup
# --------------------------------
st.set_page_config(page_title="Instacart Recs", layout="wide")
st.title("🛒 Instacart Recommender Demo")

# Script-relative paths (robust regardless of cwd)
BASE_DIR = Path(__file__).resolve().parent
artifacts = BASE_DIR / "artifacts"
model_dir = artifacts / "model"
recs_dir = artifacts / "recs"
metrics_dir = artifacts / "metrics"

def fail(msg: str, exc: Exception | None = None):
    st.error(msg)
    if exc:
        st.exception(exc)
    st.stop()

def _mm(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mx, mn = x.max(), x.min()
    return (x - mn) / (mx - mn + 1e-9) if mx > mn else np.zeros_like(x)

# --------------------------------
# Load model artifacts (robust)
# --------------------------------
st.subheader("Model Artifacts")
if not model_dir.exists():
    fail(f"Folder not found: {model_dir}. Export your model artifacts first.")

# Indices/maps
try:
    maps = joblib.load(model_dir / "indices_maps.joblib")
    users_index = pd.Index(maps["users_index"], name="user_id")
    items_index = pd.Index(maps["items_index"], name="product_id")
    u2i, p2i = maps["u2i"], maps["p2i"]
except Exception as e:
    fail("Failed to load indices_maps.joblib (missing/corrupt).", e)

# Matrices
try:
    II = sparse.load_npz(model_dir / "II_covisit.npz").tocsr()
    Xc = sparse.load_npz(model_dir / "X_content.npz").tocsr()
    pop = np.load(model_dir / "popularity.npy").astype(np.float32)
except Exception as e:
    fail("Missing or unreadable matrices (II_covisit.npz / X_content.npz / popularity.npy).", e)

# Optional embeddings
Uemb = Vemb = None
try:
    uemb_path = model_dir / "Uemb.npy"
    vemb_path = model_dir / "Vemb.npy"
    if uemb_path.exists() and vemb_path.exists():
        Uemb = np.load(uemb_path)
        Vemb = np.load(vemb_path)
except Exception as e:
    st.warning(f"Embeddings failed to load, continuing without. {e}")

# Precompute norms for content similarity
try:
    Xc_row_norm = np.sqrt(Xc.multiply(Xc).sum(axis=1)).A1 + 1e-9
except Exception as e:
    fail("Could not compute content row norms.", e)

# User histories (personalization)
user_seen = {}
try:
    us_fp = model_dir / "user_seen.joblib"
    if us_fp.exists():
        user_seen = joblib.load(us_fp)  # dict: uidx -> set(item_idx)
        st.success(f"Loaded user histories: {len(user_seen):,} users.")
    else:
        st.info("Using empty histories (user_seen.joblib not found).")
except Exception as e:
    st.warning(f"Failed to load user_seen; continuing without histories. {e}")

st.caption(f"Users: {len(users_index):,} | Items: {len(items_index):,}")

# --------------------------------
# Catalog metadata (for names)
# --------------------------------
st.subheader("Catalog Metadata")
data_dir = st.text_input(
    "Path to your Instacart data folder (must contain products.csv, aisles.csv, departments.csv):",
    value=str(BASE_DIR),
)
prods = None
try:
    prods = (
        pd.read_csv(Path(data_dir) / "products.csv")
        .merge(pd.read_csv(Path(data_dir) / "aisles.csv"), on="aisle_id", how="left")
        .merge(pd.read_csv(Path(data_dir) / "departments.csv"), on="department_id", how="left")
    )
    st.success("Loaded product metadata.")
except Exception:
    st.warning("Could not load product metadata; showing IDs only until you set a correct folder.")

# --------------------------------
# User picker & weights
# --------------------------------
st.subheader("Recommend")
try:
    recs_fp = recs_dir / "user_recommendations_topk.csv"
    if recs_fp.exists():
        recs_hist = pd.read_csv(recs_fp)
        user_choices = sorted(recs_hist["user_id"].dropna().astype(int).unique().tolist()[:5000])
    else:
        user_choices = users_index.tolist()[:5000]
    if not user_choices:
        fail("No users found in artifacts. Check your export step.")
except Exception as e:
    fail("Could not prepare user list.", e)

uid = st.selectbox("Choose a user", user_choices)
K = st.slider("Top-K", min_value=5, max_value=30, value=10)

st.subheader("Weights")
c1, c2, c3, c4 = st.columns(4)
alpha = c1.slider("α CF", 0.0, 1.0, 0.60, 0.05)
beta  = c2.slider("β Content", 0.0, 1.0, 0.20, 0.05)
gamma = c3.slider("γ Emb", 0.0, 1.0, 0.10, 0.05)
delta = c4.slider("δ Pop", 0.0, 1.0, 0.10, 0.05)
w_sum = alpha + beta + gamma + delta or 1.0
alpha, beta, gamma, delta = alpha / w_sum, beta / w_sum, gamma / w_sum, delta / w_sum

# --------------------------------
# Scoring (returns total + components)
# --------------------------------
def score_user(uidx: int, seen: set[int], alpha=0.60, beta=0.20, gamma=0.10, delta=0.10, max_neighbors=80):
    """
    Hybrid score = α * item-CF (co-vis) + β * content + γ * embeddings + δ * popularity
    Returns (total, cf_norm, content_norm, emb_norm, pop_norm)
    """
    n_items = II.shape[0]
    s_cf = np.zeros(n_items, dtype=np.float32)
    s_ct = np.zeros(n_items, dtype=np.float32)
    s_em = np.zeros(n_items, dtype=np.float32)
    s_po = pop.copy()

    seen_idx = np.array(sorted(list(seen)), dtype=int)
    if seen_idx.size > 0:
        sims = II[:, seen_idx].toarray()
        if max_neighbors and sims.size > 0:
            for j in range(sims.shape[1]):
                col = sims[:, j]
                if max_neighbors < len(col):
                    t = np.partition(col, -max_neighbors)[-max_neighbors]
                    col[col < t] = 0.0
                    sims[:, j] = col
        s_cf = sims.sum(axis=1, dtype=np.float32)

        centroid = np.asarray(Xc[seen_idx, :].mean(axis=0)).ravel()
        nc = float(np.linalg.norm(centroid))
        if nc > 0:
            numer = Xc @ centroid
            denom = Xc_row_norm * (nc + 1e-9)
            s_ct = (np.asarray(numer).ravel() / denom).astype(np.float32)

    if Uemb is not None and Vemb is not None:
        u = Uemb[uidx:uidx+1, :]
        An = np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
        Bn = (np.linalg.norm(Vemb, axis=1, keepdims=True) + 1e-9).T
        s_em = ((u @ Vemb.T).ravel() / (An * Bn).ravel()).astype(np.float32)

    cf_n = _mm(s_cf); ct_n = _mm(s_ct); em_n = _mm(s_em); po_n = _mm(s_po)
    total = alpha * cf_n + beta * ct_n + gamma * em_n + delta * po_n

    if seen:
        total[list(seen)] += 0.05  # gentle reorder boost

    return total, cf_n, ct_n, em_n, po_n

# Build per-item channel features for SHAP surrogate
def channel_table_for_user(uidx: int) -> pd.DataFrame:
    """
    Returns a DataFrame with normalized channels: item_cf, content, embeddings, popularity (for ALL items).
    """
    n_items = II.shape[0]
    seen = user_seen.get(uidx, set())

    # item-CF
    s_cf = np.zeros(n_items, dtype=np.float32)
    if seen:
        sims = II[:, np.array(sorted(list(seen)), dtype=int)].toarray()
        s_cf = sims.sum(axis=1, dtype=np.float32)

    # content centroid cosine
    s_ct = np.zeros(n_items, dtype=np.float32)
    if seen:
        centroid = np.asarray(Xc[list(seen), :].mean(axis=0)).ravel()
        nc = float(np.linalg.norm(centroid))
        if nc > 0:
            numer = Xc @ centroid
            denom = Xc_row_norm * (nc + 1e-9)
            s_ct = (np.asarray(numer).ravel() / denom).astype(np.float32)

    # embedding cosine
    s_em = np.zeros(n_items, dtype=np.float32)
    if (Uemb is not None) and (Vemb is not None):
        u = Uemb[uidx:uidx+1, :]
        An = np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
        Bn = (np.linalg.norm(Vemb, axis=1, keepdims=True) + 1e-9).T
        s_em = ((u @ Vemb.T).ravel() / (An * Bn).ravel()).astype(np.float32)

    s_po = pop.astype(np.float32)

    return pd.DataFrame({
        "item_cf": _mm(s_cf),
        "content": _mm(s_ct),
        "embeddings": _mm(s_em),
        "popularity": _mm(s_po),
    })

# --------------------------------
# Inference & display
# --------------------------------
if uid not in u2i:
    fail("This user isn't in the exported index. Pick another user.")

uidx = u2i[uid]
seen = user_seen.get(uidx, set())

try:
    scores, cf_n, ct_n, em_n, po_n = score_user(uidx, seen, alpha, beta, gamma, delta)
    top = np.argsort(-scores)[:K]
    out = pd.DataFrame({
        "rank": np.arange(1, K + 1),
        "product_id": items_index[top].astype(int),
        "score_total": scores[top],
        "contrib_cf": cf_n[top],
        "contrib_content": ct_n[top],
        "contrib_emb": em_n[top],
        "contrib_pop": po_n[top],
    })
    if prods is not None:
        out = out.merge(prods[["product_id", "product_name", "department", "aisle"]],
                        on="product_id", how="left")
    st.subheader("Recommendations")
    st.dataframe(out, width="stretch")

    st.download_button(
        "Download these recommendations (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"recs_user_{uid}_top{K}.csv",
        mime="text/csv",
    )
except Exception as e:
    fail("Scoring failed.", e)

# --------------------------------
# SHAP expander (per-user channel explainability)
# --------------------------------
with st.expander("🔎 Explain this user's relevance (SHAP)"):
    st.caption("Trains a tiny surrogate model on the 4 channels for this user, then shows mean |SHAP| per channel.")
    if not SHAP_AVAILABLE or not MPL_AVAILABLE:
        st.info("SHAP/Matplotlib not installed in this environment. Install with: `pip install shap matplotlib`.")
    else:
        if st.button("Compute SHAP for this user"):
            # 1) Build channel features for ALL items
            X = channel_table_for_user(uidx)

            # 2) Labels: treat user's seen items as positives (fast proxy)
            y = np.zeros(len(X), dtype=int)
            if user_seen:
                seen_idx = np.array(sorted(list(user_seen.get(uidx, set()))), dtype=int)
                y[seen_idx] = 1
            X["y"] = y

            if X["y"].nunique() < 2:
                st.warning("This user has only one class of items (all seen or all unseen); pick another user.")
                st.stop()

            # 3) Downsample negatives to keep it snappy
            pos = X[X.y == 1]
            neg = X[X.y == 0]
            if len(neg) > 20000:
                neg = neg.sample(20000, random_state=42)
            Xs = pd.concat([pos, neg], ignore_index=True)

            # 4) Train a small surrogate (balanced logistic regression)
            from sklearn.linear_model import LogisticRegression
            feat_cols = ["item_cf", "content", "popularity"] + (["embeddings"] if (Uemb is not None and Vemb is not None) else [])
            clf = LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")
            clf.fit(Xs[feat_cols], Xs["y"])

            # Optional sanity metric
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(Xs["y"], clf.predict_proba(Xs[feat_cols])[:, 1])
                st.caption(f"Surrogate AUC: {auc:.3f}")
            except Exception:
                pass

            # 5) SHAP values (new API first, fallback to LinearExplainer)
            try:
                explainer = shap.Explainer(clf, Xs[feat_cols])
                sv = explainer(Xs[feat_cols])
                shap_values = sv.values
            except Exception:
                explainer = shap.LinearExplainer(clf, Xs[feat_cols], feature_perturbation="interventional")
                sv = explainer.shap_values(Xs[feat_cols])
                shap_values = sv if isinstance(sv, np.ndarray) else (sv[1] if isinstance(sv, list) and len(sv) > 1 else sv[0])

            shap_abs = np.abs(shap_values).mean(axis=0)
            summary = pd.DataFrame({"feature": feat_cols, "mean|SHAP|": shap_abs}).sort_values("mean|SHAP|", ascending=True)

            # 6) Bar chart
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            ax.barh(summary["feature"], summary["mean|SHAP|"])
            ax.set_xlabel("mean |SHAP|")
            ax.set_title(f"User {uid}: Channel importance")
            st.pyplot(fig)

            # Download summary CSV
            st.download_button(
                "Download SHAP summary (CSV)",
                data=summary.sort_values("mean|SHAP|", ascending=False).to_csv(index=False).encode("utf-8"),
                file_name=f"shap_summary_user_{uid}.csv",
                mime="text/csv",
            )
