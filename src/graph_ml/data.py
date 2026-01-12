from __future__ import annotations

import json, re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling

from graph_ml.utils import set_seed

USER = "user"
BUS  = "business"

REL     = (USER, "reviews", BUS)
FRIENDS = (USER, "friends", USER)

_SENT = None

def _get_sentiment_analyzer():
    """
    Returns a VADER sentiment analyzer.
    Prefers vaderSentiment package, falls back to nltk if available.
    """
    global _SENT
    if _SENT is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _SENT = SentimentIntensityAnalyzer()
        except Exception:
            print("vaderSentiment not available")
            return None
    return _SENT

def _vader_scores(text: str, *, mode: str = "vader4", analyser=None) -> np.ndarray:
    """
    mode:
      - "vader4": [neg, neu, pos, compound]
      - "compound": [compound]
    """
    s = analyser.polarity_scores(text or "")
    if mode == "compound":
        return np.array([s["compound"]], dtype=np.float32)
    return np.array([s["neg"], s["neu"], s["pos"], s["compound"]], dtype=np.float32)

# ---------- Word2Vec (lazy, cached) ----------
_W2V = None
_W2V_DIM_DEFAULT = 300
_TOKEN_PAT = re.compile(r"[A-Za-z']+")

def _get_w2v(model_name: str = "word2vec-google-news-300"):
    global _W2V
    if _W2V is None:
        import gensim.downloader as api
        _W2V = api.load(model_name)
    return _W2V

def _simple_tokenize(s: str) -> List[str]:
    return [t.lower() for t in _TOKEN_PAT.findall(s)]

def _mean_w2v(tokens: Sequence[str], w2v, dim: int) -> np.ndarray:
    known = [t for t in tokens if t in w2v.key_to_index]
    if not known:
        return np.zeros(dim, dtype=np.float32)
    vec = np.zeros(dim, dtype=np.float32)
    for t in known:
        vec += w2v.get_vector(t)
    vec /= float(len(known))
    return vec

# ---------- Utilities ----------
def _file_fingerprint(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}-{st.st_mtime_ns}"

def _parse_friends_field(s: Optional[str]) -> List[str]:
    if not s or s == "None":
        return []
    return [t.strip() for t in s.split(",") if t.strip()]

# ---------- Main loader ----------
def load_yelp_as_hetero(
    data_dir: Path | str,
    *,
    max_reviews: Optional[int] = 100_000,
    min_review_len: int = 5,
    seed: int = 42,
    cache: bool = True,
    cache_subdir: str = "processed",
    include_user_friends: bool = True,
    max_friends_per_user: Optional[int] = 50,
    use_text_edge_attr: bool = True,
    w2v_model_name: str = "word2vec-google-news-300",
    w2v_dim: Optional[int] = None,
    use_sentiment_edge_attr: bool = True,
    sentiment_mode: str = "vader4",   # "vader4" or "compound"
    zscore_sentiment: bool = False,   # optional normalization
) -> HeteroData:

    set_seed(seed)
    data_dir = Path(data_dir)

    # ----- cache: load if present and compatible -----
    cache_path = data_dir / cache_subdir / "yelp_hetero.pt"
    settings = {
        "max_reviews": max_reviews,
        "min_review_len": min_review_len,
        "include_user_friends": include_user_friends,
        "max_friends_per_user": max_friends_per_user,
        "use_text_edge_attr": use_text_edge_attr,
        "w2v_model_name": w2v_model_name,
        "w2v_dim": w2v_dim,
        "use_sentiment_edge_attr": use_sentiment_edge_attr,
        "sentiment_mode": sentiment_mode,
        "zscore_sentiment": zscore_sentiment,
    }
    if cache and cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "data" in obj and "meta" in obj:
            cached_settings = obj["meta"].get("settings")
            if cached_settings == settings:
                print(f"[cache] Loaded preprocessed graph: {cache_path}")
                return obj["data"]
            print(f"[cache] Settings changed; rebuilding cache: {cache_path}")
        else:
            print(f"[cache] Legacy cache format; rebuilding: {cache_path}")

    # ----- required files -----
    fp_rev  = data_dir / "yelp_academic_dataset_review.json"
    fp_user = data_dir / "yelp_academic_dataset_user.json"
    fp_biz  = data_dir / "yelp_academic_dataset_business.json"
    for f in (fp_rev, fp_user, fp_biz):
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")

    # ----- optional W2V -----
    if use_text_edge_attr:
        w2v = _get_w2v(w2v_model_name)
        w2v_dim_eff = int(w2v_dim or _W2V_DIM_DEFAULT)  # 300
    else:
        w2v = None
        w2v_dim_eff = 0

    if use_sentiment_edge_attr:
        sent_dim_eff = 4 if sentiment_mode == "vader4" else 1
        an = _get_sentiment_analyzer()
    else:
        sent_dim_eff = 0

    # ----- parse reviews -----
    rev_rows: List[Tuple[str, str, float, int]] = []  # (uid,bid,stars,year)
    rev_embs: List[np.ndarray] = []
    rev_sents: List[np.ndarray] = []

    with fp_rev.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_reviews is not None and i >= max_reviews:
                break
            r = json.loads(line)

            txt = (r.get("text") or "")
            if len(txt) < min_review_len:
                continue
            uid = r.get("user_id")
            bid = r.get("business_id")
            if not uid or not bid:
                continue

            stars = float(r.get("stars", 0.0))
            date_str = r.get("date") or ""
            try:
                year = int(date_str[:4]) if len(date_str) >= 4 else -1
            except Exception:
                year = -1

            rev_rows.append((uid, bid, stars, year))

            if use_text_edge_attr:
                emb = _mean_w2v(_simple_tokenize(txt), w2v, w2v_dim_eff)
                rev_embs.append(emb)

            if use_sentiment_edge_attr:
                rev_sents.append(_vader_scores(txt, mode=sentiment_mode, analyser=an))

    if not rev_rows:
        raise RuntimeError("No reviews loaded. Increase max_reviews or lower min_review_len.")

    # ----- collect valid ids + review_count from json -----
    user_ids, biz_ids = set(), set()
    user_rc_json: Dict[str, int] = {}   # user_id -> review_count from user.json
    biz_rc_json:  Dict[str, int] = {}   # business_id -> review_count from business.json

    with fp_user.open("r", encoding="utf-8") as fh:
        for line in fh:
            j = json.loads(line)
            if (uid := j.get("user_id")):
                user_ids.add(uid)
                user_rc_json[uid] = int(j.get("review_count", 0) or 0)

    with fp_biz.open("r", encoding="utf-8") as fh:
        for line in fh:
            j = json.loads(line)
            if (bid := j.get("business_id")):
                biz_ids.add(bid)
                biz_rc_json[bid] = int(j.get("review_count", 0) or 0)


    if use_text_edge_attr or use_sentiment_edge_attr:
        feats = []
        # build tuples (row, maybe_w2v, maybe_sent)
        for k, row in enumerate(rev_rows):
            u, b, s, y = row
            if (u in user_ids and b in biz_ids):
                w = rev_embs[k] if use_text_edge_attr else None
                se = rev_sents[k] if use_sentiment_edge_attr else None
                feats.append((row, w, se))

        if not feats:
            raise RuntimeError("No overlapping user/business ids with reviews.")

        rev_rows = [r for (r, _, _) in feats]
        if use_text_edge_attr:
            rev_embs = [w for (_, w, _) in feats]  # type: ignore
        if use_sentiment_edge_attr:
            rev_sents = [se for (_, _, se) in feats]  # type: ignore
    else:
        rev_rows = [(u, b, s, y) for (u, b, s, y) in rev_rows if (u in user_ids and b in biz_ids)]
        if not rev_rows:
            raise RuntimeError("No overlapping user/business ids with reviews.")

    # ----- index maps -----
    uniq_users = sorted({u for (u, _, _, _) in rev_rows})
    uniq_biz   = sorted({b for (_, b, _, _) in rev_rows})
    u2i = {u: i for i, u in enumerate(uniq_users)}
    b2i = {b: i for i, b in enumerate(uniq_biz)}

    # ----- edge tensors -----
    u_idx = np.fromiter((u2i[u] for (u, _, _, _) in rev_rows), dtype=np.int64)
    b_idx = np.fromiter((b2i[b] for (_, b, _, _) in rev_rows), dtype=np.int64)
    stars = np.fromiter((s for (*_, s, _) in rev_rows), dtype=np.float32)
    years = np.fromiter((y for (*_, y) in rev_rows), dtype=np.int16)

    # ----- degrees in the (filtered) reviews bipartite graph -----
    # Users: how many reviews they wrote in *this* processed set
    deg_user_rev = np.bincount(u_idx, minlength=len(u2i)).astype(np.int64)          # [N_user]
    # Businesses: how many reviews they received in *this* processed set
    deg_biz_rev  = np.bincount(b_idx, minlength=len(b2i)).astype(np.int64)          # [N_biz]

    # ----- build heterodata -----
    data = HeteroData()
    data[USER].num_nodes = len(uniq_users)
    data[BUS].num_nodes  = len(uniq_biz)

    # ---------- node features ----------
    # Users: [log1p(review_count_json), log1p(degree_in_reviews_graph)]
    user_feat = np.zeros((len(u2i), 2), dtype=np.float32)
    for uid, i in u2i.items():
        rc_json = user_rc_json.get(uid, 0)
        user_feat[i, 0] = np.log1p(rc_json)
        user_feat[i, 1] = np.log1p(deg_user_rev[i])

    # Businesses: [log1p(review_count_json), log1p(degree_in_reviews_graph)]
    biz_feat = np.zeros((len(b2i), 2), dtype=np.float32)
    for bid, j in b2i.items():
        rc_json = biz_rc_json.get(bid, 0)
        biz_feat[j, 0] = np.log1p(rc_json)
        biz_feat[j, 1] = np.log1p(deg_biz_rev[j])

    data[USER].x = torch.tensor(user_feat, dtype=torch.float32)
    data[BUS].x  = torch.tensor(biz_feat,  dtype=torch.float32)

    edge_index = torch.tensor(np.vstack([u_idx, b_idx]), dtype=torch.long)
    data[REL].edge_index = edge_index
    data[REL].edge_label = torch.tensor(stars, dtype=torch.long)   # stars as integer labels
    data[REL].time       = torch.tensor(years, dtype=torch.long).view(-1)

    # ---- edge_attr = [W2V || SENT || time_zscore] ----
    time_f = data[REL].time.to(torch.float32).view(-1, 1)
    time_f = (time_f - time_f.mean()) / (time_f.std() + 1e-6)  # [E,1]

    parts = []

    if use_text_edge_attr:
        w2v_mat = torch.tensor(np.stack(rev_embs, axis=0), dtype=torch.float32)  # [E,300]
        w = w2v_mat / (w2v_mat.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        parts.append(w)

    if use_sentiment_edge_attr:
        s_mat = torch.tensor(np.stack(rev_sents, axis=0), dtype=torch.float32)   # [E,sent_dim]
        if zscore_sentiment:
            s_mat = (s_mat - s_mat.mean(dim=0, keepdim=True)) / (s_mat.std(dim=0, keepdim=True) + 1e-6)
        parts.append(s_mat)

    parts.append(time_f)

    data[REL].edge_attr = torch.cat(parts, dim=-1)
    edge_attr_dim = int(data[REL].edge_attr.size(-1))  # final width


    # ----- friends (raw, directed for now) -----
    if include_user_friends:
        friend_pairs: List[Tuple[int, int]] = []
        with fp_user.open("r", encoding="utf-8") as fh:
            for line in fh:
                j = json.loads(line)
                uid = j.get("user_id")
                if uid not in u2i:
                    continue
                u_local = u2i[uid]
                friends = _parse_friends_field(j.get("friends", "None"))
                if max_friends_per_user is not None and len(friends) > max_friends_per_user:
                    friends = friends[:max_friends_per_user]
                for fid in friends:
                    v = u2i.get(fid)
                    if v is not None and v != u_local:
                        friend_pairs.append((u_local, v))

        if friend_pairs:
            ei = torch.tensor(friend_pairs, dtype=torch.long).t().contiguous()   # [2, E_f_raw]
        else:
            ei = torch.empty((2, 0), dtype=torch.long)

        data[FRIENDS].edge_index = ei
        Ef = ei.size(1)
        # optional simple attrs if you want to use a conv that needs them later:
        data[FRIENDS].edge_attr = torch.zeros(Ef, edge_attr_dim, dtype=torch.float32)
        # dummy time earlier than any review year:
        min_year = int(data[REL].time.min().item()) if data[REL].time.numel() > 0 else 0
        data[FRIENDS].time = torch.full((Ef,), min_year - 1, dtype=torch.long)

    # ----- make the entire hetero graph undirected -----
    data = T.ToUndirected(reduce="add", merge=True)(data)

    # shape hygiene: ensure times are 1-D
    for et in data.edge_types:
        if "time" in data[et]:
            data[et].time = data[et].time.view(-1)

    # ----- cache: save -----
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "edge_attr_dim_reviews": int(edge_attr_dim),
            "settings": settings,
            "raw_fps": {
                "review": _file_fingerprint(fp_rev),
                "user": _file_fingerprint(fp_user),
                "biz": _file_fingerprint(fp_biz),
            },
            "stats": {
                "n_users": int(data[USER].num_nodes),
                "n_businesses": int(data[BUS].num_nodes),
                "n_edges_per_type": {str(et): int(data[et].edge_index.size(1)) for et in data.edge_types},
            },
        }
        torch.save({"data": data, "meta": meta}, cache_path)
        print(f"[cache] Saved preprocessed graph: {cache_path}")

    return data

# ---------- Splits ----------
def split_edge_indices(
    num_edges: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Tensor, Tensor, Tensor]:
    idx = np.arange(num_edges)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_test = int(test_ratio * num_edges)
    n_val = int(val_ratio * num_edges)
    test_idx = torch.as_tensor(idx[:n_test], dtype=torch.long)
    val_idx = torch.as_tensor(idx[n_test:n_test + n_val], dtype=torch.long)
    train_idx = torch.as_tensor(idx[n_test + n_val:], dtype=torch.long)
    return train_idx, val_idx, test_idx

def check_uniform_edge_attr_dim(data: HeteroData, rel_a, rel_b, expect_dim: Optional[int] = None):
    """
    Ensure all relations used by the model expose the same edge_attr width.
    This is required for HEATConv (single edge_dim across edge types).
    """
    dims: Dict[Tuple[str, str, str], int] = {}
    for et in (rel_a, rel_b):
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
            dims[et] = int(data[et].edge_attr.size(-1))
        else:
            dims[et] = 0
    # If friends had no attrs, create a red flag (HEATConv expects a uniform width).
    if dims[rel_b] == 0 and dims[rel_a] > 0:
        raise RuntimeError(
            f"{rel_b} has no edge_attr but {rel_a} has dim={dims[rel_a]}. "
            f"Provide zero edge_attr of the same width for {rel_b} (e.g., torch.zeros(E_f, {dims[rel_a]}))."
        )
    if expect_dim is not None and dims[rel_a] != expect_dim:
        raise RuntimeError(f"{rel_a} edge_attr dim mismatch: got {dims[rel_a]} vs expect {expect_dim}")
    if dims[rel_a] != dims[rel_b]:
        raise RuntimeError(f"edge_attr dims differ across relations: {rel_a}->{dims[rel_a]}, {rel_b}->{dims[rel_b]}")
    return dims[rel_a]

def split_edge_indices_by_year(
    data: HeteroData,
    rel: Tuple[str, str, str] = REL,
    boundary_year: int = 2021,
    include_boundary_in_train: bool = True,
    shuffle_within_splits: bool = False,
    seed: int = 42,
) -> Tuple[Tensor, Tensor]:
    """
    Temporal split of review edges by `data[rel].time` (1-D int tensor of years).
    Returns train_idx, val_idx (indices into data[rel].edge_index).
    """
    years = data[rel].time.view(-1)
    if include_boundary_in_train:
        train_mask = years <= boundary_year
        val_mask = years > boundary_year
    else:
        train_mask = years < boundary_year
        val_mask = years >= boundary_year

    train_idx = train_mask.nonzero(as_tuple=False).view(-1).to(torch.long)
    val_idx = val_mask.nonzero(as_tuple=False).view(-1).to(torch.long)

    if shuffle_within_splits:
        g = torch.Generator().manual_seed(seed)
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)]
        val_idx = val_idx[torch.randperm(val_idx.numel(), generator=g)]

    if train_idx.numel() == 0 or val_idx.numel() == 0:
        raise RuntimeError(
            f"Temporal split produced empty set(s): "
            f"train={train_idx.numel()}, val={val_idx.numel()}. "
            f"boundary_year={boundary_year}, years(min={int(years.min())}, max={int(years.max())})."
        )
    return train_idx, val_idx
