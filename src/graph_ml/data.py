from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from tqdm import tqdm

from graph_ml.utils import set_seed

USER = "user"
BUS = "business"

REL = (USER, "reviews", BUS)
FRIENDS = (USER, "friends", USER)

_SENT = None
_W2V = None
_W2V_DIM_DEFAULT = 300
_TOKEN_PAT = re.compile(r"[A-Za-z']+")


# -----------------------------
# Reservoir sampling utilities
# -----------------------------
def reservoir_sample_jsonl(
    path: Path | str,
    k: int,
    *,
    seed: int = 42,
    max_lines: Optional[int] = None,
    desc: str = "Reservoir sampling",
) -> List[Dict[str, Any]]:
    """
    Unbiased reservoir sampling of k JSON objects from a JSONL file (one JSON per line).
    One pass, O(k) memory. If max_lines is provided, sampling is unbiased within that prefix.
    """
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []

    path = str(path)
    with open(path, "r", encoding="utf-8") as fh:
        it = tqdm(enumerate(fh), desc=desc, unit="lines", total=max_lines, dynamic_ncols=True)
        for i, line in it:
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if len(reservoir) < k:
                reservoir.append(obj)
            else:
                t = i + 1  # number seen so far
                j = rng.randrange(t)
                if j < k:
                    reservoir[j] = obj

            if (i % 50_000) == 0 and i > 0:
                it.set_postfix(seen=i + 1, kept=len(reservoir))

    return reservoir


# -----------------------------
# Sentiment + W2V utilities
# -----------------------------
def _get_sentiment_analyzer():
    global _SENT
    if _SENT is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            _SENT = SentimentIntensityAnalyzer()
        except Exception:
            return None
    return _SENT


def _vader_scores(text: str, *, mode: str, analyser) -> np.ndarray:
    s = analyser.polarity_scores(text or "")
    if mode == "compound":
        return np.array([s["compound"]], dtype=np.float32)
    return np.array([s["neg"], s["neu"], s["pos"], s["compound"]], dtype=np.float32)


def _get_w2v(model_name: str):
    global _W2V
    if _W2V is None:
        import gensim.downloader as api

        _W2V = api.load(model_name)
    return _W2V


def _simple_tokenize(s: str) -> List[str]:
    return [t.lower() for t in _TOKEN_PAT.findall(s or "")]


def _mean_w2v(tokens: Sequence[str], w2v, dim: int) -> np.ndarray:
    known = [t for t in tokens if t in w2v.key_to_index]
    if not known:
        return np.zeros(dim, dtype=np.float32)
    vec = np.zeros(dim, dtype=np.float32)
    for t in known:
        vec += w2v.get_vector(t)
    vec /= float(len(known))
    return vec


# -----------------------------
# Caching utilities
# -----------------------------
def _file_fingerprint(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}-{st.st_mtime_ns}"


def _parse_friends_field(s: Optional[str]) -> List[str]:
    if not s or s == "None":
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


# -----------------------------
# "Raw" node attribute handling
# -----------------------------
def _is_number(x: Any) -> bool:
    # bool is a subclass of int, so exclude it here
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def extract_numeric_schema(
    records: List[Dict[str, Any]],
    *,
    ignore_keys: Optional[set[str]] = None,
) -> List[str]:
    """
    Return a stable, sorted list of keys whose values are numeric/bool (or None)
    across a set of raw JSON dicts.
    """
    ignore_keys = ignore_keys or set()
    keys: set[str] = set()
    for r in records:
        for k, v in r.items():
            if k in ignore_keys:
                continue
            if v is None or _is_number(v) or _is_bool(v):
                keys.add(k)
    return sorted(keys)


def vectorize_numeric(r: Dict[str, Any], keys: List[str]) -> np.ndarray:
    """
    Convert a raw JSON dict into a float32 vector for the given numeric schema.
    Missing/None/non-numeric values are filled with 0.0.
    """
    out = np.zeros(len(keys), dtype=np.float32)
    for i, k in enumerate(keys):
        v = r.get(k, 0)
        if v is None:
            out[i] = 0.0
        elif _is_bool(v):
            out[i] = 1.0 if v else 0.0
        elif _is_number(v):
            out[i] = float(v)
        else:
            out[i] = 0.0
    return out


# -----------------------------
# Main loader
# -----------------------------
def load_yelp_as_hetero(
    data_dir: Path | str,
    *,
    # Reviews
    max_reviews: Optional[int] = 100_000,  # if None: stream all (no reservoir)
    sample_method: str = "reservoir",  # "reservoir" or "prefix"
    reservoir_max_lines: Optional[int] = None,  # optional cap on lines considered for reservoir/prefix
    # Filtering (token-based)
    min_review_len: int = 5,  # MIN TOKENS (not chars)
    coverage_threshold: float = 0.2,  # only used if use_text_edge_attr=True
    # General
    seed: int = 42,
    cache: bool = True,
    cache_subdir: str = "processed",
    # Friends
    include_user_friends: bool = True,
    max_friends_per_user: Optional[int] = 50,
    # Edge attrs
    use_text_edge_attr: bool = True,
    w2v_model_name: str = "word2vec-google-news-300",
    w2v_dim: Optional[int] = None,
    use_sentiment_edge_attr: bool = True,
    sentiment_mode: str = "vader4",  # "vader4" or "compound"
    zscore_sentiment: bool = False,
    # Raw node attributes behavior
    store_raw_node_dicts: bool = False,
) -> HeteroData:
    """
    Loads Yelp into a hetero graph with:
      - user and business nodes
      - review edges user->business with edge_attr [W2V || sentiment || zscored_year]
      - optional friend edges user->user with zero edge_attr (matched width) + constant time

    Node attributes:
      - data["user"].x / data["business"].x contain *all numeric/bool raw JSON fields* (no transforms),
        based on a schema inferred from the kept nodes.
      - (optional) data["user"].raw / data["business"].raw store the full raw JSON dict per node index.
        NOTE: this can be memory-heavy; disable via store_raw_node_dicts=False if needed.
    """
    set_seed(seed)
    data_dir = Path(data_dir)

    cache_path = data_dir / cache_subdir / "yelp_hetero.pt"
    settings = {
        "max_reviews": max_reviews,
        "sample_method": sample_method,
        "reservoir_max_lines": reservoir_max_lines,
        "min_review_len": min_review_len,
        "coverage_threshold": coverage_threshold,
        "include_user_friends": include_user_friends,
        "max_friends_per_user": max_friends_per_user,
        "use_text_edge_attr": use_text_edge_attr,
        "w2v_model_name": w2v_model_name,
        "w2v_dim": w2v_dim,
        "use_sentiment_edge_attr": use_sentiment_edge_attr,
        "sentiment_mode": sentiment_mode,
        "zscore_sentiment": zscore_sentiment,
        "store_raw_node_dicts": store_raw_node_dicts,
    }

    if cache and cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "data" in obj and "meta" in obj:
            print("Cached data loaded.")
            print(f"  - Settings: {obj['meta'].get('settings')}")
            print(f"  - Data keys: {list(obj['data'].keys())}")
            print(f"  - Stats: {obj['meta'].get('stats')}")
            if obj["meta"].get("settings") == settings:
                return obj["data"]

    fp_rev = data_dir / "yelp_academic_dataset_review.json"
    fp_user = data_dir / "yelp_academic_dataset_user.json"
    fp_biz = data_dir / "yelp_academic_dataset_business.json"
    for f in (fp_rev, fp_user, fp_biz):
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")

    w2v = _get_w2v(w2v_model_name) if use_text_edge_attr else None
    w2v_dim_eff = int(w2v_dim or _W2V_DIM_DEFAULT) if use_text_edge_attr else 0

    analyser = _get_sentiment_analyzer() if use_sentiment_edge_attr else None
    if use_sentiment_edge_attr and analyser is None:
        raise RuntimeError("Sentiment requested but VADER sentiment analyzer is unavailable.")

    # ---- load reviews (prefix or reservoir) ----
    if max_reviews is None:
        review_iter: List[Dict[str, Any]] | None = None  # stream
    else:
        if sample_method not in ("reservoir", "prefix"):
            raise ValueError("sample_method must be 'reservoir' or 'prefix'")
        if sample_method == "reservoir":
            review_iter = reservoir_sample_jsonl(
                fp_rev,
                k=max_reviews,
                seed=seed,
                max_lines=reservoir_max_lines,
                desc="Reservoir sampling reviews",
            )
        else:
            review_iter = []
            with fp_rev.open("r", encoding="utf-8") as fh:
                it = tqdm(
                    enumerate(fh),
                    desc="Reading review prefix",
                    unit="lines",
                    total=(reservoir_max_lines or max_reviews),
                    dynamic_ncols=True,
                )
                for i, line in it:
                    if reservoir_max_lines is not None and i >= reservoir_max_lines:
                        break
                    if i >= max_reviews:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    review_iter.append(json.loads(line))

    # ---- parse reviews into edges + edge features ----
    rev_rows: List[Tuple[str, str, float, int]] = []
    rev_embs: List[np.ndarray] = []
    rev_sents: List[np.ndarray] = []

    n_drop_len = 0
    n_drop_cov = 0

    def _handle_review(r: Dict[str, Any]):
        nonlocal n_drop_len, n_drop_cov
        txt = r.get("text") or ""
        tokens = _simple_tokenize(txt)

        if len(tokens) < min_review_len:
            n_drop_len += 1
            return

        if use_text_edge_attr:
            known = [t for t in tokens if t in w2v.key_to_index]  # type: ignore[union-attr]
            coverage = len(known) / max(1, len(tokens))
            if coverage < coverage_threshold:
                n_drop_cov += 1
                return

        uid = r.get("user_id")
        bid = r.get("business_id")
        if not uid or not bid:
            return

        stars = float(r.get("stars", 0.0))
        date_str = r.get("date") or ""
        year = int(date_str[:4]) if len(date_str) >= 4 and date_str[:4].isdigit() else -1

        rev_rows.append((uid, bid, stars, year))

        if use_text_edge_attr:
            rev_embs.append(_mean_w2v(tokens, w2v, w2v_dim_eff))  # type: ignore[arg-type]
        if use_sentiment_edge_attr:
            rev_sents.append(_vader_scores(txt, mode=sentiment_mode, analyser=analyser))

    if review_iter is None:
        with fp_rev.open("r", encoding="utf-8") as fh:
            it = tqdm(
                enumerate(fh),
                desc="Parsing reviews",
                unit="lines",
                total=reservoir_max_lines,
                dynamic_ncols=True,
            )
            for i, line in it:
                if reservoir_max_lines is not None and i >= reservoir_max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                _handle_review(r)
                if max_reviews is not None and len(rev_rows) >= max_reviews:
                    break
                if (i % 2000) == 0 and i > 0:
                    it.set_postfix(kept=len(rev_rows), drop_len=n_drop_len, drop_cov=n_drop_cov)
    else:
        for r in tqdm(review_iter, desc="Parsing sampled reviews", unit="rev", dynamic_ncols=True):
            _handle_review(r)

    if not rev_rows:
        raise RuntimeError("No reviews loaded after filtering. Lower thresholds or increase max_reviews.")

    # ---- load users/businesses sets + full raw json dicts ----
    user_ids: set[str] = set()
    biz_ids: set[str] = set()
    user_json: Dict[str, Dict[str, Any]] = {}
    biz_json: Dict[str, Dict[str, Any]] = {}

    with fp_user.open("r", encoding="utf-8") as fh:
        for line in fh:
            j = json.loads(line)
            uid = j.get("user_id")
            if uid:
                user_ids.add(uid)
                user_json[uid] = j

    with fp_biz.open("r", encoding="utf-8") as fh:
        for line in fh:
            j = json.loads(line)
            bid = j.get("business_id")
            if bid:
                biz_ids.add(bid)
                biz_json[bid] = j

    keep_mask = [(u in user_ids and b in biz_ids) for (u, b, _, _) in rev_rows]
    if not any(keep_mask):
        raise RuntimeError("No overlapping user/business ids with reviews.")

    # filter aligned feature arrays consistently
    if use_text_edge_attr or use_sentiment_edge_attr:
        rev_rows_f: List[Tuple[str, str, float, int]] = []
        rev_embs_f: List[np.ndarray] = [] if use_text_edge_attr else []
        rev_sents_f: List[np.ndarray] = [] if use_sentiment_edge_attr else []

        for k, keep in enumerate(keep_mask):
            if not keep:
                continue
            rev_rows_f.append(rev_rows[k])
            if use_text_edge_attr:
                rev_embs_f.append(rev_embs[k])
            if use_sentiment_edge_attr:
                rev_sents_f.append(rev_sents[k])

        rev_rows = rev_rows_f
        if use_text_edge_attr:
            rev_embs = rev_embs_f
        if use_sentiment_edge_attr:
            rev_sents = rev_sents_f
    else:
        rev_rows = [row for row, keep in zip(rev_rows, keep_mask) if keep]

    uniq_users = sorted({u for (u, _, _, _) in rev_rows})
    uniq_biz = sorted({b for (_, b, _, _) in rev_rows})
    u2i = {u: i for i, u in enumerate(uniq_users)}
    b2i = {b: i for i, b in enumerate(uniq_biz)}

    u_idx = np.fromiter((u2i[u] for (u, _, _, _) in rev_rows), dtype=np.int64)
    b_idx = np.fromiter((b2i[b] for (_, b, _, _) in rev_rows), dtype=np.int64)
    stars = np.fromiter((s for (*_, s, _) in rev_rows), dtype=np.float32)
    years = np.fromiter((y for (*_, y) in rev_rows), dtype=np.int16)

    deg_user_rev = np.bincount(u_idx, minlength=len(u2i)).astype(np.int64)
    deg_biz_rev = np.bincount(b_idx, minlength=len(b2i)).astype(np.int64)

    data = HeteroData()
    data[USER].num_nodes = len(uniq_users)
    data[BUS].num_nodes = len(uniq_biz)

    # ---- NEW: Use ALL numeric/bool raw JSON fields as node features (no transforms) ----
    USER_IGNORE = {"user_id", "average_stars"}  # keep id out of numeric schema
    BIZ_IGNORE = {"business_id", "stars"}  # keep id out of numeric schema

    user_records = [user_json[u] for u in uniq_users if u in user_json]
    biz_records = [biz_json[b] for b in uniq_biz if b in biz_json]

    user_num_keys = extract_numeric_schema(user_records, ignore_keys=USER_IGNORE)
    biz_num_keys = extract_numeric_schema(biz_records, ignore_keys=BIZ_IGNORE)

    # If schema ends up empty for some reason, fall back to old 2-dim features (rare)
    if len(user_num_keys) == 0:
        user_feat = np.zeros((len(u2i), 2), dtype=np.float32)
        for uid, i in u2i.items():
            j = user_json.get(uid, {})
            user_feat[i, 0] = float(j.get("review_count", 0) or 0)
            user_feat[i, 1] = float(deg_user_rev[i])
        data[USER].x = torch.tensor(user_feat, dtype=torch.float32)
        data[USER].feat_names = ["review_count", "deg_user_rev"]
    else:
        user_x = np.stack([vectorize_numeric(user_json[u], user_num_keys) for u in uniq_users], axis=0)
        data[USER].x = torch.tensor(user_x, dtype=torch.float32)
        data[USER].feat_names = user_num_keys

    if len(biz_num_keys) == 0:
        biz_feat = np.zeros((len(b2i), 2), dtype=np.float32)
        for bid, j_idx in b2i.items():
            j = biz_json.get(bid, {})
            biz_feat[j_idx, 0] = float(j.get("review_count", 0) or 0)
            biz_feat[j_idx, 1] = float(deg_biz_rev[j_idx])
        data[BUS].x = torch.tensor(biz_feat, dtype=torch.float32)
        data[BUS].feat_names = ["review_count", "deg_biz_rev"]
    else:
        biz_x = np.stack([vectorize_numeric(biz_json[b], biz_num_keys) for b in uniq_biz], axis=0)
        data[BUS].x = torch.tensor(biz_x, dtype=torch.float32)
        data[BUS].feat_names = biz_num_keys

    # Store full raw dicts (strings/lists/dicts etc.) aligned with node indices
    if store_raw_node_dicts:
        data[USER].raw = [user_json[u] for u in uniq_users]
        data[BUS].raw = [biz_json[b] for b in uniq_biz]

    # ---- reviews edges ----
    data[REL].edge_index = torch.tensor(np.vstack([u_idx, b_idx]), dtype=torch.long)
    # Note: original code stores stars as long; keeping unchanged for compatibility with your pipeline.
    data[REL].edge_label = torch.tensor(stars, dtype=torch.long)
    data[REL].time = torch.tensor(years, dtype=torch.long).view(-1)

    # edge_attr = [W2V || SENT || time_zscore]
    time_f = data[REL].time.to(torch.float32).view(-1, 1)
    time_f = (time_f - time_f.mean()) / (time_f.std() + 1e-6)

    parts: List[torch.Tensor] = []

    if use_text_edge_attr:
        w2v_mat = torch.tensor(np.stack(rev_embs, axis=0), dtype=torch.float32)
        w2v_mat = w2v_mat / (w2v_mat.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        parts.append(w2v_mat)

    if use_sentiment_edge_attr:
        s_mat = torch.tensor(np.stack(rev_sents, axis=0), dtype=torch.float32)
        if zscore_sentiment:
            s_mat = (s_mat - s_mat.mean(dim=0, keepdim=True)) / (s_mat.std(dim=0, keepdim=True) + 1e-6)
        parts.append(s_mat)

    parts.append(time_f)
    data[REL].edge_attr = torch.cat(parts, dim=-1)
    reviews_edge_attr_dim = int(data[REL].edge_attr.size(-1))

    # ---- friends edges ----
    if include_user_friends:
        friend_pairs: List[Tuple[int, int]] = []

        # Counters
        n_users_seen = 0                  # user json lines that are in u2i
        n_total_friends_listed = 0        # total friends parsed before truncation (for kept users)
        n_truncated = 0                   # how many friend entries removed by max_friends_per_user
        n_missing_in_graph = 0            # friend ids not in u2i
        n_self_loops = 0                  # friend id == uid
        n_kept_pairs = 0                  # appended edges
        n_dup_removed = 0                 # removed by optional dedupe

        with fp_user.open("r", encoding="utf-8") as fh:
            it = tqdm(fh, desc="Parsing friends", unit="lines", dynamic_ncols=True)

            for i, line in enumerate(it):
                j = json.loads(line)
                uid = j.get("user_id")
                if uid not in u2i:
                    continue

                n_users_seen += 1
                u_local = u2i[uid]

                friends_all = _parse_friends_field(j.get("friends", "None"))
                n_total_friends_listed += len(friends_all)

                # Apply truncation and count what you threw away
                friends = friends_all
                if max_friends_per_user is not None and len(friends_all) > max_friends_per_user:
                    n_truncated += (len(friends_all) - max_friends_per_user)
                    friends = friends_all[:max_friends_per_user]

                for fid in friends:
                    v = u2i.get(fid)
                    if v is None:
                        n_missing_in_graph += 1
                        continue
                    if v == u_local:
                        n_self_loops += 1
                        continue

                    friend_pairs.append((u_local, v))
                    n_kept_pairs += 1

                # Update progress display occasionally
                if (i % 2000) == 0 and i > 0:
                    it.set_postfix(
                        users=n_users_seen,
                        listed=n_total_friends_listed,
                        cut=n_truncated,
                        miss=n_missing_in_graph,
                        self_loops=n_self_loops,
                        kept=n_kept_pairs,
                    )

        # Optional: dedupe and count duplicates removed
        if friend_pairs:
            before = len(friend_pairs)
            friend_pairs = list(set(friend_pairs))
            n_dup_removed = before - len(friend_pairs)

        if friend_pairs:
            ei = torch.tensor(friend_pairs, dtype=torch.long).t().contiguous()
        else:
            ei = torch.empty((2, 0), dtype=torch.long)

        data[FRIENDS].edge_index = ei
        ef = ei.size(1)
        data[FRIENDS].edge_attr = torch.zeros(ef, reviews_edge_attr_dim, dtype=torch.float32)
        min_year = int(data[REL].time.min().item()) if data[REL].time.numel() > 0 else 0
        data[FRIENDS].time = torch.full((ef,), min_year - 1, dtype=torch.long)

        print(
            f"[friends] users_seen={n_users_seen:,} "
            f"listed={n_total_friends_listed:,} "
            f"cut_by_max={n_truncated:,} "
            f"missing_in_u2i={n_missing_in_graph:,} "
            f"self_loops={n_self_loops:,} "
            f"kept_pairs={n_kept_pairs:,} "
            f"dup_removed={n_dup_removed:,} "
            f"final_edges={ef:,}"
        )

    # Make undirected (adds reverse edges / merges)
    data = T.ToUndirected(reduce="add", merge=True)(data)

    for et in data.edge_types:
        if "time" in data[et]:
            data[et].time = data[et].time.view(-1)

    # ---- cache ----
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "edge_attr_dim_reviews": int(reviews_edge_attr_dim),
            "settings": settings,
            "raw_fps": {
                "review": _file_fingerprint(fp_rev),
                "user": _file_fingerprint(fp_user),
                "biz": _file_fingerprint(fp_biz),
            },
            "node_feat_schema": {
                "user": list(getattr(data[USER], "feat_names", [])),
                "business": list(getattr(data[BUS], "feat_names", [])),
            },
            "stats": {
                "n_users": int(data[USER].num_nodes),
                "n_businesses": int(data[BUS].num_nodes),
                "n_edges_per_type": {str(et): int(data[et].edge_index.size(1)) for et in data.edge_types},
                "dropped_min_tokens": int(n_drop_len),
                "dropped_low_coverage": int(n_drop_cov),
            },
        }
        torch.save({"data": data, "meta": meta}, cache_path)

    return data

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
    val_idx = torch.as_tensor(idx[n_test : n_test + n_val], dtype=torch.long)
    train_idx = torch.as_tensor(idx[n_test + n_val :], dtype=torch.long)
    return train_idx, val_idx, train_idx


def check_uniform_edge_attr_dim(
    data: HeteroData,
    rel_a: Tuple[str, str, str],
    rel_b: Tuple[str, str, str],
    expect_dim: Optional[int] = None,
) -> int:
    dims: Dict[Tuple[str, str, str], int] = {}
    for et in (rel_a, rel_b):
        ea = getattr(data[et], "edge_attr", None)
        dims[et] = int(ea.size(-1)) if ea is not None else 0

    if dims[rel_b] == 0 and dims[rel_a] > 0:
        raise RuntimeError(
            f"{rel_b} has no edge_attr but {rel_a} has dim={dims[rel_a]}. "
            f"Provide zero edge_attr of the same width for {rel_b}."
        )

    if expect_dim is not None and dims[rel_a] != expect_dim:
        raise RuntimeError(f"{rel_a} edge_attr dim mismatch: got {dims[rel_a]} vs expect {expect_dim}")

    if dims[rel_a] != dims[rel_b]:
        raise RuntimeError(f"edge_attr dims differ: {rel_a}->{dims[rel_a]}, {rel_b}->{dims[rel_b]}")

    return dims[rel_a]


def split_edge_indices_by_year(
    data: HeteroData,
    rel: Tuple[str, str, str],
    boundary_year: int = 2021,
    include_boundary_in_train: bool = True,
    test_ratio_in_future: float = 0.5,  # fraction of "future" used for test; rest is val
    shuffle_within_splits: bool = False,
    seed: int = 42,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Train is all <= boundary (or < boundary).
    Future is all > boundary (or >= boundary).
    Future is then split into val/test by ratio (optionally shuffled within future).
    """
    years = data[rel].time.view(-1).to(torch.long)

    if include_boundary_in_train:
        train_mask = years <= boundary_year
        future_mask = years > boundary_year
    else:
        train_mask = years < boundary_year
        future_mask = years >= boundary_year

    train_idx = train_mask.nonzero(as_tuple=False).view(-1).to(torch.long)
    future_idx = future_mask.nonzero(as_tuple=False).view(-1).to(torch.long)

    if train_idx.numel() == 0 or future_idx.numel() == 0:
        raise RuntimeError(
            f"Temporal split produced empty set(s): train={train_idx.numel()}, future={future_idx.numel()}, "
            f"boundary_year={boundary_year}, years(min={int(years.min())}, max={int(years.max())})."
        )

    if not (0.0 < test_ratio_in_future < 1.0):
        raise ValueError("test_ratio_in_future must be between 0 and 1")

    if shuffle_within_splits:
        g = torch.Generator().manual_seed(seed)
        future_idx = future_idx[torch.randperm(future_idx.numel(), generator=g)]

    n_test = int(round(test_ratio_in_future * future_idx.numel()))
    n_test = max(1, min(n_test, future_idx.numel() - 1))  # ensure both non-empty

    test_idx = future_idx[:n_test]
    val_idx = future_idx[n_test:]

    # (optional) shuffle train too
    if shuffle_within_splits:
        g = torch.Generator().manual_seed(seed)
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)]

    return train_idx, val_idx, test_idx