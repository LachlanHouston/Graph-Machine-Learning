import json
from pathlib import Path
from typing import List, Optional, Tuple
import hashlib
import numpy as np

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from graph_ml.utils import set_seed

import re
import gensim.downloader as api

_W2V = None
_W2V_DIM = 300

def _get_w2v():
    global _W2V
    if _W2V is None:
        # 'word2vec-google-news-300' is large (~1.5GB). Alternatives:
        #  - 'glove-wiki-gigaword-300' (300d)
        #  - 'glove-wiki-gigaword-200' (200d)
        _W2V = api.load('word2vec-google-news-300')
    return _W2V

_token_pat = re.compile(r"[A-Za-z']+")
def _simple_tokenize(s: str):
    return [t.lower() for t in _token_pat.findall(s)]

def _mean_w2v(tokens, w2v, dim: int):
    vec = np.zeros(dim, dtype=np.float32)
    count = 0
    for t in tokens:
        if t in w2v.key_to_index:
            vec += w2v.get_vector(t)
            count += 1
    if count > 0:
        vec /= count
    return vec

def _file_fingerprint(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}-{st.st_mtime_ns}"

def _parse_friends_field(s: Optional[str]) -> List[str]:
    """Yelp 'friends' is a comma-separated string or 'None'."""
    if not s or s == "None":
        return []
    # Defensive split; Yelp can have spaces after commas
    return [t.strip() for t in s.split(",") if t.strip()]

def load_yelp_as_hetero(
    data_dir: Path,
    max_reviews: Optional[int] = 100_000,
    min_review_len: int = 5,
    seed: int = 42,
    cache: bool = True,
    cache_subdir: str = "processed",
    include_user_friends: bool = True,
    max_friends_per_user: Optional[int] = 50,  # cap to avoid extreme degrees; set None for no cap
) -> HeteroData:
    """
    Builds (or loads) a cached HeteroData with:
      nodes:
        - "user"
        - "business"
      edges:
        - ("user","reviews","business"): edge_index + edge_label (stars)
        - ("business","rev_reviews","user"): reverse
        - ("user","friends","user"): optional undirected user social graph (both directions)

    Only users that appear in the review subgraph are kept for friendships.
    """
    data_dir = Path(data_dir)

    # ----- try cache -----
    if cache:
        cache_dir = data_dir / cache_subdir
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"yelp_hetero.pt"
        if cache_path.exists():
            obj = torch.load(cache_path, map_location="cpu", weights_only=False)
            data = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
            print(f"[cache] Loaded preprocessed graph: {cache_path}")
            return data
    
    fp_rev = data_dir / "yelp_academic_dataset_review.json"
    fp_user = data_dir / "yelp_academic_dataset_user.json"
    fp_biz = data_dir / "yelp_academic_dataset_business.json"
    for f in [fp_rev, fp_user, fp_biz]:
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")

        # ----- build fresh -----
    set_seed(seed)
    w2v = _get_w2v()
    w2v_dim = _W2V_DIM

    rev_rows: List[Tuple[str, str, float, int]] = []
    rev_embs: List[np.ndarray] = []  # NEW

    with fp_rev.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_reviews is not None and i >= max_reviews:
                break
            r = json.loads(line)
            txt = (r.get("text") or "")
            if len(txt) < min_review_len:
                continue
            u = r.get("user_id"); b = r.get("business_id")
            if u is None or b is None:
                continue
            stars = float(r.get("stars", 0.0))
            date_str = r.get("date") or ""
            try:
                year = int(date_str[:4]) if len(date_str) >= 4 else -1
            except Exception:
                year = -1

            rev_rows.append((u, b, stars, year))

            # --- NEW: compute mean word2vec for this review ---
            toks = _simple_tokenize(txt)
            emb = _mean_w2v(toks, w2v, w2v_dim)   # np.float32 [300]
            rev_embs.append(emb)

    if not rev_rows:
        raise RuntimeError("No reviews loaded. Increase max_reviews or lower min_review_len.")

    # Valid user/business ids (for overlap filtering)
    user_ids, biz_ids = set(), set()
    with fp_user.open("r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            uid = j.get("user_id")
            if uid:
                user_ids.add(uid)
    with fp_biz.open("r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            bid = j.get("business_id")
            if bid:
                biz_ids.add(bid)

    # Keep only overlapping reviews (and aligned embeddings)
    kept = [(u, b, s, y, e) for (u,b,s,y), e in zip(rev_rows, rev_embs)
            if (u in user_ids and b in biz_ids)]
    if not kept:
        raise RuntimeError("No overlapping user/business ids with reviews.")

    rev_rows = [(u,b,s,y) for (u,b,s,y,_) in kept]
    rev_embs = [e for (*_, e) in kept]


    # Index maps from review subgraph
    uniq_users = sorted({u for (u, _, _, _) in rev_rows})
    uniq_biz   = sorted({b for (_, b, _, _) in rev_rows})
    u2i = {u: i for i, u in enumerate(uniq_users)}
    b2i = {b: i for i, b in enumerate(uniq_biz)}

    # Build review edges
    u_idx = np.fromiter((u2i[u] for (u, _, _, _) in rev_rows), dtype=np.int64)
    b_idx = np.fromiter((b2i[b] for (_, b, _, _) in rev_rows), dtype=np.int64)
    stars = np.fromiter((s for (*_, s, _) in rev_rows), dtype=np.float32)
    years = np.fromiter((y for (*_, y) in rev_rows), dtype=np.int16)

    # Optional normalization (float32 column)
    y_min, y_max = int(years.min()), int(years.max())
    if y_max > y_min:
        years_norm = (years.astype(np.float32) - y_min) / (y_max - y_min)
    else:
        years_norm = np.zeros_like(years, dtype=np.float32)

    data = HeteroData()
    data["user"].num_nodes = len(uniq_users)
    data["business"].num_nodes = len(uniq_biz)

    rel_reviews = ("user", "reviews", "business")
    edge_index_reviews = torch.tensor(np.vstack([u_idx, b_idx]), dtype=torch.long)
    data[rel_reviews].edge_index = edge_index_reviews
    data[rel_reviews].edge_label = torch.tensor(stars, dtype=torch.long)
    data[rel_reviews].time = torch.tensor(years, dtype=torch.long).view(-1)

    # --- NEW: word2vec edge features [E, 300] ---
    edge_attr_w2v = torch.tensor(np.stack(rev_embs, axis=0), dtype=torch.float32)
    data[rel_reviews].edge_attr = edge_attr_w2v

    # reverse edge mirrors features/time
    rel_rev = ("business", "rev_reviews", "user")
    data[rel_rev].edge_index = edge_index_reviews.flip(0)
    data[rel_rev].time = data[rel_reviews].time
    data[rel_rev].edge_attr = data[rel_reviews].edge_attr

    if include_user_friends:
        # Collect friendships only among users that appear in the review subgraph
        friend_pairs: List[Tuple[int, int]] = []
        dropped_self = 0
        with fp_user.open("r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                uid = j.get("user_id")
                if uid not in u2i:
                    continue 
                u_idx_global = u2i[uid]
                friends_list = _parse_friends_field(j.get("friends", "None"))

                if max_friends_per_user is not None and len(friends_list) > max_friends_per_user:
                    friends_list = friends_list[:max_friends_per_user]

                for fid in friends_list:
                    if fid == uid:
                        dropped_self += 1
                        continue
                    v = u2i.get(fid)
                    if v is None:
                        continue  # friend not in active user set
                    a, b = (u_idx_global, v) if u_idx_global <= v else (v, u_idx_global)
                    friend_pairs.append((a, b))

        if friend_pairs:
            # de-duplicate
            friend_pairs = sorted(set(friend_pairs))
            # make undirected by adding both directions
            undirected = np.array(friend_pairs, dtype=np.int64)
            rev = undirected[:, ::-1]
            both = np.vstack([undirected, rev])
            uu_edge_index = torch.tensor(both.T, dtype=torch.long)
        else:
            uu_edge_index = torch.empty((2, 0), dtype=torch.long)

        data[("user", "friends", "user")].edge_index = uu_edge_index
        print(f"Friend graph: {uu_edge_index.size(1)} directed edges "
              f"({len(friend_pairs)} undirected pairs). Dropped self-links: {dropped_self}")
        
        # Add time dummy feature for friends edges
        E_f = data[("user","friends","user")].edge_index.size(1)
        data[("user", "friends", "user")].time = torch.full(
            (E_f,), float(y_min) - 1.0, dtype=torch.long
        )

        # Ensure that time is a 1D vector
        for et in [("user","reviews","business"), ("business","rev_reviews","user")]:
            if "time" in data[et]:
                t = data[et].time
                data[et].time = t.view(-1)

    if cache:
        cache_dir = data_dir / cache_subdir
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"yelp_hetero.pt"
        meta = {
            "max_reviews": max_reviews,
            "min_review_len": min_review_len,
            "seed": seed,
            "include_user_friends": include_user_friends,
            "max_friends_per_user": max_friends_per_user,
            "raw_fps": {
                "review": _file_fingerprint(fp_rev),
                "user": _file_fingerprint(fp_user),
                "biz": _file_fingerprint(fp_biz),
            },
            "schema": "user-reviews-business+friends@v2",
            "stats": {
                "n_users": int(data["user"].num_nodes),
                "n_businesses": int(data["business"].num_nodes),
                "n_reviews": int(edge_index_reviews.size(1)),
                "n_friend_edges_directed": int(
                    data.get(("user","friends","user"), {}).get("edge_index", torch.empty(2,0)).size(1)
                ),
            },
        }
        torch.save({"data": data, "meta": meta}, cache_path)
        print(f"[cache] Saved preprocessed graph: {cache_path}")

    return data

def split_edge_indices(
    num_edges: int, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42
) -> Tuple[Tensor, Tensor, Tensor]:
    idx = np.arange(num_edges)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_test = int(test_ratio * num_edges)
    n_val  = int(val_ratio  * num_edges)
    test_idx = torch.as_tensor(idx[:n_test], dtype=torch.long)
    val_idx  = torch.as_tensor(idx[n_test:n_test+n_val], dtype=torch.long)
    train_idx= torch.as_tensor(idx[n_test+n_val:], dtype=torch.long)
    return train_idx, val_idx, test_idx

def split_edge_indices_by_year(
    data,
    rel: Tuple[str, str, str] = ("user","reviews","business"),
    boundary_year: int = 2021,
    include_boundary_in_train: bool = True,
    shuffle_within_splits: bool = False,
    seed: int = 42,
) -> Tuple[Tensor, Tensor]:
    """
    Split edges temporally using the year stored in `data[rel].time`:
      - Train: years <= boundary_year   (if include_boundary_in_train=True)
               or years <  boundary_year (if False)
      - Val:   the complement (strictly after the boundary)

    Returns:
        train_idx, val_idx  (both 1-D Long tensors)
    """
    years = data[rel].time.view(-1)  # ensure 1-D
    if include_boundary_in_train:
        train_mask = years <= boundary_year
        val_mask   = years >  boundary_year
    else:
        train_mask = years <  boundary_year
        val_mask   = years >= boundary_year

    train_idx = train_mask.nonzero(as_tuple=False).view(-1).to(torch.long)
    val_idx   = val_mask.nonzero(as_tuple=False).view(-1).to(torch.long)

    if shuffle_within_splits:
        g = torch.Generator()
        g.manual_seed(seed)
        perm_tr = torch.randperm(train_idx.numel(), generator=g)
        perm_va = torch.randperm(val_idx.numel(), generator=g)
        train_idx = train_idx[perm_tr]
        val_idx   = val_idx[perm_va]

    if train_idx.numel() == 0 or val_idx.numel() == 0:
        raise RuntimeError(
            f"Temporal split produced empty set(s): "
            f"train={train_idx.numel()}, val={val_idx.numel()}. "
            f"Check boundary_year={boundary_year} and your data years "
            f"(min={int(years.min())}, max={int(years.max())})."
        )
    return train_idx, val_idx