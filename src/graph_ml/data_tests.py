import numpy as np
import torch
from collections import defaultdict
from warnings import filterwarnings
filterwarnings("ignore")

from sklearn.metrics import f1_score

def _get_cat_to_col_mapper(graph):
    """
    Map a category NODE id -> column index in graph.y (multi-label matrix).
    Try common conventions; fall back to identity if no mapping is provided.
    """
    # Common: a dict mapping node-id -> col
    for name in ["category_id_to_col", "cat2col", "label_map", "category_map"]:
        if hasattr(graph, name):
            m = getattr(graph, name)
            if isinstance(m, dict):
                return lambda cid: m.get(int(cid), None)
            if callable(m):
                return m

    # Fallback: assume category node ids already match y columns
    C = graph.y.shape[1]
    return lambda cid: int(cid) if 0 <= int(cid) < C else None

def _build_business_to_categories(graph,
                                  biz_type="business",
                                  cat_type="category",
                                  rel_fwd="has_category",
                                  rel_rev="rev_has_category"):
    """
    Build a dict: business_id -> set(category_node_ids) using either direction that exists.
    Works with the pyHGT-style nested dict: edge_list[target_type][source_type][relation][target_id] -> {source_id: time}
    """
    b2c = defaultdict(set)

    # Preferred (cheap): category -> business reverse relation, indexed by business target id
    rev = (graph.edge_list.get(biz_type, {})
                         .get(cat_type, {})
                         .get(rel_rev, None))
    if rev is not None:
        for b_id, src_dict in rev.items():
            b2c[int(b_id)].update(int(c_id) for c_id in src_dict.keys())

    # Fallback: business -> category forward relation, indexed by category target id
    fwd = (graph.edge_list.get(cat_type, {})
                         .get(biz_type, {})
                         .get(rel_fwd, None))
    if fwd is not None:
        # fwd[cat_id] -> {biz_id: time}  (need to invert to biz -> cat)
        for c_id, src_dict in fwd.items():
            for b_id in src_dict.keys():
                b2c[int(b_id)].add(int(c_id))

    return b2c

def _eval_split(mask, b2c, cat2col, graph):
    idxs = np.asarray(np.where(mask)[0], dtype=int)
    if idxs.size == 0:
        return float("nan"), float("nan")

    Y_true = np.asarray(graph.y[idxs], dtype=np.int32)   # [N, C]
    Y_pred = np.zeros_like(Y_true, dtype=np.int32)

    for i, b in enumerate(idxs):
        cats = b2c.get(int(b), ())
        for cid in cats:
            col = cat2col(cid)
            if col is not None and 0 <= col < Y_pred.shape[1]:
                Y_pred[i, col] = 1

    # Exact 1-hop “read from edge” prediction
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    return micro, macro

def one_hop_leak_test(graph,
                      biz_type="business",
                      cat_type="category",
                      rel_fwd="has_category",
                      rel_rev="rev_has_category"):
    """
    Prints micro/macro F1 for a 1-hop baseline that just reads Business–Category edges.
    HIGH scores here => your graph still exposes label edges (leakage).
    """
    # Build mapping & neighbors from the *raw* full graph
    cat2col = _get_cat_to_col_mapper(graph)
    b2c = _build_business_to_categories(graph, biz_type, cat_type, rel_fwd, rel_rev)

    # Pick masks (support either flat masks or dict-style splits)
    train_mask = getattr(graph, "train_mask", None)
    valid_mask = getattr(graph, "valid_mask", None)
    test_mask  = getattr(graph, "test_mask",  None)
    assert train_mask is not None and valid_mask is not None and test_mask is not None, \
        "Need train/valid/test masks on the business nodes."

    tr_micro, tr_macro = _eval_split(train_mask, b2c, cat2col, graph)
    va_micro, va_macro = _eval_split(valid_mask, b2c, cat2col, graph)
    te_micro, te_macro = _eval_split(test_mask,  b2c, cat2col, graph)

    print("[1-hop leak test] (reads has_category / rev_has_category edges directly)")
    print(f"  Train: micro-F1={tr_micro:.4f}  macro-F1={tr_macro:.4f}")
    print(f"  Valid: micro-F1={va_micro:.4f}  macro-F1={va_macro:.4f}")
    print(f"   Test: micro-F1={te_micro:.4f}  macro-F1={te_macro:.4f}")

def one_hop_on_batch(edge_index, edge_type, graph,
                     rel_fwd='has_category', rel_rev='rev_has_category'):
    # Map rel names -> ids
    rel2id = {}
    rel2id = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    rel2id['self'] = len(rel2id)

    # If either forbidden rel ID appears in edge_type, it's leaking
    bad_ids = [rel2id[r] for r in (rel_fwd, rel_rev) if r in rel2id]
    if bad_ids:
        bad_ids = torch.tensor(bad_ids, device=edge_type.device)
        if (edge_type[...,None] == bad_ids).any().item():
            return 1.0  # clear leak

    # Otherwise, there is nothing to “read” 1-hop => predict nothing
    return 0.0