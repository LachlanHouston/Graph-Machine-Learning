import math
import torch
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D

def visualize_link_batch(
    batch,
    highlight_rel=("user", "reviews", "business"),
    max_nodes=800,
    with_labels=False,
    node_size_scale=280.0,
    figsize=(12, 9),
    seed=42,
    save_path=None,
    annotate_targets=True,          # label target edges with year
    annotate_sample_k=0,            # also label up to k random edges per relation
    year_fmt=lambda y: str(int(y)), # how to format year labels
):
    """
    Draw the hetero subgraph with clear type styling + optional edge-year labels.
    Ensures target edge endpoints are always included in the visualization even when pruning.
    """
    # -------- palettes & styles --------
    node_palette = {
        "user":     "#1f77b4",
        "business": "#ff7f0e",
    }
    node_markers = {
        "user": "o",
        "business": "s",
    }
    edge_palette = {
        ("user","reviews","business"):     "#4c78a8",
        ("user","friends","user"):         "#54a24b",
    }
    edge_styles = {
        ("user","reviews","business"):     "solid",
        ("user","friends","user"):         (0, (1, 3)),
    }
    highlight_color = "#d62728"
    highlight_width = 3.2

    # -------- build graph --------
    G = nx.MultiDiGraph()
    node_id_maps = {}
    deg = defaultdict(int)

    def _hide_et(et):
        # et is a 3-tuple: (src_type, rel_name, dst_type)
        return str(et[1]).startswith("rev_")

    visible_edge_types = [et for et in batch.edge_types if not _hide_et(et)]

    total_nodes = sum(int(getattr(batch[nt], "num_nodes", 0)) for nt in batch.node_types)
    take_ratio = 1.0 if total_nodes <= max_nodes else max_nodes / float(total_nodes)

    # ---- must-keep: endpoints of target edges (so targets always render) ----
    must_keep = {nt: set() for nt in batch.node_types}
    if highlight_rel in batch.edge_types and hasattr(batch[highlight_rel], "edge_label_index"):
        el = batch[highlight_rel].edge_label_index  # [2, M]
        s_type, _, d_type = highlight_rel
        if el is not None and el.numel() > 0:
            for s in el[0].tolist():
                must_keep[s_type].add(int(s))
            for d in el[1].tolist():
                must_keep[d_type].add(int(d))

    # ---- add nodes (ensure must-keep are included even under pruning) ----
    for ntype in batch.node_types:
        num = int(getattr(batch[ntype], "num_nodes", 0))
        if num == 0:
            continue
        all_ids = torch.arange(num)
        keep_ids = set(must_keep.get(ntype, set()))

        if take_ratio < 1.0:
            k = max(len(keep_ids), int(math.ceil(num * take_ratio)))
            remaining = [i for i in all_ids.tolist() if i not in keep_ids]
            need = k - len(keep_ids)
            if need > 0 and remaining:
                add = torch.tensor(remaining)[torch.randperm(len(remaining))[:need]].tolist()
            else:
                add = []
            chosen = list(keep_ids) + add
            ids = torch.tensor(sorted(set(chosen)), dtype=torch.long)
        else:
            ids = all_ids

        node_id_maps[ntype] = {int(i): f"{ntype[:3]}:{int(i)}" for i in ids.tolist()}
        for i in ids.tolist():
            G.add_node((ntype, int(i)), ntype=ntype)

    # ---- add edges ----
    # keep a per-relation list aligning edges to the source .time tensor indices
    rel_edges = {et: [] for et in visible_edge_types}
    for et in visible_edge_types:
        ei = batch[et].edge_index
        if ei.numel() == 0:
            continue
        src_type, _, dst_type = et
        allowed_src = node_id_maps.get(src_type, {})
        allowed_dst = node_id_maps.get(dst_type, {})
        s_list, d_list = ei[0].tolist(), ei[1].tolist()
        for idx, (s, d) in enumerate(zip(s_list, d_list)):
            if s in allowed_src and d in allowed_dst:
                u = (src_type, s); v = (dst_type, d)
                G.add_edge(u, v, etype=et, eidx=idx)
                rel_edges[et].append(((u, v), idx))
                deg[u] += 1; deg[v] += 1

    # ---- node sizes & layout ----
    sizes = [max(60.0, math.sqrt(max(1, deg[n])) * node_size_scale) for n in G.nodes]
    k = None if len(G) < 200 else 1 / math.sqrt(len(G))
    pos = nx.spring_layout(G, seed=seed, k=k)

    plt.figure(figsize=figsize)

    # ---- draw nodes per type ----
    for ntype in batch.node_types:
        nodelist = [n for n in G.nodes if n[0] == ntype]
        if not nodelist:
            continue
        idxs = [list(G.nodes).index(n) for n in nodelist]
        nsizes = [sizes[i] for i in idxs]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodelist,
            node_size=nsizes,
            node_color=node_palette.get(ntype, "#7f7f7f"),
            node_shape=node_markers.get(ntype, "o"),
            alpha=0.9,
            linewidths=0.8,
            edgecolors="#ffffff",
        )

    # ---- draw edges per relation ----
    for et in visible_edge_types:
        edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("etype") == et]
        if not edges:
            continue
        nx.draw_networkx_edges(
            G, pos, edgelist=edges,
            width=1.1, alpha=0.5,
            edge_color=edge_palette.get(et, "#999999"),
            style=edge_styles.get(et, "solid"),
            arrows=False,
        )

    # ---- optional sampling annotations (random edges per relation) ----
    if annotate_sample_k and annotate_sample_k > 0:
        for et in visible_edge_types:
            if not hasattr(batch[et], "time") or batch[et].time is None:
                continue
            times = batch[et].time
            if times.dim() > 1:
                times = times.view(-1)
            pairs = rel_edges.get(et, [])
            if not pairs:
                continue
            k_samp = min(annotate_sample_k, len(pairs))
            perm = torch.randperm(len(pairs))[:k_samp].tolist()
            for j in perm:
                (u, v), eidx = pairs[j]
                t = times[eidx].item()
                mid = (pos[u] + pos[v]) / 2.0
                plt.text(
                    mid[0], mid[1], year_fmt(t),
                    fontsize=8,
                    ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.4),
                    color=edge_palette.get(et, "#333333"),
                )

    # ---- highlighted target edges (always drawn now because endpoints are forced in) ----
    if highlight_rel in batch.edge_types and hasattr(batch[highlight_rel], "edge_label_index"):
        el = batch[highlight_rel].edge_label_index  # [2, M]
        if el is not None and el.numel() > 0:
            src_type, _, dst_type = highlight_rel
            allowed_src = node_id_maps.get(src_type, {})
            allowed_dst = node_id_maps.get(dst_type, {})
            hl_edges = []
            for s, d in zip(el[0].tolist(), el[1].tolist()):
                if (s in allowed_src) and (d in allowed_dst):
                    hl_edges.append(((src_type, s), (dst_type, d)))

            if hl_edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=hl_edges, width=highlight_width, alpha=0.95,
                    edge_color=highlight_color, arrows=False
                )

                if annotate_targets:
                    tvec = getattr(batch[highlight_rel], "edge_label_time", None)
                    if tvec is not None:
                        tvec = tvec.view(-1) if tvec.dim() > 1 else tvec
                        L = min(len(hl_edges), tvec.numel())
                        for ((u, v), t) in zip(hl_edges[:L], tvec[:L].tolist()):
                            mid = (pos[u] + pos[v]) / 2.0
                            plt.text(
                                mid[0], mid[1], year_fmt(t),
                                fontsize=9, fontweight="bold",
                                ha="center", va="center",
                                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.5),
                                color=highlight_color,
                            )

    # ---- labels (optional) ----
    if with_labels and len(G) <= 200:
        labels = {n: f"{n[0][:3]}:{n[1]}" for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, alpha=0.9)

    # ---- legend ----
    node_handles = [
        Line2D([0], [0], marker=marker, color="none",
               markerfacecolor=node_palette.get(nt, "#7f7f7f"),
               markeredgecolor="#ffffff", markeredgewidth=0.8,
               markersize=10, label=nt)
        for nt, marker in node_markers.items()
        if any(n[0] == nt for n in G.nodes)
    ]
    edge_handles = [
        Line2D([0], [0], color=edge_palette.get(et, "#999999"),
               linestyle=edge_styles.get(et, "solid"), linewidth=2, label=str(et))
        for et in visible_edge_types
        if any(d.get("etype")==et for *_, d in G.edges(data=True))
    ]
    handles = node_handles + edge_handles + [Line2D([0],[0], color=highlight_color, linewidth=3, label="target edges")]
    if handles:
        plt.legend(handles=handles, fontsize=9, frameon=False, loc="upper right")

    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    from torch_geometric.loader import LinkNeighborLoader
    from graph_ml.data import load_yelp_as_hetero, split_edge_indices_by_year

    rel = ("user", "reviews", "business")

    data = load_yelp_as_hetero('data/raw/',
                               max_reviews=250_000,
                               min_review_len=0, seed=42,
                               cache=True, cache_subdir="processed",
                               include_user_friends=True, max_friends_per_user=50)

    print(data.metadata())

    train_idx, val_idx = split_edge_indices_by_year(
        data, rel=rel, boundary_year=2017, include_boundary_in_train=True
    )

    num_neighbors = {
        ("user","reviews","business"):   [0, 0, 0, 0],
        ("business","rev_reviews","user"): [0, 0, 0, 0],
        ("user","friends","user"):       [0, 0, 0, 0],
    }

    stars = data[rel].edge_label
    years = data[rel].time.squeeze()

    train_label = stars[train_idx]
    val_label   = stars[val_idx]

    batch_size = 10

    common_kwargs = dict(
        data=data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )

    train_loader = LinkNeighborLoader(
        **common_kwargs,
        edge_label_index=(rel, data[rel].edge_index[:, train_idx]),
        edge_label=train_label,
        edge_label_time=years[train_idx],
        time_attr="time",
    )

    val_loader = LinkNeighborLoader(
        **{**common_kwargs, "shuffle": False},
        edge_label_index=(rel, data[rel].edge_index[:, val_idx]),
        edge_label=val_label,
        edge_label_time=years[val_idx],
        time_attr="time",
    )

    for batch in train_loader:
        visualize_link_batch(
            batch.cpu(),
            highlight_rel=("user","reviews","business"),
            annotate_targets=True,
            annotate_sample_k=20,
            year_fmt=lambda y: str(int(y)),
            save_path=f"reports/figures/yelp_train_batch_size_{batch_size}.png"
        )
        break

    for batch in val_loader:
        visualize_link_batch(
            batch.cpu(),
            highlight_rel=("user","reviews","business"),
            annotate_targets=True,
            annotate_sample_k=20,
            year_fmt=lambda y: str(int(y)),
            save_path=f"reports/figures/yelp_val_batch_size_{batch_size}.png"
        )
        break