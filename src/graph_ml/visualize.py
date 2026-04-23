import math
import numpy as np
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
    annotate_sample_k=0,
    year_fmt=lambda y: str(int(y)),
):
    """
    Draw the hetero subgraph with clear type styling + optional edge-year labels.
    Ensures target edge endpoints are always included in the visualization even when pruning.
    """
    # Styles
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
        ("user","friends","user"):         (1, (1, 2)),
    }
    highlight_color = "#d62728"
    highlight_width = 3.2

    # Graph building
    G = nx.MultiDiGraph()
    node_id_maps = {}
    deg = defaultdict(int)

    def _hide_et(et):
        # et is a 3-tuple: (src_type, rel_name, dst_type)
        return str(et[1]).startswith("rev_")

    visible_edge_types = [et for et in batch.edge_types if not _hide_et(et)]

    total_nodes = sum(int(getattr(batch[nt], "num_nodes", 0)) for nt in batch.node_types)
    take_ratio = 1.0 if total_nodes <= max_nodes else max_nodes / float(total_nodes)

    # Target edge strategy
    must_keep = {nt: set() for nt in batch.node_types}
    if highlight_rel in batch.edge_types and hasattr(batch[highlight_rel], "edge_label_index"):
        el = batch[highlight_rel].edge_label_index  # [2, M]
        s_type, _, d_type = highlight_rel
        if el is not None and el.numel() > 0:
            for s in el[0].tolist():
                must_keep[s_type].add(int(s))
            for d in el[1].tolist():
                must_keep[d_type].add(int(d))

    # Add nodes based on given take ratio
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

    # Add edges
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

    # Node size customization
    sizes = [max(120.0, math.sqrt(max(1, deg[n])) * node_size_scale) for n in G.nodes]
    k = None if len(G) < 200 else 1 / math.sqrt(len(G))
    pos = nx.spring_layout(G, seed=seed, k=0.9, iterations=100)

    plt.figure(figsize=figsize)

    # Draw nodes
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

    # Draw edges
    for et in visible_edge_types:
        edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("etype") == et]
        if not edges:
            continue
        nx.draw_networkx_edges(
            G, pos, edgelist=edges,
            width=3.1, alpha=0.75,
            edge_color=edge_palette.get(et, "#999999"),
            style=edge_styles.get(et, "solid"),
            arrows=False,
        )

    # Sampling annotations
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
            if et != ('user', 'friends', 'user'):
                for j in perm:
                    (u, v), eidx = pairs[j]
                    t = times[eidx].item()
                    mid = (pos[u] + pos[v]) / 2.0
                    plt.text(
                        mid[0], mid[1], year_fmt(t),
                        fontsize=10,
                        ha="center", va="center",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.4),
                        color=edge_palette.get(et, "#333333"),
                    )

    # Highlight target edges
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
                    el = getattr(batch[highlight_rel], "edge_label_index", None)

                    if tvec is not None and el is not None and el.numel() > 0:
                        tvec = tvec.view(-1)
                        M = min(el.size(1), tvec.numel())

                        src_type, _, dst_type = highlight_rel
                        allowed_src = node_id_maps.get(src_type, {})
                        allowed_dst = node_id_maps.get(dst_type, {})

                        for j in range(M):
                            s = int(el[0, j].item())
                            d = int(el[1, j].item())
                            if (s not in allowed_src) or (d not in allowed_dst):
                                continue

                            u = (src_type, s)
                            v = (dst_type, d)
                            mid = (pos[u] + pos[v]) / 2.0
                            t = tvec[j].item()

                            plt.text(
                                mid[0], mid[1], year_fmt(t),
                                fontsize=14, fontweight="bold",
                                ha="center", va="center",
                                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.5),
                                color=highlight_color,
                            )

    # Labels
    if with_labels and len(G) <= 200:
        labels = {n: f"{n[0][:3]}:{n[1]}" for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, alpha=0.9)

    # Legend
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
               linestyle=edge_styles.get(et, "solid"), linewidth=5, label=str(et))
        for et in visible_edge_types
        if any(d.get("etype")==et for *_, d in G.edges(data=True))
    ]
    handles = node_handles + edge_handles + [Line2D([0],[0], color=highlight_color, linewidth=5, label="target edges")]
    if handles:
        plt.legend(handles=handles, fontsize=14, frameon=True, loc="upper right")

    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

def visualize_star_shift(
    data,
    rel=("user", "reviews", "business"),
    *,
    year_stride: int = 5,
    min_count_per_year: int = 200,
    normalize: str = "fraction",   # "count" | "fraction"
    show_overall: bool = True,
    overall_alpha: float = 0.25,
    figsize: tuple[int, int] = (11, 6),
):
    """
    Args:
        year_stride: plot every Nth year (after sorting unique years).
        min_count_per_year: skip years with fewer than this many ratings.
        normalize:
            - "count": show counts
            - "fraction": show within-year fractions (recommended for comparing shapes)
        show_overall: overlay overall distribution as a light reference in each panel.
    """

    if not hasattr(data[rel], "time"):
        print("No time information available for relation", rel)
        return
    if not hasattr(data[rel], "edge_label"):
        print("No edge labels (ratings) available for relation", rel)
        return

    years_t = data[rel].time.view(-1).detach().cpu().to(torch.long)
    ratings_t = data[rel].edge_label.view(-1).detach().cpu().to(torch.long)

    # Filter out invalid years/ratings if present
    valid = (years_t >= 0) & (ratings_t >= 1) & (ratings_t <= 5)
    years_t = years_t[valid]
    ratings_t = ratings_t[valid]

    if years_t.numel() == 0:
        print("No valid (year, rating) pairs after filtering.")
        return

    years = sorted(set(years_t.tolist()))
    years = years[::max(1, year_stride)]

    # Precompute overall reference distribution (1..5)
    def dist_from_ratings(r: torch.Tensor) -> np.ndarray:
        # bins 1..5 inclusive
        counts = torch.bincount(r.clamp(1, 5) - 1, minlength=5).to(torch.float32)
        if normalize == "fraction":
            counts = counts / (counts.sum() + 1e-12)
        return counts.numpy()

    overall = dist_from_ratings(ratings_t)

    # Collect per-year distributions
    year_items = []
    for y in years:
        m = (years_t == int(y))
        n = int(m.sum().item())
        if n < min_count_per_year:
            continue
        year_items.append((int(y), n, dist_from_ratings(ratings_t[m])))

    if not year_items:
        print(f"No years left after applying year_stride={year_stride} and min_count_per_year={min_count_per_year}.")
        return

    # Layout
    K = len(year_items)
    ncols = 4 if K >= 8 else 3 if K >= 6 else 2 if K >= 2 else 1
    nrows = int(np.ceil(K / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], max(figsize[1], 2.2 * nrows)), sharey=True)
    axes = np.array(axes).reshape(-1)

    x = np.arange(1, 6)
    width = 0.78

    # common y-limit for fraction mode
    if normalize == "fraction":
        ymax = max(max(d.max() for _, _, d in year_items), overall.max()) * 1.15
    else:
        ymax = None

    for ax_i, ax in enumerate(axes):
        if ax_i >= K:
            ax.axis("off")
            continue

        y, n, d = year_items[ax_i]

        # Overall baseline (light bars)
        if show_overall:
            ax.bar(x, overall, width=width, alpha=overall_alpha, edgecolor="none", label="overall")

        # Year distribution (solid bars)
        ax.bar(x, d, width=width, alpha=0.95, edgecolor="white", linewidth=0.8, label=str(y))

        # A bit of cleanup
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_axisbelow(True)

        title = f"{y}  (n={n:,})"
        ax.set_title(title, fontsize=11)

        if ymax is not None:
            ax.set_ylim(0, ymax)

    # Global labels
    fig.suptitle("Star Rating Distributions Over Time", fontsize=14, y=0.99)
    fig.supxlabel("Stars (1-5)")
    fig.supylabel("Fraction of reviews" if normalize == "fraction" else "Count")

    # Single legend (if baseline shown)
    if show_overall:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", frameon=False, bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.95)
    plt.savefig("reports/figures/star_shift_over_time.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    from torch_geometric.loader import LinkNeighborLoader
    from graph_ml.data import load_yelp_as_hetero, split_edge_indices_by_year

    rel = ("user", "reviews", "business")

    data = load_yelp_as_hetero('data/raw/',
                               max_reviews=500_000,
                               min_review_len=4, seed=42, coverage_threshold=0.5,
                               cache=True, cache_subdir="processed",
                               include_user_friends=True, max_friends_per_user=100,
                               use_text_edge_attr=True)

    print(data.metadata())

    visualize_star_shift(data, rel=rel, year_stride=4, show_overall=False)

    train_idx, val_idx, _ = split_edge_indices_by_year(
        data, rel=rel, boundary_year=2019, 
        include_boundary_in_train=True, seed=42, 
        shuffle_within_splits=True
    )

    num_neighbors = {
        ("user","reviews","business"):   [10, 5, 5, 0],
        ("business","rev_reviews","user"): [0, 0, 0, 0],
        ("user","friends","user"):       [5, 5, 0, 0],
        ("user","rev_friends","user"):   [0, 0, 0, 0],
    }

    stars = data[rel].edge_label
    years = data[rel].time.view(-1)
    train_years = years[train_idx]
    val_years   = years[val_idx]

    train_label = stars[train_idx]
    val_label   = stars[val_idx]

    batch_size = 4

    common_kwargs = dict(
        data=data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        time_attr='time',
    )

    train_loader = LinkNeighborLoader(
        **common_kwargs,
        edge_label_index=(rel, data[rel].edge_index[:, train_idx]),
        edge_label=train_label,
        edge_label_time=train_years,
    )

    val_loader = LinkNeighborLoader(
        **{**common_kwargs, "shuffle": False},
        edge_label_index=(rel, data[rel].edge_index[:, val_idx]),
        edge_label=val_label,
        edge_label_time=val_years,
    )

    for batch in train_loader:
        visualize_link_batch(
            batch.cpu(),
            highlight_rel=("user","reviews","business"),
            annotate_targets=True,
            annotate_sample_k=10,
            year_fmt=lambda y: str(int(y)),
            save_path=f"reports/figures/yelp_train_batch_size_{batch_size}.png"
        )
        break

    for batch in val_loader:
        visualize_link_batch(
            batch.cpu(),
            highlight_rel=("user","reviews","business"),
            annotate_targets=True,
            annotate_sample_k=10,
            year_fmt=lambda y: str(int(y)),
            save_path=f"reports/figures/yelp_val_batch_size_{batch_size}.png"
        )
        break