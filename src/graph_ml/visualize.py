# pip install networkx matplotlib pyvis
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import dill

from graph_ml.data import Graph
from graph_ml.utils import sample_subgraph, feature_MAG

def visualize_sampled_subgraph(
    graph,
    target_type: str,
    seed_node_ids,
    sampled_depth: int = 2,
    sampled_number: int = 8,
    feature_extractor=None,
    years_array=None,
    show=True,
    save_png_path=None,
    ):
    seed_node_ids = np.asarray(seed_node_ids, dtype=np.int64)
    if years_array is None:
        # fallback: try global graph.years if present, else zeros
        if hasattr(graph, "years") and isinstance(graph.years, np.ndarray) and len(graph.years) > 0:
            years_array = graph.years
        else:
            years_array = np.zeros_like(seed_node_ids, dtype=np.int32)

    seed_times = years_array[seed_node_ids] if len(years_array) > seed_node_ids.max() else np.zeros_like(seed_node_ids)
    inp = {target_type: np.vstack([seed_node_ids, seed_times]).T}  # shape (N, 2)

    feature, times, edge_list, indxs, texts = sample_subgraph(
        graph,
        inp=inp,
        sampled_depth=sampled_depth,
        sampled_number=sampled_number,
        feature_extractor=feature_extractor
    )

    G = _to_networkx_from_sample(edge_list, indxs, times, texts, include_self=False, include_reverse=False)

    seed_local = []
    orig_ids = set(map(int, seed_node_ids.tolist()))
    if target_type in indxs:
        for ser, orig in enumerate(indxs[target_type]):
            try:
                if int(orig) in orig_ids:
                    seed_local.append((target_type, ser))
            except Exception:
                pass

    _plot_subgraph(G, seed_nodes=seed_local, title=f"Sampled subgraph")
    if save_png_path:
        plt.savefig(save_png_path, bbox_inches='tight', dpi=160)
    if show:
        plt.show()
    else:
        plt.close()
    return G

def _to_networkx_from_sample(edge_list, indxs, times, texts=None, include_self=False, include_reverse=False):
    """
    Build a MultiDiGraph from sample_subgraph() outputs.
    Skips self-edges and reverse edges by default.
    """
    G = nx.MultiDiGraph()

    # Add nodes
    for ntype, serial_to_orig in indxs.items():
        for ser, orig in enumerate(serial_to_orig):
            tval = None
            if times is not None and ntype in times and ser < len(times[ntype]):
                tval = times[ntype][ser]
            label = None
            if texts is not None and ntype in texts and ser < len(texts[ntype]):
                label = texts[ntype][ser]

            G.add_node(
                (ntype, ser),
                type=ntype,
                ser=ser,
                orig=int(orig) if _is_int_like(orig) else orig,
                time=tval if tval is not None else None,
                label=label if label is not None else f"{ntype}:{orig}"
            )

    # Removes self and reversed edges
    def _keep_relation(rel: str) -> bool:
        if not include_self and rel == "self":
            return False
        if not include_reverse and rel.startswith("rev_"):
            return False
        return True
    
    for t_type, src_dict in edge_list.items():
        for s_type, rel_dict in src_dict.items():
            for relation, pairs in rel_dict.items():
                if not _keep_relation(relation):
                    continue
                for t_ser, s_ser in pairs:
                    u = (s_type, s_ser)
                    v = (t_type, t_ser)
                    if u in G and v in G:
                        G.add_edge(u, v,
                                   relation=relation,
                                   source_type=s_type,
                                   target_type=t_type)
    return G

def _plot_subgraph(G, seed_nodes=None, title=None, figsize=(11, 9), node_size=520, font_size=8):
    node_types = sorted({G.nodes[n]['type'] for n in G.nodes})
    relations  = sorted({edata['relation'] for _, _, edata in G.edges(data=True)})

    # Color palettes
    node_color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4']))
    edge_color_cycle = cycle(['#888888', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    type2color = {t: next(node_color_cycle) for t in node_types}
    rel2color  = {r: next(edge_color_cycle) for r in relations}

    # Layout
    pos = nx.spring_layout(G, seed=0, k=1.5)

    # Node visuals
    ncolors = [type2color[G.nodes[n]['type']] for n in G.nodes]
    edgecolors = []
    linewidths = []
    seed_set = set(seed_nodes) if seed_nodes else set()
    for n in G.nodes:
        if n in seed_set:
            edgecolors.append('black'); linewidths.append(2.2)
        else:
            edgecolors.append('black'); linewidths.append(0.8)

    # Edge visuals
    ecolors = [rel2color[d['relation']] for _, _, d in G.edges(data=True)]

    # Labels
    labels = {n: G.nodes[n].get('label', f"{G.nodes[n]['type']}:{G.nodes[n]['ser']}") for n in G.nodes}

    # Draw
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=ncolors,
                           edgecolors=edgecolors, linewidths=linewidths)
    nx.draw_networkx_edges(G, pos, edge_color=ecolors, arrows=True, arrowsize=12,
                           width=1.6, connectionstyle='arc3,rad=0.06')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size)
    if title: plt.title(title)

    # Legends
    from matplotlib.lines import Line2D
    node_legend = [Line2D([0],[0], marker='o', color='w', label=t, markerfacecolor=type2color[t],
                          markeredgecolor='black', markersize=10) for t in node_types]
    edge_legend = [Line2D([0],[0], color=rel2color[r], lw=2, label=r) for r in relations]
    leg1 = plt.legend(handles=node_legend, title="Node types", loc='upper left', bbox_to_anchor=(1.02, 1.00))
    plt.gca().add_artist(leg1)
    plt.legend(handles=edge_legend, title="Relations", loc='lower left', bbox_to_anchor=(1.02, 0.02))
    plt.axis('off')
    plt.tight_layout()

def _is_int_like(x):
    try:
        _ = int(x)
        return True
    except Exception:
        return False

if __name__ == '__main__':

    graph = dill.load(open('data/proc_Yelp_top300cats.pk', 'rb'))

    # Choose some seed businesses (global ids)
    seed_business_ids = [101, 105, 201]

    # Visualize
    G = visualize_sampled_subgraph(
        graph,
        target_type='business',
        seed_node_ids=seed_business_ids,
        sampled_depth=2,
        sampled_number=8,
        feature_extractor=feature_MAG,
        years_array=graph.years,
        show=True,
        save_png_path="yelp_sample.png"
)