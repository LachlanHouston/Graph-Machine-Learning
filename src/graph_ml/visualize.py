import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def hetero_to_nx_yelp_full(
    data,
    include_edge_labels: bool = True,
    node_prefix: bool = True,
) -> nx.Graph:
    """
    Convert PyG HeteroData to NetworkX including:
      - user --reviews--> business  (with float stars if present)
      - user --friends--> user
    """
    def ukey(i): return f"u:{int(i)}" if node_prefix else ("user", int(i))
    def bkey(i): return f"b:{int(i)}" if node_prefix else ("business", int(i))

    G = nx.Graph()
    G.add_nodes_from((ukey(i), {"bipartite": 0, "ntype": "user"})
                     for i in range(data["user"].num_nodes))
    G.add_nodes_from((bkey(i), {"bipartite": 1, "ntype": "business"})
                     for i in range(data["business"].num_nodes))

    # reviews edges
    rel = ("user","reviews","business")
    ei = data[rel].edge_index
    stars = getattr(data[rel], "stars", None)
    ycls  = getattr(data[rel], "y", None)
    if stars is not None:
        s_np = stars.cpu().numpy()
        for u, b, s in zip(ei[0].cpu().numpy(), ei[1].cpu().numpy(), s_np):
            G.add_edge(ukey(u), bkey(b), etype="reviews", stars=float(s))
    else:
        ynp = ycls.cpu().numpy() if ycls is not None else None
        for i, (u, b) in enumerate(zip(ei[0].cpu().numpy(), ei[1].cpu().numpy())):
            s = ((ynp[i] + 2) / 2.0) if ynp is not None else None
            G.add_edge(ukey(u), bkey(b), etype="reviews", stars=s)

    # friends edges
    if ("user","friends","user") in data.edge_types:
        eif = data[("user","friends","user")].edge_index
        for u, v in zip(eif[0].cpu().numpy(), eif[1].cpu().numpy()):
            if u == v: 
                continue
            G.add_edge(ukey(u), ukey(v), etype="friends")

    return G


def sample_bipartite_subgraph(
    G: nx.Graph,
    max_users: int = 200,
    max_businesses: int = 200,
    seed: int = 69
) -> nx.Graph:
    rng = random.Random(seed)
    users = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    biz   = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]

    users_sample = set(rng.sample(users, min(max_users, len(users))))
    biz_sample   = set(rng.sample(biz,   min(max_businesses, len(biz))))

    keep = users_sample | biz_sample
    H = G.subgraph(keep).copy()

    # Optionally, drop isolated nodes (no edges after sampling)
    H.remove_nodes_from(list(nx.isolates(H)))
    return H

def ego_subgraph(G: nx.Graph, center_node: str, radius: int = 1) -> nx.Graph:
    return nx.ego_graph(G, center_node, radius=radius, undirected=True)

def yelp_bipartite_pos_pretty(G, jitter=0.06, seed=42):
    """
    Two-column bipartite layout but prettier:
    - users at x≈0, businesses at x≈1 (small x jitter)
    - y positions sorted by degree (denser areas get space)
    """
    rng = np.random.default_rng(seed)
    users = [n for n, d in G.nodes(data=True) if d.get("ntype") == "user"]
    biz   = [n for n, d in G.nodes(data=True) if d.get("ntype") == "business"]

    # order by degree (spread high-degree nodes out)
    users_sorted = sorted(users, key=lambda n: G.degree(n), reverse=True)
    biz_sorted   = sorted(biz,   key=lambda n: G.degree(n), reverse=True)

    def spaced_y(nodes):
        # evenly spaced y in [-1,1], then small jitter
        m = max(1, len(nodes))
        y = np.linspace(-1, 1, m)
        y += rng.normal(0, 0.02, size=m)
        return y

    yu = spaced_y(users_sorted)
    yb = spaced_y(biz_sorted)

    pos = {}
    for i, n in enumerate(users_sorted):
        pos[n] = np.array([0.0 + rng.normal(0, jitter), yu[i]])
    for i, n in enumerate(biz_sorted):
        pos[n] = np.array([1.0 + rng.normal(0, jitter), yb[i]])
    return pos

def draw_yelp_layers_pretty(
    G: nx.Graph,
    node_size_users=20,
    node_size_biz=28,
    alpha_reviews=0.55,
    alpha_friends=0.12,
    max_review_edges=None,     # e.g. 40_000 to downsample edges for speed
    max_friend_edges=None,     # e.g. 10_000
    seed=42,
):
    rng = np.random.default_rng(seed)
    users = [n for n, d in G.nodes(data=True) if d.get("ntype") == "user"]
    biz   = [n for n, d in G.nodes(data=True) if d.get("ntype") == "business"]

    pos = yelp_bipartite_pos_pretty(G, seed=seed)

    # Partition edge lists
    review_edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == "reviews"]
    friend_edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == "friends"]

    # Optional downsampling (visual sanity on large graphs)
    if max_review_edges is not None and len(review_edges) > max_review_edges:
        review_edges = list(rng.choice(review_edges, size=max_review_edges, replace=False))
    if max_friend_edges is not None and len(friend_edges) > max_friend_edges:
        friend_edges = list(rng.choice(friend_edges, size=max_friend_edges, replace=False))

    plt.figure(figsize=(12, 8))

    # Nodes
    n_users = nx.draw_networkx_nodes(G, pos, nodelist=users, node_size=node_size_users)
    n_biz   = nx.draw_networkx_nodes(G, pos, nodelist=biz,   node_size=node_size_biz, node_shape="s")

    # Reviews: color by stars (float), draw with slight curvature
    if review_edges:
        stars = [G.edges[e].get("stars", 3.0) for e in review_edges]
        e_reviews = nx.draw_networkx_edges(
            G, pos, edgelist=review_edges, width=0.6, alpha=alpha_reviews,
            edge_color=stars, edge_cmap=plt.cm.get_cmap(), edge_vmin=1.0, edge_vmax=5.0,
            connectionstyle="arc3,rad=0.08"  # subtle curve to reduce overlap
        )
        cbar = plt.colorbar(e_reviews)
        cbar.set_label("Review stars")
    # Friends: faint gray curves
    if friend_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=friend_edges, width=0.45, alpha=alpha_friends,
            edge_color="gray", connectionstyle="arc3,rad=0.08"
        )

    # Legends (proxy artists)
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='Users',
               markerfacecolor=n_users.get_facecolor()[0], markersize=np.sqrt(node_size_users)),
        Line2D([0], [0], marker='s', color='w', label='Businesses',
               markerfacecolor=n_biz.get_facecolor()[0], markersize=np.sqrt(node_size_biz)),
        Line2D([0], [0], color='gray', lw=2, alpha=0.5, label='Friends'),
        Line2D([0], [0], color='k', lw=2, alpha=0.6, label='Reviews (color = stars)'),
    ]
    plt.legend(handles=legend_elems, frameon=False, loc="upper center", ncol=2)

    plt.axis("off")
    plt.title("Yelp graph: Users ↔ Businesses (reviews) + User–User (friends)")
    plt.tight_layout()
    plt.show()

def draw_degree_hist(G: nx.Graph):
    degs = [d for _, d in G.degree()]
    plt.figure(figsize=(8, 5))
    plt.hist(degs, bins=50)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title("Degree distribution (sampled)")
    plt.tight_layout()
    plt.show()    