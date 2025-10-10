import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData
from typing import Tuple

class RelTemporalEncoding(nn.Module):
    """
    Encode scalar(s) Δt with a small learnable cosine basis, as in pyHGT.
    Input: y in R^{E, d_in} (here d_in=2: Δt_u, Δt_b)
    Output: R^{E, d_out} with tiny dimension.
    """
    def __init__(self, d_in=2, n_freq=4, d_out=16):
        super().__init__()
        # One frequency vector per input channel
        self.freq = nn.Parameter(torch.randn(d_in, n_freq))   # learnable "w"
        self.phase= nn.Parameter(torch.zeros(d_in, n_freq))   # learnable "phi"
        self.proj = nn.Linear(d_in * n_freq, d_out)

    def forward(self, y):  # y: [E, d_in]
        # expand: [E, d_in, 1] -> broadcast with [d_in, n_freq]
        # result: [E, d_in, n_freq]
        w = y.unsqueeze(-1) * self.freq + self.phase
        c = torch.cos(w)                       # cosine basis
        c = c.flatten(1)                       # [E, d_in*n_freq]
        return self.proj(c)                    # [E, d_out]

class EdgeHGT(nn.Module):
    def __init__(self, metadata, hidden_dim=128, num_layers=2, heads=2, dropout=0.1,
                 use_pair_interactions=True, time_out=16):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.use_pair_interactions = use_pair_interactions

        # Featureless nodes → id embeddings
        self.embeds = nn.ModuleDict()
        self.norms  = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)
        for ntype in node_types:
            self.embeds[ntype] = None
            self.norms[ntype]  = nn.LayerNorm(hidden_dim)

        self.hgt = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=heads,
            ) for _ in range(num_layers)
        ])

        self.time_enc = RelTemporalEncoding(d_in=2, n_freq=4, d_out=time_out)

        self._global_num_nodes = {}

    def set_num_nodes(self, num_nodes_by_type: dict):
        for ntype, N in num_nodes_by_type.items():
            if self.embeds[ntype] is None:
                self.embeds[ntype] = nn.Embedding(N, self.hidden_dim)
                nn.init.normal_(self.embeds[ntype].weight, std=0.02)
        self._global_num_nodes = num_nodes_by_type

    def forward(self, batch: HeteroData,
                rel: Tuple[str,str,str] = ("user","reviews","business")) -> torch.Tensor:
        """
        Returns an edge representation per labeled edge: [E_b, D_edge].
        If edge_feat is provided, shape should be [E_b, edge_feat_dim].
        """
        x_dict = {}
        for ntype in batch.node_types:
            n_id: torch.Tensor = batch[ntype].n_id
            x = self.embeds[ntype](n_id)
            x = self.norms[ntype](F.gelu(x))
            x = self.dropout(x)
            x_dict[ntype] = x

        edge_index_dict = {etype: batch[etype].edge_index for etype in batch.edge_types}

        for conv in self.hgt:
            x_dict = conv(x_dict, edge_index_dict)
            for ntype in x_dict:
                x_dict[ntype] = self.dropout(self.norms[ntype](x_dict[ntype]))

        # Pair endpoints for labeled edges
        eidx: torch.Tensor = batch[rel].edge_label_index
        u_loc, v_loc = eidx[0], eidx[1]
        hu = x_dict[rel[0]][u_loc]; hv = x_dict[rel[2]][v_loc]
        parts = [hu, hv]
        if self.use_pair_interactions:
            parts += [torch.abs(hu - hv), hu * hv]

        # dt = batch[rel].edge_attr[:, :2].to(hu.dtype)     # [E,2] = (Δt_u, Δt_b)
        # t_emb = self.time_enc(dt)                         # [E,time_out]
        # parts.append(t_emb)

        return torch.cat(parts, dim=-1)
    
class EdgeClassifier(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=5, use_pair_interactions=True, time_out=0):
        super().__init__()
        base = 2*hidden_dim
        inter = 2*hidden_dim if use_pair_interactions else 0
        in_dim = base + inter + time_out
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, edge_repr: Tensor) -> Tensor:
        return self.mlp(edge_repr).squeeze(-1)