import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData
from typing import Tuple

class RelTemporalEncoding(nn.Module):
    """
    Cosine basis temporal encoding (learnable frequencies + phases).
    Input:  y  [E, d_in]   (we'll use d_in=1 with years)
    Output: [E, d_out]
    """
    def __init__(self, d_in: int = 1, n_freq: int = 8, d_out: int = 16, dropout: float = 0.0):
        super().__init__()
        self.d_in = d_in
        self.n_freq = n_freq
        self.d_out = d_out

        # Learnable frequencies (positive) and phases
        self.log_freq = nn.Parameter(torch.zeros(n_freq, d_in))   # exp -> (0, ∞)
        self.phase    = nn.Parameter(torch.zeros(n_freq, d_in))

        # Linear projection to d_out
        self.proj = nn.Linear(n_freq * d_in, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y: Tensor) -> Tensor:
        """
        y: [E, d_in] (float). Recommended to be roughly standardized.
        """
        # [E, d_in] -> [1, E, d_in] to broadcast over frequencies
        y_exp = y.unsqueeze(0)                                     # [1, E, d_in]
        freq  = torch.exp(self.log_freq).unsqueeze(1)              # [F, 1, d_in]
        phase = self.phase.unsqueeze(1)                            # [F, 1, d_in]

        # Cosine features: [F, E, d_in]
        c = torch.cos(y_exp * freq + phase)

        # Collapse to [E, F*d_in]
        c = c.permute(1, 0, 2).reshape(y.shape[0], -1)

        out = self.proj(c)                                         # [E, d_out]
        out = self.dropout(torch.nn.functional.gelu(out))
        return out

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

        self.time_enc = RelTemporalEncoding(d_in=1, n_freq=8, d_out=time_out, dropout=dropout)

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
            x_in = {k: v for k, v in x_dict.items()}
            x_pn = {k: self.norms[k](v) for k, v in x_dict.items()}
            x_out = conv(x_pn, edge_index_dict)
            for k in x_out:
                x_out[k] = x_in[k] + self.dropout(x_out[k])
            x_dict = x_out

        # Pair endpoints for labeled edges
        eidx: torch.Tensor = batch[rel].edge_label_index
        u_loc, v_loc = eidx[0], eidx[1]
        hu = x_dict[rel[0]][u_loc]; hv = x_dict[rel[2]][v_loc]
        parts = [hu, hv]
        if self.use_pair_interactions:
            parts += [torch.abs(hu - hv), hu * hv]

        if hasattr(batch[rel], "edge_label_time"):
            # shape: [E] or [E, 1] -> make it [E, 1] float
            t = batch[rel].edge_label_time.view(-1, 1).to(hu.dtype)

            # Standardize per-batch (keeps things numerically stable)
            t_mu  = t.mean(dim=0, keepdim=True)
            t_std = t.std(dim=0, keepdim=True).clamp_min(1e-6)
            t_z   = (t - t_mu) / t_std

            # Encode and append
            t_emb = self.time_enc(t_z)
            parts.append(t_emb)
        else:
            # If a batch ever arrives without timestamps, just skip.
            pass

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