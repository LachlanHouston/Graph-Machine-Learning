# model_hgt_no_edge_dim.py

from __future__ import annotations
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import HGTConv

REL = ('user', 'reviews', 'business')

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

class HGTStarPredictor(nn.Module):
    """
    HGT-based heterogeneous GNN for predicting review stars on edges.
    """

    def __init__(
        self,
        metadata,
        num_nodes: Dict[str, int],
        hidden_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_attr_dim: Optional[int] = 301,      # e.g., 300 word2vec + 1 time
        use_edge_attr_on: Optional[set] = None,
        time_out: Optional[int] = 16,
        out_mode: str = "ordinal",
    ):
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.edge_attr_dim = edge_attr_dim
        self.use_edge_attr_on = use_edge_attr_on
        self.out_mode = out_mode.lower()
        assert self.out_mode in {"ordinal", "multiclass"}

        # Per-type embeddings over global node IDs
        self.embeds = nn.ModuleDict({
            nt: nn.Embedding(num_embeddings=num_nodes[nt], embedding_dim=hidden_dim)
            for nt in metadata[0]
        })

        # HGT stack (no edge_dim support)
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        # Temporal encoding for edge timestamps
        if time_out is not None:
            self.time_enc = RelTemporalEncoding(d_in=1, n_freq=8, d_out=time_out, dropout=dropout)

        # Edge head: [h_src || h_dst] (+ optional projected edge_attr of labeled edges)
        self.use_edge_text_at_head = (self.edge_attr_dim is not None)
        if time_out is not None and self.edge_attr_dim is not None:
            self.edge_attr_dim = self.edge_attr_dim + time_out
            self.edge_text_proj = nn.Linear(self.edge_attr_dim, self.hidden_dim)
            pred_in = 2 * self.hidden_dim + self.hidden_dim
        else:
            self.edge_text_proj = None
            pred_in = 2 * self.hidden_dim + (self.hidden_dim if self.use_edge_text_at_head else 0)

        # Edge-attr residual injection path
        if self.edge_attr_dim is not None:
            self.edge_attr_proj = nn.Linear(self.edge_attr_dim, hidden_dim)
            self.edge_res_gate = nn.ParameterDict({
                f"layer{li}|{nt}": nn.Parameter(torch.tensor(0.5))
                for li in range(num_layers) for nt in metadata[0]
            })
        else:
            self.edge_attr_proj = None
            self.edge_res_gate = None

        pred_out = 4 if self.out_mode == "ordinal" else 5
        self.edge_pred = nn.Sequential(
            nn.Linear(pred_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_out),
        )

        self.act = nn.ReLU()
        self.do = nn.Dropout(dropout)

    # ---- internals ----

    def _collect_edge_attr_dict(
        self,
        batch,
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Optional[Dict[Tuple[str, str, str], Tensor]]:
        if batch is None or self.edge_attr_dim is None:
            return None
        ea = {}
        for et in edge_index_dict.keys():
            if not hasattr(batch[et], "edge_attr"):
                continue
            eattr = batch[et].edge_attr
            if eattr is None:
                continue
            if self.use_edge_attr_on is not None and et not in self.use_edge_attr_on:
                continue
            if eattr.dim() == 1:
                eattr = eattr.view(-1, self.edge_attr_dim)
            elif eattr.size(-1) != self.edge_attr_dim:
                raise RuntimeError(
                    f"edge_attr dim mismatch for {et}: got {eattr.size(-1)}, expected {self.edge_attr_dim}"
                )
            ea[et] = eattr
        return ea or None

    def encode_nodes(
        self,
        batch,
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], Tensor]] = None,
    ) -> Dict[str, Tensor]:
        # Embed node IDs per type
        h_dict = {nt: self.embeds[nt](batch[nt].n_id) for nt in batch.node_types}

        # HGT layers
        for li, conv in enumerate(self.convs):
            h_new = conv(h_dict, edge_index_dict)

            # Residual injection of projected edge_attr into destination nodes
            if self.edge_attr_proj is not None and edge_attr_dict is not None:
                res = {nt: torch.zeros_like(h_new[nt]) for nt in h_new}
                for etype, ei in edge_index_dict.items():
                    if etype not in edge_attr_dict:
                        continue
                    eemb = self.edge_attr_proj(edge_attr_dict[etype])  # [E, H]
                    dst = ei[1]                                        # [E]
                    res[etype[2]].index_add_(0, dst, eemb)
                for nt in h_new:
                    gate = self.edge_res_gate[f"layer{li}|{nt}"]
                    h_new[nt] = h_new[nt] + gate * res[nt]

            for nt in h_new:
                h_new[nt] = self.do(self.act(h_new[nt]))
            h_dict = h_new

        return h_dict

    # ---- public forward ----

    def forward(
        self,
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        *,
        edge_label_index: Tensor,
        label_edge_type: Tuple[str, str, str],
        batch=None,
    ) -> Tensor:
        
        time_tensor = None
        for edge_type in batch.edge_types:
            if edge_type == ('business', 'rev_reviews', 'user'):
                empty = torch.empty((batch[edge_type].num_edges, self.time_enc.d_out), device=batch[edge_type].edge_index.device)
                batch[edge_type].edge_attr = torch.cat([batch[edge_type].edge_attr, empty], dim=-1)
                continue
            rel_store = batch[edge_type]
            if hasattr(rel_store, "edge_attr"):
                # If you’ve packed time into the last column of edge_attr,
                # slice it out and align like above.
                eattr = rel_store.edge_attr
                time_tensor = eattr[:, -1]

                t = time_tensor.view(-1, 1)
                t_mu  = t.mean(dim=0, keepdim=True)
                t_std = t.std(dim=0, keepdim=True).clamp_min(1e-6)
                t_z   = (t - t_mu) / t_std
                t_emb = self.time_enc(t_z)
                batch[edge_type].edge_attr = torch.cat([batch[edge_type].edge_attr, t_emb], dim=-1)
            

        edge_attr_dict = self._collect_edge_attr_dict(batch, edge_index_dict)
        h_dict = self.encode_nodes(batch, edge_index_dict, edge_attr_dict)

        src_type, _, dst_type = label_edge_type
        src, dst = edge_label_index
        h_src = h_dict[src_type][src]
        h_dst = h_dict[dst_type][dst]
        z = torch.cat([h_src, h_dst], dim=-1)

        if self.use_edge_text_at_head and batch is not None and hasattr(batch[label_edge_type], "edge_attr"):
            M = edge_label_index.size(1)
            e_text = self.edge_text_proj(batch[label_edge_type].edge_attr[:M])
            z = torch.cat([z, e_text], dim=-1)

        return self.edge_pred(z)
