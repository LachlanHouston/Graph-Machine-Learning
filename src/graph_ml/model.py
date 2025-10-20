# model_hgt_no_edge_dim.py

from __future__ import annotations
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import HGTConv


class HGTStarPredictor(nn.Module):
    """
    HGT-based heterogeneous GNN for predicting review stars on edges.

    We embed node IDs from the LinkNeighborLoader mini-batch (batch[nt].n_id),
    so input "features" are just indices per node type.
    """

    def __init__(
        self,
        metadata,
        num_nodes: Dict[str, int],
        in_dims: Dict[str, int],                 # kept for API compatibility; unused with embeddings
        hidden_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_attr_dim: Optional[int] = 301,      # e.g., 300 word2vec + 1 time
        use_edge_attr_on: Optional[set] = None,  # restrict which relations inject edge_attr (None = auto)
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

        # Edge head: [h_src || h_dst] (+ optional projected edge_attr of labeled edges)
        self.use_edge_text_at_head = (self.edge_attr_dim is not None)
        if self.use_edge_text_at_head:
            self.edge_text_proj = nn.Linear(self.edge_attr_dim, self.hidden_dim)
            pred_in = 2 * self.hidden_dim + self.hidden_dim
        else:
            self.edge_text_proj = None
            pred_in = 2 * self.hidden_dim

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
        batch,  # HeteroData mini-batch; we use batch[nt].n_id as indices
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
        x_dict: Dict[str, Tensor],  # ignored; kept for signature compatibility
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        *,
        edge_label_index: Tensor,
        edge_type: Tuple[str, str, str],
        batch=None,
    ) -> Tensor:
        edge_attr_dict = self._collect_edge_attr_dict(batch, edge_index_dict)
        h_dict = self.encode_nodes(batch, edge_index_dict, edge_attr_dict)

        src_type, _, dst_type = edge_type
        src, dst = edge_label_index
        h_src = h_dict[src_type][src]
        h_dst = h_dict[dst_type][dst]
        z = torch.cat([h_src, h_dst], dim=-1)

        if self.use_edge_text_at_head and batch is not None and hasattr(batch[edge_type], "edge_attr"):
            M = edge_label_index.size(1)
            e_text = self.edge_text_proj(batch[edge_type].edge_attr[:M])
            z = torch.cat([z, e_text], dim=-1)

        return self.edge_pred(z)
