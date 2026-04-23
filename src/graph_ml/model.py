from __future__ import annotations
from typing import Dict, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HGTConv

REL = ("user", "reviews", "business")


class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid: int, max_len: int = 240, dropout: float = 0.0):
        super().__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))

        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.weight.requires_grad_(False)

        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t_idx: Tensor) -> Tensor:
        if t_idx.dim() > 1:
            t_idx = t_idx.view(-1)
        return self.dropout(self.lin(self.emb(t_idx)))


class HeteroHGTStarPredictor(nn.Module):
    """
    Hetero-native edge classifier:

      batch (HeteroData)
        -> x_dict, edge_index_dict
        -> HGTConv stack
        -> gather embeddings for supervised edges in batch[label_edge_type].edge_label_index
        -> single Linear([h_src || h_dst || (edge_attr?) || (time_enc?)]) -> logits
    """

    def __init__(
        self,
        metadata,
        node_in_dims: Dict[str, int],
        num_users: int, 
        num_businesses: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 5,
        edge_attr_dim: int = 0,
        time_out: int = 16,
        edge_embed_dim: int = 128,
        max_time_len: int = 240,
        base_time: Optional[int] = None,
        use_edge_attr_in_head: bool = True,
        use_time_in_head: bool = True,
    ):
        super().__init__()
        self.metadata = metadata
        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])

        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.edge_attr_dim = int(edge_attr_dim or 0)
        self.time_out = int(time_out or 0)
        self.max_time_len = int(max_time_len)
        self.base_time = base_time
        self.edge_emb_dim = int(edge_embed_dim)

        self.use_edge_attr_in_head = bool(use_edge_attr_in_head) and (self.edge_attr_dim > 0)
        self.use_time_in_head = bool(use_time_in_head) and (self.time_out > 0)

        self.time_enc = RelTemporalEncoding(self.time_out, max_len=self.max_time_len, dropout=dropout) \
            if self.use_time_in_head else None

        # Type-specific input projections
        self.node_in = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.LayerNorm(node_in_dims[ntype]),
                nn.Linear(node_in_dims[ntype], self.hidden_dim),
            )
            for ntype in self.node_types
        })

        # HGTConv stack
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                metadata=self.metadata,
                heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        # Edge attribute boosting
        self.edge_in_dim = (self.edge_attr_dim if self.use_edge_attr_in_head else 0) + \
                           (self.time_out if self.use_time_in_head else 0)
        
        self.edge_repr = nn.Identity() if edge_attr_dim == 0 else nn.Sequential(
            nn.LayerNorm(self.edge_in_dim),
            nn.Linear(self.edge_in_dim, self.edge_emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.edge_emb_dim, self.edge_emb_dim),
        )

        self.edge_gain = nn.Parameter(torch.tensor(1.0))

        self.do = nn.Dropout(self.dropout)

        # Single linear layer
        head_in = 2 * self.hidden_dim + (self.edge_emb_dim if self.edge_in_dim > 0 else 0)
        self.classifier = nn.Linear(head_in, num_classes)

    def _time_to_index(self, t_raw: Tensor) -> Tensor:
        t = t_raw.view(-1).long()
        t0 = int(self.base_time) if self.base_time is not None else int(t.min().item())
        return (t - t0).clamp(min=0, max=self.max_time_len - 1)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        *,
        edge_label_index: Tensor,
        label_edge_type: Tuple[str, str, str] = REL,
        edge_label_attr: Optional[Tensor] = None,
        edge_label_time: Optional[Tensor] = None,
    ) -> Tensor:
        # 1) Project nodes
        x_dict = {nt: self.node_in[nt](x.float()) for nt, x in x_dict.items()}

        # 2) HGTConv stack
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {nt: self.do(F.relu(x)) for nt, x in x_dict.items()}

        # 3) Gather embeddings
        src_type, _, dst_type = label_edge_type
        src, dst = edge_label_index[0], edge_label_index[1]
        h_src = x_dict[src_type][src]
        h_dst = x_dict[dst_type][dst]
        parts = [h_src, h_dst]

        # 4) Optional supervised edge_attr in head
        if self.use_edge_attr_in_head:
            if edge_label_attr is None:
                edge_label_attr = torch.zeros(
                    (src.numel(), self.edge_attr_dim),
                    device=h_src.device,
                    dtype=torch.float32,
                )

        # 5) Optional supervised time in head
        if self.use_time_in_head:
            if edge_label_time is None:
                edge_label_time = torch.zeros((src.numel(),), device=h_src.device, dtype=torch.long)

            t_idx = self._time_to_index(edge_label_time)
            t_feat = self.time_enc(t_idx)

        # 6) Edge boost
        if self.edge_in_dim > 0:
            e = torch.cat([edge_label_attr, t_feat], dim=-1)
            e = self.edge_repr(e)
            e = e * self.edge_gain
            parts.append(e)

        z = torch.cat(parts, dim=-1)
        return self.classifier(z) # Classify
