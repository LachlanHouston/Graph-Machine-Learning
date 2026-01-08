from __future__ import annotations
from typing import Dict, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HEATConv

REL = ('user', 'reviews', 'business')


class RelTemporalEncoding(nn.Module):
    """
    Sinusoidal temporal encoding over discrete time indices in [0, max_len-1].
    Returns an embedding of dimension n_hid given integer time indices.
    """
    def __init__(self, n_hid: int, max_len: int = 240, dropout: float = 0.0):
        super().__init__()
        position = torch.arange(0., max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))

        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.weight.requires_grad_(False)

        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t_idx: Tensor) -> Tensor:
        """
        t_idx: Long tensor [E] or [E,1] with integer indices in [0, max_len-1].
        returns: [E, n_hid]
        """
        if t_idx.dim() > 1:
            t_idx = t_idx.view(-1)
        enc = self.lin(self.emb(t_idx))        # [E, n_hid]
        return self.dropout(enc)


class GlobalTransformer(nn.Module):
    """
    Simple Transformer block over the whole subgraph's node set (global attention).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.ln1  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.ln2  = nn.LayerNorm(d_model)
        self.do   = nn.Dropout(dropout)

    def forward(self, h: Tensor) -> Tensor:
        """
        h: [N, d_model] (nodes in the subgraph)
        returns: [N, d_model]
        """
        # Treat nodes as a length-N sequence, batch size = 1
        x = h.unsqueeze(1)
        x2, _ = self.attn(x, x, x)  # self-attention over all nodes
        x = self.ln1(x + self.do(x2))
        x2 = self.ffn(x)
        x = self.ln2(x + self.do(x2))
        return x.squeeze(1)         # [N, d_model]


class HeteroHEATStarPredictor(nn.Module):
    """
    Rating prediction (1..5) on review edges using HEATConv.
    - Uses node features `x` (same width for all node types) as input, no ID embeddings.
    - Hetero -> homo via `batch.to_homogeneous(node_attrs=['x'], edge_attrs=['edge_attr','time'])`.
    - First HEATConv layer consumes raw node features; later layers operate in `hidden_dim`.
    - Edge attributes are augmented with temporal encoding and passed via `edge_dim`.
    - Output is 4 logits (ordinal setup: thresholds between stars 1..5).
    """

    def __init__(
        self,
        metadata,
        node_feat_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_attr_dim: Optional[int] = None,
        edge_type_emb_dim: int = 32,
        edge_attr_emb_dim: int = 64,
        concat_heads: bool = True,
        time_out: int = 16,
        max_time_len: int = 240,
        base_time: Optional[int] = None,
    ):
        super().__init__()
        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])
        self.node_type_to_id = {nt: i for i, nt in enumerate(self.node_types)}

        self.node_feat_dim = int(node_feat_dim)
        self.hidden_dim = hidden_dim

        # Temporal encoding for edges
        self.time_out = int(time_out) if time_out is not None else 0
        self.max_time_len = int(max_time_len)
        self.base_time = base_time
        self.time_enc = RelTemporalEncoding(n_hid=self.time_out, max_len=self.max_time_len, dropout=dropout)

        # Edge attr dims: data attrs + time encoding
        self.edge_attr_dim_data = int(edge_attr_dim or 0)
        self.edge_attr_dim_total = self.edge_attr_dim_data + self.time_out

        # Global Transformer blocks (one per HEAT layer)
        self.global_blocks = nn.ModuleList(
            [GlobalTransformer(d_model=hidden_dim, n_heads=num_heads, dropout=dropout)
             for _ in range(num_layers)]
        )
        self.local_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # HEATConv stack
        convs = []

        # First layer: in_channels = node_feat_dim, out_channels = hidden_dim
        convs.append(HEATConv(
            in_channels=self.node_feat_dim,
            out_channels=hidden_dim,
            num_node_types=len(self.node_types),
            num_edge_types=len(self.edge_types),
            heads=num_heads,
            concat=concat_heads,
            dropout=dropout,
            edge_dim=self.edge_attr_dim_total,
            edge_attr_emb_dim=edge_attr_emb_dim,
            edge_type_emb_dim=edge_type_emb_dim,
            bias=True,
        ))

        # Subsequent layers: in_channels = hidden_dim
        for _ in range(1, num_layers):
            convs.append(HEATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_node_types=len(self.node_types),
                num_edge_types=len(self.edge_types),
                heads=num_heads,
                concat=concat_heads,
                dropout=dropout,
                edge_dim=self.edge_attr_dim_total,
                edge_attr_emb_dim=edge_attr_emb_dim,
                edge_type_emb_dim=edge_type_emb_dim,
                bias=True,
            ))
        self.convs = nn.ModuleList(convs)

        # If concat=True, HEATConv output dim = heads * out_channels
        out_per_layer = hidden_dim * num_heads if concat_heads else hidden_dim
        self.post_lin = nn.Linear(out_per_layer, hidden_dim) if concat_heads else nn.Identity()

        # Edge prediction head: [h_src || h_dst] -> 4 logits (ordinal)
        pred_in = 2 * hidden_dim
        pred_out = 5
        self.edge_pred = nn.Sequential(
            nn.Linear(pred_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_out),
        )

        self.act = nn.ReLU()
        self.do = nn.Dropout(dropout)

    # ---------- helpers ----------

    @staticmethod
    def _positions_by_type(node_type: Tensor, t_id: int) -> Tensor:
        # Homogeneous positions of nodes with this type id.
        return (node_type == t_id).nonzero(as_tuple=False).view(-1)

    def _time_to_index(self, t_raw: Tensor) -> Tensor:
        """
        Map integer times (e.g., years) to indices in [0, max_time_len-1].
        """
        t = t_raw.view(-1).long()
        if self.base_time is None:
            t0 = int(t.min().item())
        else:
            t0 = int(self.base_time)
        idx = (t - t0).clamp(min=0, max=self.max_time_len - 1)
        return idx

    def _build_homo_inputs(self, batch):
        """
        Build homogeneous view and edge attributes with time encoding.

        Returns:
          x_h   : [N_h, node_feat_dim]   (raw node features, same width for all types)
          ei_h  : [2, E_h]
          nt_h  : [N_h] (long node_type ids)
          et_h  : [E_h] (long edge_type ids)
          ea_h  : [E_h, edge_attr_dim_total]
          pos_idx_by_type: Dict[str, Tensor] mapping type -> homogeneous positions
        """
        # Hetero -> homo, carrying node features `x` + edge_attr + time
        homo = batch.to_homogeneous(node_attrs=['x'], edge_attrs=['edge_attr', 'time'])
        x_h  = homo.x           # [N_h, node_feat_dim]
        ei_h = homo.edge_index
        nt_h = homo.node_type
        et_h = homo.edge_type

        device = ei_h.device
        N_h = nt_h.numel()

        if x_h.size(-1) != self.node_feat_dim:
            raise RuntimeError(f"Expected node_feat_dim={self.node_feat_dim}, "
                               f"got x.size(-1)={x_h.size(-1)} from homogeneous graph.")

        x_h = x_h.to(device=device, dtype=torch.float32)

        # Homogeneous positions per type
        pos_idx_by_type: Dict[str, Tensor] = {
            nt: self._positions_by_type(nt_h, t_id)
            for nt, t_id in self.node_type_to_id.items()
        }

        # Construct edge_attr fed to HEATConv: [edge_attr || time_enc]
        ea_data = getattr(homo, 'edge_attr', None)
        if ea_data is None:
            if self.edge_attr_dim_data != 0:
                raise RuntimeError(
                    "Homogeneous graph has no edge_attr but edge_attr_dim_data > 0. "
                    "Ensure data[*].edge_attr exists for all relations or set edge_attr_dim=0."
                )
            ea_data = torch.empty((ei_h.size(1), 0), device=device, dtype=torch.float32)
        else:
            ea_data = ea_data.to(device=device, dtype=torch.float32)

        if not hasattr(homo, 'time'):
            raise RuntimeError("Homogeneous graph is missing 'time'. "
                               "Make sure edge_attrs=['edge_attr','time'] are passed to to_homogeneous.")
        t_idx = self._time_to_index(homo.time)
        t_enc = self.time_enc(t_idx.to(device=device))   # [E_h, time_out]

        ea_h = torch.cat([ea_data, t_enc], dim=-1)       # [E_h, edge_attr_dim_total]
        if ea_h.size(-1) != self.edge_attr_dim_total:
            raise RuntimeError(
                f"edge_attr dim mismatch: got {ea_h.size(-1)}, expected {self.edge_attr_dim_total}"
            )

        return x_h, ei_h, nt_h, et_h, ea_h, pos_idx_by_type

    def _hetero_pairs_to_homo(
        self,
        edge_label_index: Tensor,
        label_edge_type: Tuple[str, str, str],
        pos_idx_by_type: Dict[str, Tensor],
    ) -> Tensor:
        """
        Map hetero (src,dst) indices for the given relation to homogeneous indices
        using the positions list for each node type.
        """
        src_type, _, dst_type = label_edge_type
        src_local, dst_local = edge_label_index
        src_h = pos_idx_by_type[src_type][src_local]
        dst_h = pos_idx_by_type[dst_type][dst_local]
        return torch.stack([src_h, dst_h], dim=0)

    # ---------- forward ----------

    def forward(
        self,
        *,
        edge_label_index: Tensor,
        label_edge_type: Tuple[str, str, str],
        batch=None,
    ) -> Tensor:
        """
        Forward pass for a mini-batch produced by LinkNeighborLoader.
        """
        # 1) Build homogeneous inputs
        x_h, ei_h, nt_h, et_h, ea_h, pos_idx_by_type = self._build_homo_inputs(batch)

        # 2) HEATConv stack
        h = x_h
        for li, conv in enumerate(self.convs):
            h_loc = conv(
                x=h,
                edge_index=ei_h,
                node_type=nt_h,
                edge_type=et_h,
                edge_attr=ea_h,
            )
            h_loc = self.post_lin(h_loc)   # -> [N_h, hidden_dim] if concat_heads, else identity
            h_loc = F.relu(h_loc)
            h_loc = self.do(h_loc)

            if li == 0:
                # First layer: no residual from pre-conv features, just take transformed h_loc
                h = h_loc
            else:
                # Later layers: residual + layer norm in hidden_dim
                h = self.local_norms[li](h + h_loc)

            # Global transformer block
            h_glob = self.global_blocks[li](h)
            h = h + h_glob

        # 3) Score target pairs for the review relation
        edge_label_index_h = self._hetero_pairs_to_homo(edge_label_index, label_edge_type, pos_idx_by_type)
        src_h, dst_h = edge_label_index_h
        z = torch.cat([h[src_h], h[dst_h]], dim=-1)
        return self.edge_pred(z)                        # [M, 4] (ordinal logits)