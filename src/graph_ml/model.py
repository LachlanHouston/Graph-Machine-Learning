from __future__ import annotations
from typing import Dict, Tuple, Optional

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import HGTConv

REL = ('user', 'reviews', 'business')


class RelTemporalEncoding(nn.Module):
    '''
    Sinusoidal temporal encoding over discrete time indices in [0, max_len-1].
    Returns an embedding of dimension n_hid given integer time indices.
    '''
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
        t_idx: Long tensor of shape [E] or [E,1] with integer indices in [0, max_len-1].
        returns: [E, n_hid]
        """
        if t_idx.dim() > 1:
            t_idx = t_idx.view(-1)
        enc = self.lin(self.emb(t_idx))        # [E, n_hid]
        return self.dropout(enc)


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
        edge_attr_dim: Optional[int] = 301,      # e.g., 300 word2vec + 1 time (raw)
        use_edge_attr_on: Optional[set] = None,
        time_out: Optional[int] = 16,            # dimension of time embedding to CONCAT
        out_mode: str = "ordinal",
        max_time_len: int = 240,                 # max discrete time bins for sinusoid table
    ):
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.base_edge_attr_dim = edge_attr_dim  # raw edge attr dim before time concat
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

        # Temporal encoding module (provides vectors of length time_out)
        if time_out is not None:
            self.time_enc = RelTemporalEncoding(n_hid=time_out, max_len=max_time_len, dropout=dropout)
        else:
            self.time_enc = None
        self.time_out = time_out
        self.max_time_len = max_time_len

        if self.base_edge_attr_dim is not None and self.time_out is not None:
            self.edge_attr_dim = self.base_edge_attr_dim + self.time_out
        else:
            self.edge_attr_dim = self.base_edge_attr_dim

        # Edge head: [h_src || h_dst] (+ projected edge_attr of labeled edges)
        self.use_edge_text_at_head = (self.edge_attr_dim is not None)
        if self.use_edge_text_at_head:
            self.edge_text_proj = nn.Linear(self.edge_attr_dim, self.hidden_dim)
            pred_in = 2 * self.hidden_dim + self.hidden_dim
        else:
            self.edge_text_proj = None
            pred_in = 2 * self.hidden_dim

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

    @staticmethod
    def _shift_to_index(t_raw: Tensor, max_len: int) -> Tensor:
        """
        Map arbitrary numeric times (e.g., years) to [0, max_len-1] integers.
        """
        if t_raw is None:
            return None
        if t_raw.dim() > 1:
            t_raw = t_raw.view(-1)
        # convert to float for shifting, then to long indices
        t_float = t_raw.to(torch.float32)
        t_min = torch.minimum(t_float.min(), torch.tensor(0., device=t_float.device))
        t_idx = (t_float - t_min).round().to(torch.long)
        if max_len is not None:
            t_idx = t_idx.clamp_min(0).clamp_max(max_len - 1)
        return t_idx

    def _collect_edge_attr_dict(
        self,
        batch,
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        edge_attr_aug: Optional[Dict[Tuple[str, str, str], Tensor]] = None,
    ) -> Optional[Dict[Tuple[str, str, str], Tensor]]:
        """
        Collect per-relation edge attributes for message passing.
        """
        if batch is None or self.edge_attr_dim is None:
            return None
        ea = {}
        for et in edge_index_dict.keys():
            # prefer augmented attrs if provided
            if edge_attr_aug is not None and et in edge_attr_aug:
                eattr = edge_attr_aug[et]
            elif hasattr(batch[et], "edge_attr"):
                eattr = batch[et].edge_attr
            else:
                continue

            if eattr is None:
                continue

            if eattr.dim() == 1:
                eattr = eattr.view(-1, self.edge_attr_dim)
            elif self.edge_attr_dim is not None and eattr.size(-1) != self.edge_attr_dim:
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
                    eemb = self.edge_attr_proj(edge_attr_dict[etype])
                    dst = ei[1]
                    res[etype[2]].index_add_(0, dst, eemb)
                for nt in h_new:
                    gate = self.edge_res_gate[f"layer{li}|{nt}"]
                    h_new[nt] = h_new[nt] + gate * res[nt]

            for nt in h_new:
                h_new[nt] = self.do(self.act(h_new[nt]))
            h_dict = h_new

        return h_dict

    def forward(
        self,
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        *,
        edge_label_index: Tensor,
        label_edge_type: Tuple[str, str, str],
        batch=None,
    ) -> Tensor:

        edge_attr_aug: Dict[Tuple[str, str, str], Tensor] = {}

        if batch is not None and self.edge_attr_dim is not None:
            for edge_type in batch.edge_types:
                rel_store = batch[edge_type]
                if not hasattr(rel_store, "edge_attr") or rel_store.edge_attr is None:
                    continue

                eattr = rel_store.edge_attr
                E = eattr.size(0)
                device = eattr.device
                dtype = eattr.dtype

                # Try to get per-edge time for this relation
                t_raw = None
                if hasattr(rel_store, "time") and rel_store.time is not None:
                    t_raw = rel_store.time

                if self.time_out is not None:
                    if t_raw is None:
                        t_feat = torch.zeros(E, self.time_out, device=device, dtype=dtype)
                    else:
                        t_idx = self._shift_to_index(t_raw, self.max_time_len).to(device=device)
                        t_feat = self.time_enc(t_idx).to(dtype)  # [E, time_out]
                    eattr_aug = torch.cat([eattr, t_feat], dim=-1)  # [E, base+time_out]
                else:
                    eattr_aug = eattr

                edge_attr_aug[edge_type] = eattr_aug

        edge_attr_dict = self._collect_edge_attr_dict(batch, edge_index_dict, edge_attr_aug=edge_attr_aug)
        h_dict = self.encode_nodes(batch, edge_index_dict, edge_attr_dict)

        src_type, _, dst_type = label_edge_type
        src, dst = edge_label_index
        h_src = h_dict[src_type][src]
        h_dst = h_dict[dst_type][dst]
        z = torch.cat([h_src, h_dst], dim=-1)

        if self.use_edge_text_at_head and batch is not None and hasattr(batch[label_edge_type], "edge_attr"):
            rel_store = batch[label_edge_type]
            head_eattr = rel_store.edge_attr
            if head_eattr is not None:
                M = edge_label_index.size(1)

                # Compose label-aligned time features (or zeros) and concat
                if self.time_out is not None:
                    if hasattr(rel_store, "edge_label_time") and rel_store.edge_label_time is not None:
                        t_lab = rel_store.edge_label_time  # [M] or [M,1]
                        t_idx_lab = self._shift_to_index(t_lab, self.max_time_len).to(device=head_eattr.device)
                        t_feat_lab = self.time_enc(t_idx_lab).to(head_eattr.dtype)  # [M, time_out]
                    else:
                        t_feat_lab = torch.zeros(M, self.time_out, device=head_eattr.device, dtype=head_eattr.dtype)

                    # Align head_eattr to labels:
                    if head_eattr.size(0) >= M:
                        head_feats = torch.cat([head_eattr[:M], t_feat_lab], dim=-1)  # [M, edge_attr_dim]
                    else:
                        # Extremely rare fallback: if not enough rows, pad with zeros
                        pad = torch.zeros(M - head_eattr.size(0), head_eattr.size(1), device=head_eattr.device, dtype=head_eattr.dtype)
                        head_feats = torch.cat([torch.cat([head_eattr, pad], dim=0), t_feat_lab], dim=-1)
                else:
                    head_feats = head_eattr[:M] if head_eattr.size(0) >= M else head_eattr

                e_text = self.edge_text_proj(head_feats)  # -> [M, hidden_dim]
                z = torch.cat([z, e_text], dim=-1)

        return self.edge_pred(z)