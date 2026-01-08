# baseline_mean_with_f1.py
import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.explain import Explainer, CaptumExplainer, GNNExplainer
from torch_geometric.explain.algorithm import AttentionExplainer


from graph_ml.data import load_yelp_as_hetero, REL, split_edge_indices_by_year, check_uniform_edge_attr_dim
from graph_ml.utils import set_seed, multilabel_f1_from_logits
from graph_ml.model_HEAT import HeteroHEATStarPredictor

class ExplainAdapter(nn.Module):
    def __init__(self, core_model, default_label_edge_type=REL):
        super().__init__()
        self.core = core_model
        self.default_label_edge_type = default_label_edge_type

        self._ctx_batch = None
        self._ctx_edge_label_index = None
        self._ctx_label_edge_type = None

        # keep originals so we can restore after masking
        self._orig_edge_attr = None

    def set_context(self, batch, edge_label_index=None, label_edge_type=None):
        self._ctx_batch = batch
        self._ctx_label_edge_type = label_edge_type or self.default_label_edge_type
        if edge_label_index is None:
            edge_label_index = batch[self._ctx_label_edge_type].edge_label_index
        self._ctx_edge_label_index = edge_label_index

        # snapshot original edge_attr for all edge types (some may not have edge_attr)
        self._orig_edge_attr = {}
        for etype in batch.edge_types:
            store = batch[etype]
            if hasattr(store, "edge_attr") and store.edge_attr is not None:
                self._orig_edge_attr[etype] = store.edge_attr

    def clear_context(self):
        # restore edge_attr if needed
        if self._ctx_batch is not None and self._orig_edge_attr is not None:
            for etype, ea in self._orig_edge_attr.items():
                self._ctx_batch[etype].edge_attr = ea
        self._ctx_batch = None
        self._ctx_edge_label_index = None
        self._ctx_label_edge_type = None
        self._orig_edge_attr = None

    def forward(self, x_dict, edge_index_dict, node_mask=None, edge_mask=None, *args, **kwargs):
        batch = self._ctx_batch
        if batch is None:
            raise ValueError("Call wrapped_model.set_context(batch, ...) before explainer(...).")

        # ---- Apply node mask (attributes) ----
        # node_mask is usually a dict for HeteroData, but handle tensor fallback.
        if isinstance(node_mask, dict):
            for ntype, x in x_dict.items():
                m = node_mask.get(ntype, None)
                batch[ntype].x = x if m is None else (x * m)
        else:
            # no per-type mask -> just write x
            for ntype, x in x_dict.items():
                batch[ntype].x = x

        # ---- Apply edge mask (attributes) by scaling edge_attr ----
        if isinstance(edge_mask, dict):
            for etype, m in edge_mask.items():
                if etype not in batch.edge_types:
                    continue
                store = batch[etype]
                if not (hasattr(store, "edge_attr") and store.edge_attr is not None):
                    continue

                ea0 = self._orig_edge_attr.get(etype, store.edge_attr)

                # m can be [E] (object) or [E, F] (attributes)
                if m.dim() == 1 and m.numel() == ea0.size(0):
                    store.edge_attr = ea0 * m.view(-1, 1)
                elif m.dim() == 2 and m.shape == ea0.shape:
                    store.edge_attr = ea0 * m
                else:
                    # still ensure m influences output to avoid "unused tensor" issues
                    store.edge_attr = ea0
                    out = self.core(
                        edge_label_index=self._ctx_edge_label_index,
                        label_edge_type=self._ctx_label_edge_type,
                        batch=batch,
                    )
                    return out + 0.0 * m.sum()

        out = self.core(
            edge_label_index=self._ctx_edge_label_index,
            label_edge_type=self._ctx_label_edge_type,
            batch=batch,
        )

        # Fallback to ensure edge_mask tensor counts as "used" if it comes as a single tensor
        if torch.is_tensor(edge_mask) and not isinstance(edge_mask, dict):
            out = out + 0.0 * edge_mask.sum()

        return out

class EdgeScorer(nn.Module):
    def __init__(self, core_model):
        super().__init__()
        self.core = core_model

    def forward(self, data):
        # Predict logits for all seed edges in this mini-batch:
        out = self.core(
            edge_label_index=data[REL].edge_label_index,
            label_edge_type=REL,
            batch=data,
        )  # shape [M, C]
        return out  # Explainer will pick 'index' later

def viz_edge_importance(edge_index, edge_weight, title="Edge importance"):
    edge_index = edge_index.cpu()
    w = edge_weight.cpu()
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())
    # Normalize weights for plotting widths
    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
    widths = [1.0 + 4.0*wn.item() for wn in w_norm]
    pos = nx.spring_layout(G, seed=0)
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, with_labels=False, node_size=50, width=widths, edge_color=w_norm, edge_cmap=plt.cm.viridis)
    plt.title(title); plt.tight_layout(); plt.show()

def ordinal_targets(y):
    B = y.size(0)
    k = torch.arange(1, 5, device=y.device).unsqueeze(0).expand(B, -1)
    return (y.unsqueeze(1) > k).float()

# ---------- Metrics ----------
def evaluate_regression(y_pred: torch.Tensor, y_true: torch.Tensor):
    mae = (y_pred - y_true.float()).abs().mean().item()
    rmse = ((y_pred - y_true.float())**2).mean().sqrt().item()
    acc_round = (y_pred.round().clamp(1, 5).long() == y_true).float().mean().item()
    return {"mae": mae, "rmse": rmse, "acc@rounded": acc_round}

@torch.no_grad()
def multiclass_f1_from_int_preds(y_pred_int: torch.Tensor, y_true_int: torch.Tensor, average: str = "macro", eps: float = 1e-8):
    """
    y_pred_int, y_true_int: int tensors in {1,2,3,4,5}
    average: 'macro' or 'micro'
    """
    # Remap to {0..4}
    y_pred = y_pred_int.to(torch.long) - 1
    y_true = y_true_int.to(torch.long) - 1
    C = 5

    # Confusion counts per class
    f1_per_class = []
    tp_sum = fp_sum = fn_sum = 0.0

    for c in range(C):
        tp = ((y_pred == c) & (y_true == c)).sum().float()
        fp = ((y_pred == c) & (y_true != c)).sum().float()
        fn = ((y_pred != c) & (y_true == c)).sum().float()

        if average == "macro":
            prec = tp / (tp + fp + eps)
            rec  = tp / (tp + fn + eps)
            f1_c = 2 * prec * rec / (prec + rec + eps)
            f1_per_class.append(f1_c)
        else:
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

    if average == "macro":
        return torch.stack(f1_per_class).mean().item()
    elif average == "micro":
        prec = tp_sum / (tp_sum + fp_sum + eps)
        rec  = tp_sum / (tp_sum + fn_sum + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        return f1.item()
    else:
        raise ValueError("average must be 'macro' or 'micro'")

# ---------- Baseline ----------
def mean_baseline(data_dir: Path, max_reviews: int, seed: int):
    data = load_yelp_as_hetero(
        data_dir,
        max_reviews=max_reviews,
        cache=True,
        include_user_friends=True,
        use_text_edge_attr=True,   # irrelevant for this baseline
        seed=seed,
    )

    y = data[REL].edge_label.view(-1)  # {1..5} long

    train_idx, val_idx = split_edge_indices_by_year(
        data, boundary_year=2017, include_boundary_in_train=True, shuffle_within_splits=True, seed=seed
    )

    # ---- compute global mean on TRAIN only ----
    mean_train = y[train_idx].float().mean().item()

    def predict(idx: torch.Tensor):
        if idx.numel() == 0:
            return None, None
        y_true = y[idx]
        y_pred = torch.full_like(y_true, fill_value=mean_train, dtype=torch.float32)
        return y_pred, y_true

    print(f"Train edges: {train_idx.numel()} | Val: {val_idx.numel()}")
    print(f"Global mean (train): {mean_train:.4f}")

    # ---- VAL ----
    y_pred, y_true = predict(val_idx)
    if y_pred is not None:
        # Regression-style metrics
        val_reg = evaluate_regression(y_pred, y_true)

        # Convert to integer classes for F1 (rounded & clamped into 1..5)
        y_pred_int = y_pred.round().clamp(1, 5).long()
        y_true_int = y_true.long()

        f1_macro = multiclass_f1_from_int_preds(y_pred_int, y_true_int, average="macro")
        f1_micro = multiclass_f1_from_int_preds(y_pred_int, y_true_int, average="micro")

        print(f"[Val] mean predictor -> {val_reg} | F1_macro={f1_macro:.4f} | F1_micro={f1_micro:.4f}")

# Most frequent rating baseline
def top_frequent_baseline(data_dir: Path, max_reviews: int, seed: int):
    data = load_yelp_as_hetero(
        data_dir,
        max_reviews=max_reviews,
        cache=True,
        include_user_friends=True,
        use_text_edge_attr=True,   # irrelevant for this baseline
        seed=seed,
    )

    y = data[REL].edge_label.view(-1)  # {1..5} long

    train_idx, val_idx = split_edge_indices_by_year(
        data, boundary_year=2017, include_boundary_in_train=True, shuffle_within_splits=True, seed=seed
    )

    # ---- compute most frequent rating on TRAIN only ----
    y_train = y[train_idx]
    counts = torch.bincount(y_train, minlength=6)  # index 0 unused
    top_rating = counts[1:].argmax().item() + 1  # +1 since ratings start at 1

    def predict(idx: torch.Tensor):
        if idx.numel() == 0:
            return None, None
        y_true = y[idx]
        y_pred = torch.full_like(y_true, fill_value=top_rating, dtype=torch.long)
        return y_pred, y_true

    print(f"Train edges: {train_idx.numel()} | Val: {val_idx.numel()}")
    print(f"Most frequent rating (train): {top_rating}")

    # ---- VAL ----
    y_pred, y_true = predict(val_idx)
    if y_pred is not None:
        # Regression-style metrics
        val_reg = evaluate_regression(y_pred.float(), y_true)

        # Convert to integer classes for F1 (already int)
        y_pred_int = y_pred.long()
        y_true_int = y_true.long()

        f1_macro = multiclass_f1_from_int_preds(y_pred_int, y_true_int, average="macro")
        f1_micro = multiclass_f1_from_int_preds(y_pred_int, y_true_int, average="micro")

        print(f"[Val] top-frequent predictor -> {val_reg} | F1_macro={f1_macro:.4f} | F1_micro={f1_micro:.4f}")

def evaluate_model(checkpoint_path: Path, data_dir: Path, seed: int):
    rel = ("user", "reviews", "business")

    # -------- Load data --------
    data = load_yelp_as_hetero(
        data_dir,
        cache=True,
        include_user_friends=True,
        use_text_edge_attr=True,
        seed=seed,
    )

    # Fanouts per hop per relation (you can tweak)
    num_neighbors = {
        ("user", "reviews", "business"): [20, 15, 10, 5],
        ("business", "rev_reviews", "user"): [6, 3, 0, 0],
        ("user", "friends", "user"): [6, 4, 2, 0],
    }

    # Temporal split (positives only)
    train_idx, val_idx = split_edge_indices_by_year(
        data, rel=rel, boundary_year=2017, include_boundary_in_train=True
    )

    edge_attr_dim = check_uniform_edge_attr_dim(data, rel, ("user", "friends", "user"))

    # Seed edge pairs (positives only)
    val_pos   = data[rel].edge_index[:, val_idx]
    val_stars = data[rel].edge_label[val_idx]
    years     = data[rel].time.view(-1)
    val_years = years[val_idx]

    # -------- Loader --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = max(1, (os.cpu_count() or 1) - 1) if device.type == "cuda" else 0
    prefetch_factor = 2 if device.type == "cuda" else None
    print(f"Using num_workers={num_workers} for data loading")

    common_kwargs = dict(
        data=data,
        num_neighbors=num_neighbors,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        time_attr='time',
    )

    val_loader = LinkNeighborLoader(
        **common_kwargs,
        edge_label_index=(rel, val_pos),
        edge_label=val_stars,
        edge_label_time=val_years,
    )

    # -------- Model --------

    model = HeteroHEATStarPredictor(
        metadata=data.metadata(),
        node_feat_dim=data['user'].x.size(1),
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        edge_attr_dim=edge_attr_dim,
    ).to(device)

    # -------- Load checkpoint (robust) --------
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = None
        if isinstance(ckpt, dict):
            # common keys: 'model', 'state_dict'
            if 'model' in ckpt and isinstance(ckpt['model'], dict):
                state = ckpt['model']
            elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
                state = ckpt['state_dict']
        if state is None and isinstance(ckpt, dict):
            # maybe it's already the raw state_dict
            state = ckpt
        if state is None:
            print(f"[warn] Could not find a model state_dict in {checkpoint_path}, skipping load.")
        else:
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[ckpt] Missing keys: {missing}\n[ckpt] Unexpected keys: {unexpected}")
        print(f"[ckpt] Loaded from: {checkpoint_path}")
    else:
        print(f"[ckpt] WARNING: checkpoint not found at {checkpoint_path}. Evaluating random-init model.")

    # -------- Eval loop --------
    model.eval()
    # y_true_stars = []
    # y_pred_stars = []
    # val_logits_list = []
    # val_targets_list = []
    # se_sum = 0.0
    # n_sum = 0

    # iter = 0

    # with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
    #     for batch in tqdm(val_loader, desc="Val", leave=False, dynamic_ncols=True):
    #         iter += 1
    #         batch = batch.to(device, non_blocking=True)
    #         y = batch[rel].edge_label.float()

    #         out = model(  # ordinal logits [B,4] or [B,5] depending on your model
    #             edge_label_index=batch[rel].edge_label_index,
    #             label_edge_type=rel,
    #             batch=batch,
    #         )

    #         # Ordinal targets for reporting F1; not used in RMSE
    #         t = ordinal_targets(y)

    #         # Decode stars from ordinal logits: 1 + count(sigmoid > 0.5)
    #         probs = torch.sigmoid(out)
    #         preds = 1 + (probs > 0.5).sum(dim=1)

    #         se_sum += ((preds - y) ** 2).sum().item()
    #         n_sum  += y.numel()
    #         y_true_stars.append(y)
    #         y_pred_stars.append(preds)

    #         val_logits_list.append(out.detach().cpu())
    #         val_targets_list.append(t.detach().cpu())

    #         if iter > 50:
    #             break

    # # -------- Metrics --------
    # val_rmse = float((se_sum / max(1, n_sum)) ** 0.5)

    # y_true = torch.cat(y_true_stars).float()
    # y_pred = torch.cat(y_pred_stars).float()

    # y_true_i = y_true.clamp(1, 5).round().int().cpu()
    # y_pred_i = y_pred.clamp(1, 5).round().int().cpu()
    # val_acc = float((y_true_i == y_pred_i).float().mean().item())

    # val_logits  = torch.cat(val_logits_list, dim=0)
    # val_targets = torch.cat(val_targets_list, dim=0)

    # f1_micro = float(multilabel_f1_from_logits(
    #     logits=val_logits, targets=val_targets, threshold=0.5, average="micro"
    # ))
    # f1_macro = float(multilabel_f1_from_logits(
    #     logits=val_logits, targets=val_targets, threshold=0.5, average="macro"
    # ))

    # print(f"[Val] RMSE={val_rmse:.4f} | acc@rounded={val_acc:.4f} | F1_macro={f1_macro:.4f} | F1_micro={f1_micro:.4f}")

    # # -------- Predicted vs True scatter (saved to disk) --------
    # # If your model outputs expected rating (soft expectation), you can plot that instead of the thresholded ints.
    # plt.figure(figsize=(6, 6))
    # plt.scatter(y_true.cpu().numpy(), y_pred.cpu().numpy(), alpha=0.5)
    # plt.plot([1, 5], [1, 5], linestyle='--')
    # plt.xlabel('True rating (stars)')
    # plt.ylabel('Predicted rating (stars)')
    # plt.title('Predicted vs True Ratings (Val)')
    # plt.xticks(np.arange(1, 6)); plt.yticks(np.arange(1, 6))
    # plt.grid(True, linestyle=':')
    # out_path = Path("eval_pred_vs_true.png")
    # plt.tight_layout(); plt.savefig(out_path, dpi=150)
    # print(f"[Val] Saved scatter plot: {out_path.resolve()}")

    # explainer = Explainer(
    #     model=EdgeScorer(model).eval(),     # your trained model
    #     algorithm=AttentionExplainer(reduce='mean'),
    #     explanation_type='model',              # explain a single edge’s prediction
    #     edge_mask_type='object',                 # <- pull attentions instead of optimizing a mask
    #     node_mask_type='attributes',                # (ignored unless you need node attributions)
    #     model_config=dict(
    #         mode='multiclass_classification',
    #         task_level='edge',                      # we explain an edge-level prediction
    #         return_type='raw',                   # your model returns logits
    #     )
    # )

    wrapped_model = ExplainAdapter(model).to(device).eval()

    explainer = Explainer(
        model=wrapped_model,
        algorithm=CaptumExplainer("IntegratedGradients"),
        explanation_type="model",
        node_mask_type=None,              # optional: turn off node attr for now
        edge_mask_type="object",      # <-- IMPORTANT
        model_config=dict(
            mode="multiclass_classification",
            task_level="edge",
            return_type="raw",
        ),
    )

    # Build a tiny loader that seeds ONE target review edge (clean visualization):
    val_loader_1 = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        batch_size=4,
        shuffle=False,
        edge_label_index=(REL, data[REL].edge_index[:, val_idx]),
        edge_label=data[REL].edge_label[val_idx],
        edge_label_time=data[REL].time[val_idx],
        time_attr="time",
    )

    batch = next(iter(val_loader_1)).to(device)
    wrapped_model.set_context(batch=batch, edge_label_index=batch[rel].edge_label_index, label_edge_type=rel)

    explanation = explainer(
        batch.x_dict,
        batch.edge_index_dict,
        index=0,
        # target=...  # you may need this next for multiclass IG
    )

    wrapped_model.clear_context()

    # print("edge_mask:", explanation.edge_mask_dict)
    print("node_mask:", explanation.node_mask_dict)

    explanation.visualize_feature_importance(top_k=10)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default='data/raw')
    parser.add_argument("--max_reviews", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # mean_baseline(args.data_dir, args.max_reviews, args.seed) # Global mean (train): 3.8233 [Val] mean predictor -> {'mae': 1.1910451650619507, 'rmse': 1.4042346477508545, 'acc@rounded': 0.19322076439857483}
    # top_frequent_baseline(args.data_dir, args.max_reviews, args.seed) # top-frequent predictor -> {'mae': 1.0489575862884521, 'rmse': 1.7481062412261963, 'acc@rounded': 0.5313649773597717} | F1_macro=0.1388 | F1_micro=0.5314
    evaluate_model(checkpoint_path=Path("models/focal_loss.pt"), data_dir=args.data_dir, seed=args.seed)