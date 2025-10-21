# baseline_mean_with_f1.py
from pathlib import Path
import argparse
import torch
from graph_ml.data import load_yelp_as_hetero, REL, split_edge_indices_by_year

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default='data/raw')
    parser.add_argument("--max_reviews", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    mean_baseline(args.data_dir, args.max_reviews, args.seed) # Global mean (train): 3.8233 [Val] mean predictor -> {'mae': 1.1910451650619507, 'rmse': 1.4042346477508545, 'acc@rounded': 0.19322076439857483}
