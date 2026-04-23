from pathlib import Path
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from tqdm import tqdm
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

from graph_ml.data import (
    load_yelp_as_hetero,
    REL,
    split_edge_indices_by_year,
    check_uniform_edge_attr_dim,
)
from graph_ml.utils import set_seed, build_num_neighbors_from_cfg, ordinal_targets
from graph_ml.model import HeteroHGTStarPredictor


def plot_confmatrix(y_true_i, y_pred_i, title="Test Confusion Matrix"):
    class_labels = ["1 star", "2 star", "3 star", "4 star", "5 star"]
    labels = [1, 2, 3, 4, 5]

    y_true = y_true_i.cpu().numpy()
    y_pred = y_pred_i.cpu().numpy()

    cm = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
    )

    cm_norm = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(6.5, 6))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="PuBu")

    ax.set(
        xticks=np.arange(len(class_labels)),
        yticks=np.arange(len(class_labels)),
        xticklabels=class_labels,
        yticklabels=class_labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    # Add subtle white grid between cells
    ax.set_xticks(np.arange(-0.5, len(class_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Use normalized matrix for text contrast threshold
    thresh = cm_norm.max() / 1.2 if cm_norm.max() > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text = f"{count}\n[{pct:.1f}%]"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "#222222",
                fontsize=10,
                fontweight="medium",
            )

    fig.tight_layout()
    plt.savefig("reports/figures/test_cormat.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_per_class_metrics(y_true_i, y_pred_i, labels=(1, 2, 3, 4, 5), prefix="[Test]"):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_i,
        y_pred_i,
        labels=list(labels),
        average=None,
        zero_division=0,
    )

    print(f"{prefix} per-class metrics:")
    for cls, p, r, f, s in zip(labels, precision, recall, f1, support):
        print(
            f"  Class {cls}: "
            f"precision={p:.4f} | recall={r:.4f} | f1={f:.4f} | support={s}"
        )

    return {
        int(cls): {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for cls, p, r, f, s in zip(labels, precision, recall, f1, support)
    }


def _load_ckpt_into_model(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    if checkpoint_path is None:
        return
    p = Path(checkpoint_path)
    if not p.exists():
        print(f"[ckpt] WARNING: checkpoint not found at {checkpoint_path}. Using random-init model.")
        return

    ckpt = torch.load(str(p), map_location=device)
    state = None
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        else:
            state = ckpt

    if not isinstance(state, dict):
        print(f"[warn] Could not find a model state_dict in {checkpoint_path}, skipping load.")
        return

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[ckpt] Missing keys: {missing}\n[ckpt] Unexpected keys: {unexpected}")
    print(f"[ckpt] Loaded from: {checkpoint_path}")


def evaluate_regression(y_pred: torch.Tensor, y_true: torch.Tensor):
    mae = (y_pred - y_true.float()).abs().mean().item()
    rmse = ((y_pred - y_true.float()) ** 2).mean().sqrt().item()
    acc_round = (y_pred.round().clamp(1, 5).long() == y_true).float().mean().item()
    return {"mae": mae, "rmse": rmse, "acc@rounded": acc_round}


def _build_data_for_baseline(cfg: DictConfig):
    data_dir = to_absolute_path(cfg.data.data_dir)
    rel = ("user", "reviews", "business")
    data = load_yelp_as_hetero(
        data_dir=data_dir,
        max_reviews=cfg.data.max_reviews,
        sample_method=getattr(cfg.data, "sample_method", "reservoir"),
        min_review_len=cfg.data.min_review_len,
        coverage_threshold=cfg.data.coverage_threshold,
        seed=cfg.data.seed,
        cache=True,
        cache_subdir="processed",
        include_user_friends=True,
        max_friends_per_user=100,
        use_text_edge_attr=cfg.data.use_text_edge_attr,
    )

    train_idx, val_idx, test_idx = split_edge_indices_by_year(
        data,
        rel=rel,
        boundary_year=cfg.data.year_cutoff,
        include_boundary_in_train=True,
        shuffle_within_splits=True,
        seed=cfg.data.seed,
    )

    y = data[rel].edge_label.view(-1)
    return data, rel, y, train_idx, val_idx, test_idx


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def mean_baseline(cfg: DictConfig):
    set_seed(cfg.data.seed)
    _, _, y, train_idx, _, test_idx = _build_data_for_baseline(cfg)

    mean_train = int(round(y[train_idx].float().mean().item()))

    def predict(idx: torch.Tensor):
        if idx.numel() == 0:
            return None, None
        y_true = y[idx]
        y_pred = torch.full((y_true.numel(),), mean_train, dtype=torch.float32, device=y_true.device)
        return y_pred, y_true

    print(f"Train edges: {train_idx.numel()} | Test: {test_idx.numel()}")
    print(f"Global mean (train): {mean_train:.4f}")

    y_pred, y_true = predict(test_idx)
    if y_pred is not None:
        test_reg = evaluate_regression(y_pred, y_true)
        y_pred_i = y_pred.round().clamp(1, 5).long().cpu().numpy()
        y_true_i = y_true.long().cpu().numpy()

        labels = [1, 2, 3, 4, 5]
        f1_macro = f1_score(y_true_i, y_pred_i, average="macro", labels=labels, zero_division=0)
        f1_micro = f1_score(y_true_i, y_pred_i, average="micro", labels=labels, zero_division=0)
        per_class = print_per_class_metrics(y_true_i, y_pred_i, labels=labels, prefix="[Test mean baseline]")

        print(f"[Test] mean predictor -> {test_reg} | F1_macro={f1_macro:.4f} | F1_micro={f1_micro:.4f}")
        print({"per_class": per_class})


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def top_frequent_baseline(cfg: DictConfig):
    set_seed(cfg.data.seed)
    _, _, y, train_idx, _, test_idx = _build_data_for_baseline(cfg)

    y_train = y[train_idx]
    counts = torch.bincount(y_train, minlength=6)
    top_rating = counts[1:].argmax().item() + 1

    def predict(idx: torch.Tensor):
        if idx.numel() == 0:
            return None, None
        y_true = y[idx]
        y_pred = torch.full_like(y_true, fill_value=top_rating, dtype=torch.long)
        return y_pred, y_true

    print(f"Train edges: {train_idx.numel()} | Test: {test_idx.numel()}")
    print(f"Most frequent rating (train): {top_rating}")

    y_pred, y_true = predict(test_idx)
    if y_pred is not None:
        test_reg = evaluate_regression(y_pred.float(), y_true)
        y_pred_i = y_pred.long().cpu().numpy()
        y_true_i = y_true.long().cpu().numpy()

        labels = [1, 2, 3, 4, 5]
        f1_macro = f1_score(y_true_i, y_pred_i, average="macro", labels=labels, zero_division=0)
        f1_micro = f1_score(y_true_i, y_pred_i, average="micro", labels=labels, zero_division=0)
        per_class = print_per_class_metrics(y_true_i, y_pred_i, labels=labels, prefix="[Test top-frequent baseline]")

        print(f"[Test] top-frequent predictor -> {test_reg} | F1_macro={f1_macro:.4f} | F1_micro={f1_micro:.4f}")
        print({"per_class": per_class})


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def evaluate_model(cfg: DictConfig):
    set_seed(cfg.data.seed)

    checkpoint_paths = ["models/model.pt", "models/model_seed44.pt", "models/model_seed46.pt", "models/model_seed48.pt", "models/model_seed50.pt", "models/model_seed52.pt"]
    rel = ("user", "reviews", "business")
    data_dir = to_absolute_path(cfg.data.data_dir)

    data = load_yelp_as_hetero(
        data_dir=data_dir,
        max_reviews=cfg.data.max_reviews,
        sample_method="reservoir",
        min_review_len=cfg.data.min_review_len,
        coverage_threshold=cfg.data.coverage_threshold,
        seed=cfg.data.seed,
        cache=True,
        cache_subdir="processed",
        include_user_friends=True,
        max_friends_per_user=100,
        use_text_edge_attr=cfg.data.use_text_edge_attr,
        use_sentiment_edge_attr=True,
    )

    num_neighbors = build_num_neighbors_from_cfg(cfg)

    _, _, test_idx = split_edge_indices_by_year(
        data, rel=rel, boundary_year=cfg.data.year_cutoff, include_boundary_in_train=True
    )

    edge_attr_dim = check_uniform_edge_attr_dim(data, rel, ("user", "friends", "user"))
    years = data[rel].time.view(-1)

    test_pos = data[rel].edge_index[:, test_idx]
    test_stars = data[rel].edge_label[test_idx]
    test_years = years[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for pth in checkpoint_paths:

        model = HeteroHGTStarPredictor(
            metadata=data.metadata(),
            node_in_dims={ntype: data[ntype].x.size(-1) for ntype in data.node_types},
            num_users=data["user"].num_nodes,
            num_businesses=data["business"].num_nodes,
            hidden_dim=cfg.model.n_hid,
            num_layers=cfg.model.n_layers,
            num_heads=cfg.model.n_heads,
            dropout=cfg.model.dropout,
            num_classes=5,
            edge_attr_dim=edge_attr_dim,
            edge_embed_dim=cfg.model.edge_embed_dim,
            time_out=16,
            use_edge_attr_in_head=True,
            use_time_in_head=True,
        ).to(device)

        _load_ckpt_into_model(model, pth, device)
        model.eval()

        test_edge_attr = data[rel].edge_attr[test_idx] if data[rel].edge_attr is not None else None
        test_edge_attr = test_edge_attr.to(device, non_blocking=True) if test_edge_attr is not None else None

        num_workers = max(1, os.cpu_count() - 1) if device.type == "cuda" else 0
        print(f"Using num_workers={num_workers} for data loading")

        prefetch_factor = 2 if device.type == "cuda" else None

        common_kwargs = dict(
            data=data,
            num_neighbors=num_neighbors,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor,
            time_attr="time",
        )

        test_loader = LinkNeighborLoader(
            **{**common_kwargs, "shuffle": False},
            edge_label_index=(rel, test_pos),
            edge_label=test_stars,
            edge_label_time=test_years,
        )

        se_sum = 0
        n_sum = 0
        y_true_stars = []
        y_pred_stars = []

        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="Test", leave=False, dynamic_ncols=True):
                batch = batch.to(device, non_blocking=True)
                y = batch[rel].edge_label.to(torch.long)
                y = (y - 1).clamp_(0, 4)

                out = model(
                    batch.x_dict,
                    batch.edge_index_dict,
                    edge_label_index=batch[rel].edge_label_index,
                    label_edge_type=rel,
                    edge_label_attr=test_edge_attr[batch[rel].input_id] if batch[rel].edge_attr is not None else None,
                    edge_label_time=getattr(batch[rel], "edge_label_time", None),
                )

                pred_cls = out.argmax(dim=1)
                preds = 1 + pred_cls

                se_sum += ((pred_cls - y) ** 2).sum().item()
                n_sum += y.numel()
                y_true_stars.append(y + 1)
                y_pred_stars.append(preds)

        test_rmse = (se_sum / max(1, n_sum)) ** 0.5

        y_true = torch.cat(y_true_stars).float()
        y_pred = torch.cat(y_pred_stars).float()
        y_true_i = y_true.clamp(1, 5).round().int().cpu()
        y_pred_i = y_pred.clamp(1, 5).round().int().cpu()
        test_acc = (y_true_i == y_pred_i).float().mean().item()
        test_mae = torch.abs(y_true - y_pred).mean().item()

        print(len(y_true))

        labels = [1, 2, 3, 4, 5]
        f1_macro = f1_score(y_true_i.numpy(), y_pred_i.numpy(), average="macro", labels=labels, zero_division=0)
        f1_micro = f1_score(y_true_i.numpy(), y_pred_i.numpy(), average="micro", labels=labels, zero_division=0)
        per_class = print_per_class_metrics(
            y_true_i.numpy(),
            y_pred_i.numpy(),
            labels=labels,
            prefix="[Test HGT]",
        )

        plot_confmatrix(y_true_i=y_true_i, y_pred_i=y_pred_i)

        results_dict = {
            "mae": test_mae,
            "rmse": test_rmse,
            "acc": test_acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "per_class": per_class,
        }

        results.append([test_mae, test_rmse, f1_micro])

        print(results_dict)

    results = np.concatenate(results)
    print(results)
    return results_dict


if __name__ == "__main__":
    print('Evaluating models...')
    mean_baseline()
    top_frequent_baseline()
    evaluate_model()
    print('Evaluating complete.')