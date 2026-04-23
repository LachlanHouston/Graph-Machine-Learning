import os
import math
from pathlib import Path
from warnings import filterwarnings

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm

from graph_ml.data import (
    check_uniform_edge_attr_dim,
    load_yelp_as_hetero,
    split_edge_indices_by_year,
)
from graph_ml.model import HeteroHGTStarPredictor
from graph_ml.utils import (
    EarlyStopping,
    FocalLoss,
    build_num_neighbors_from_cfg,
    get_n_params,
    log_confmatrix,
    set_seed,
)

try:
    import wandb
except ImportError:
    wandb = None

filterwarnings("ignore")


def build_model(cfg: DictConfig, data, edge_attr_dim: int, device: torch.device):
    return HeteroHGTStarPredictor(
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


def make_loaders(cfg: DictConfig, data, rel, train_pos, val_pos, train_stars, val_stars, train_years, val_years, device):
    num_neighbors = build_num_neighbors_from_cfg(cfg)
    num_workers = max(1, os.cpu_count() - 1) if device.type == "cuda" else 0
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

    train_loader = LinkNeighborLoader(
        **common_kwargs,
        edge_label_index=(rel, train_pos),
        edge_label=train_stars,
        edge_label_time=train_years,
    )

    val_loader = LinkNeighborLoader(
        **{**common_kwargs, "shuffle": False},
        edge_label_index=(rel, val_pos),
        edge_label=val_stars,
        edge_label_time=val_years,
    )

    return train_loader, val_loader, common_kwargs


@torch.inference_mode()
def eval_single(model, loader, rel, edge_attr, loss_fn, device, log_conf=False, run=None, step=0, conf_title="Confusion Matrix"):
    model.eval()
    y_true_stars, y_pred_stars = [], []
    val_loss_sum = 0.0
    se_sum = 0.0
    n_sum = 0

    for batch in tqdm(loader, desc="Val", leave=False, dynamic_ncols=True):
        batch = batch.to(device, non_blocking=True)
        y = batch[rel].edge_label.to(torch.long)
        y = (y - 1).clamp_(0, 4)

        out = model(
            batch.x_dict,
            batch.edge_index_dict,
            edge_label_index=batch[rel].edge_label_index,
            label_edge_type=rel,
            edge_label_attr=edge_attr[batch[rel].input_id] if batch[rel].edge_attr is not None else None,
            edge_label_time=getattr(batch[rel], "edge_label_time", None),
        )

        val_loss_sum += loss_fn(out, y).item()
        pred_cls = out.argmax(dim=1)

        se_sum += ((pred_cls - y) ** 2).sum().item()
        n_sum += y.numel()

        y_true_stars.append((y + 1).detach().cpu())
        y_pred_stars.append((pred_cls + 1).detach().cpu())
        break

    val_loss = val_loss_sum / max(1, len(loader))
    val_rmse = (se_sum / max(1, n_sum)) ** 0.5

    y_true = torch.cat(y_true_stars).float()
    y_pred = torch.cat(y_pred_stars).float()
    y_true_i = y_true.clamp(1, 5).round().int()
    y_pred_i = y_pred.clamp(1, 5).round().int()

    val_acc = (y_true_i == y_pred_i).float().mean().item()
    val_mae = torch.abs(y_true - y_pred).mean().item()

    f1_macro = f1_score(y_true_i.numpy(), y_pred_i.numpy(), average="macro")
    f1_micro = f1_score(y_true_i.numpy(), y_pred_i.numpy(), average="micro")

    if log_conf and (run is not None):
        log_confmatrix(run, y_true_i, y_pred_i, step=step, normalize=None, title=conf_title)

    return {
        "val_loss": val_loss,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_acc": val_acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "y_true_i": y_true_i,
        "y_pred_i": y_pred_i,
    }


@torch.inference_mode()
def eval_ensemble(models, loader, rel, edge_attr, device, eps=1e-12):
    for m in models:
        m.eval()

    all_y = []
    all_pred = []
    all_mean_rating = []
    all_std_rating = []
    all_entropy = []
    all_mi = []

    se_sum = 0.0
    mae_sum = 0.0
    n_sum = 0

    stars = torch.arange(1, 6, device=device, dtype=torch.float32)

    for batch in tqdm(loader, desc="Eval (ensemble+unc)", leave=False, dynamic_ncols=True):
        batch = batch.to(device, non_blocking=True)
        y = batch[rel].edge_label.to(torch.long)
        y = (y - 1).clamp_(0, 4)

        probs = []
        entropies = []
        ratings = []

        for m in models:
            logits = m(
                batch.x_dict,
                batch.edge_index_dict,
                edge_label_index=batch[rel].edge_label_index,
                label_edge_type=rel,
                edge_label_attr=edge_attr[batch[rel].input_id] if batch[rel].edge_attr is not None else None,
                edge_label_time=getattr(batch[rel], "edge_label_time", None),
            )

            p = torch.softmax(logits, dim=1)
            probs.append(p)
            entropies.append(-(p * (p + eps).log()).sum(dim=1))
            ratings.append((p * stars).sum(dim=1))

        P = torch.stack(probs, dim=0)
        Hm = torch.stack(entropies, dim=0)
        Rm = torch.stack(ratings, dim=0)

        p_bar = P.mean(dim=0)
        pred_cls = p_bar.argmax(dim=1)

        mean_rating = Rm.mean(dim=0)
        std_rating = Rm.std(dim=0, unbiased=True) if Rm.size(0) > 1 else torch.zeros_like(mean_rating)

        H_bar = -(p_bar * (p_bar + eps).log()).sum(dim=1)
        mi = H_bar - Hm.mean(dim=0)

        y_stars = (y + 1).float()
        pred_stars = (pred_cls + 1).float()

        se_sum += ((pred_stars - y_stars) ** 2).sum().item()
        mae_sum += (pred_stars - y_stars).abs().sum().item()
        n_sum += y.numel()

        all_y.append((y + 1).detach().cpu())
        all_pred.append((pred_cls + 1).detach().cpu())
        all_mean_rating.append(mean_rating.detach().cpu())
        all_std_rating.append(std_rating.detach().cpu())
        all_entropy.append(H_bar.detach().cpu())
        all_mi.append(mi.detach().cpu())
        break

    y_all = torch.cat(all_y).int()
    pred_all = torch.cat(all_pred).int()

    mean_rating_all = torch.cat(all_mean_rating).float()
    std_rating_all = torch.cat(all_std_rating).float()
    entropy_all = torch.cat(all_entropy).float()
    mi_all = torch.cat(all_mi).float()

    rmse = math.sqrt(se_sum / max(1, n_sum))
    mae = mae_sum / max(1, n_sum)
    acc = (y_all == pred_all).float().mean().item()
    f1_macro = f1_score(y_all.numpy(), pred_all.numpy(), average="macro")
    f1_micro = f1_score(y_all.numpy(), pred_all.numpy(), average="micro")

    unc_summary = {
        "std_rating_mean": std_rating_all.mean().item(),
        "std_rating_p90": std_rating_all.quantile(0.90).item(),
        "entropy_mean": entropy_all.mean().item(),
        "entropy_p90": entropy_all.quantile(0.90).item(),
        "mi_mean": mi_all.mean().item(),
        "mi_p90": mi_all.quantile(0.90).item(),
    }

    return {
        "rmse": rmse,
        "mae": mae,
        "acc": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "y": y_all,
        "pred": pred_all,
        "mean_rating": mean_rating_all,
        "std_rating": std_rating_all,
        "entropy": entropy_all,
        "mutual_info": mi_all,
        "unc_summary": unc_summary,
    }


def train_one(
    cfg: DictConfig,
    seed: int,
    data,
    rel,
    edge_attr_dim: int,
    train_edge_attr,
    val_edge_attr,
    device: torch.device,
    model_save_path: str,
    wandb_run=None,
):
    set_seed(seed)
    model = build_model(cfg, data, edge_attr_dim, device)
    print("Number of Parameters for Total model:", get_n_params(model))

    train_idx, val_idx, _ = split_edge_indices_by_year(
        data, rel=rel, boundary_year=cfg.data.year_cutoff, include_boundary_in_train=True, seed=seed
    )

    frac = 0.85
    perm = torch.randperm(train_idx.numel())
    sub = perm[: int(frac * train_idx.numel())]
    train_idx = train_idx[sub]

    train_edge_attr = data[rel].edge_attr[train_idx] if data[rel].edge_attr is not None else None
    val_edge_attr = data[rel].edge_attr[val_idx] if data[rel].edge_attr is not None else None

    train_edge_attr = train_edge_attr.to(device, non_blocking=True) if train_edge_attr is not None else None
    val_edge_attr = val_edge_attr.to(device, non_blocking=True) if val_edge_attr is not None else None

    train_pos = data[rel].edge_index[:, train_idx]
    val_pos = data[rel].edge_index[:, val_idx]

    train_stars = data[rel].edge_label[train_idx]
    val_stars = data[rel].edge_label[val_idx]

    years = data[rel].time.view(-1)
    train_years = years[train_idx]
    val_years = years[val_idx]

    train_loader, val_loader, _ = make_loaders(
        cfg=cfg,
        data=data,
        rel=rel,
        train_pos=train_pos,
        val_pos=val_pos,
        train_stars=train_stars,
        val_stars=val_stars,
        train_years=train_years,
        val_years=val_years,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, eps=1e-8)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-3,
        threshold_mode="rel",
        cooldown=0,
        min_lr=cfg.train.lr_scheduler_min,
    )

    early_stopper = EarlyStopping(
        patience=cfg.train.early_stopping_patience,
        delta=cfg.train.early_stopping_delta,
    )

    loss_fn = FocalLoss(gamma=2.0)
    train_step = 0

    epoch_bar = tqdm(range(1, cfg.train.n_epoch + 1), desc=f"Epochs (seed={seed})", leave=True)
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        y_true_stars, y_pred_stars = [], []

        batch_bar = tqdm(train_loader, desc="Train batches", leave=False, dynamic_ncols=True)
        for batch in batch_bar:
            batch = batch.to(device)
            y_train = batch[rel].edge_label.to(torch.long)
            y_train = (y_train - 1).clamp_(0, 4)

            optimizer.zero_grad(set_to_none=True)
            out = model(
                batch.x_dict,
                batch.edge_index_dict,
                edge_label_index=batch[rel].edge_label_index,
                label_edge_type=rel,
                edge_label_attr=train_edge_attr[batch[rel].input_id] if batch[rel].edge_attr is not None else None,
                edge_label_time=getattr(batch[rel], "edge_label_time", None),
            )

            loss = loss_fn(out, y_train)
            pred_cls = out.argmax(dim=1)
            preds = 1 + pred_cls

            y_true_stars.append((y_train + 1).detach())
            y_pred_stars.append(preds.detach())

            loss.backward()
            if cfg.train.clip and cfg.train.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
            optimizer.step()

            total_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            if wandb_run is not None and (train_step % 10 == 0):
                wandb.log({"train/loss": loss.item()}, step=train_step)
            train_step += 1
            break

        avg_loss = total_loss / max(1, len(train_loader))
        y_true_i = torch.cat(y_true_stars).int().cpu()
        y_pred_i = torch.cat(y_pred_stars).int().cpu()
        train_acc = (y_true_i == y_pred_i).float().mean().item()

        log_conf = bool(getattr(cfg, "wandb", None) and cfg.wandb.enabled and (wandb_run is not None) and (epoch % cfg.wandb.conf_matrix == 0))
        metrics = eval_single(
            model=model,
            loader=val_loader,
            rel=rel,
            edge_attr=val_edge_attr,
            loss_fn=loss_fn,
            device=device,
            log_conf=log_conf,
            run=wandb_run,
            step=train_step,
            conf_title=f"Val Confusion Matrix (seed={seed})",
        )

        scheduler.step(metrics["val_rmse"])
        current_lr = optimizer.param_groups[0]["lr"]

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "lr": current_lr,
                    "train/loss": avg_loss,
                    "train/acc": train_acc,
                    "val/loss": metrics["val_loss"],
                    "val/mae": metrics["val_mae"],
                    "val/rmse": metrics["val_rmse"],
                    "val_acc": metrics["val_acc"],
                    "val/f1_micro": metrics["f1_micro"],
                    "val/f1_macro": metrics["f1_macro"],
                },
                step=train_step,
            )

        epoch_bar.set_postfix(
            {
                "train_loss": f"{avg_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
                "val_loss": f"{metrics['val_loss']:.4f}",
                "val_rmse": f"{metrics['val_rmse']:.4f}",
                "val_mae": f"{metrics['val_mae']:.4f}",
                "val_acc": f"{metrics['val_acc']:.4f}",
                "f1_micro": f"{metrics['f1_micro']:.4f}",
                "f1_macro": f"{metrics['f1_macro']:.4f}",
                "lr": f"{current_lr:.2e}",
            }
        )

        early_stopper(metrics["val_loss"], model)
        if early_stopper.early_stop:
            early_stopper.save_best_model(model_save_path)
            break

    if not early_stopper.early_stop:
        early_stopper.save_best_model(model_save_path)

    return model_save_path


def load_model_from_ckpt(cfg: DictConfig, data, edge_attr_dim: int, device: torch.device, ckpt_path: str):
    m = build_model(cfg, data, edge_attr_dim, device)
    state = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(state)
    m.eval()
    return m


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    data_dir = to_absolute_path(cfg.data.data_dir)
    device = torch.device(f"cuda:{cfg.data.cuda}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    rel = ("user", "reviews", "business")

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
    )

    print(data.metadata())

    edge_attr_dim = check_uniform_edge_attr_dim(data, rel, ("user", "friends", "user"))

    train_idx, val_idx, test_idx = split_edge_indices_by_year(
        data, rel=rel, boundary_year=cfg.data.year_cutoff, include_boundary_in_train=True
    )

    train_edge_attr = data[rel].edge_attr[train_idx] if data[rel].edge_attr is not None else None
    val_edge_attr = data[rel].edge_attr[val_idx] if data[rel].edge_attr is not None else None
    test_edge_attr = data[rel].edge_attr[test_idx] if data[rel].edge_attr is not None else None

    train_edge_attr = train_edge_attr.to(device, non_blocking=True) if train_edge_attr is not None else None
    val_edge_attr = val_edge_attr.to(device, non_blocking=True) if val_edge_attr is not None else None
    test_edge_attr = test_edge_attr.to(device, non_blocking=True) if test_edge_attr is not None else None

    train_pos = data[rel].edge_index[:, train_idx]
    val_pos = data[rel].edge_index[:, val_idx]
    test_pos = data[rel].edge_index[:, test_idx]

    train_stars = data[rel].edge_label[train_idx]
    val_stars = data[rel].edge_label[val_idx]
    test_stars = data[rel].edge_label[test_idx]

    years = data[rel].time.view(-1)
    train_years = years[train_idx]
    val_years = years[val_idx]
    test_years = years[test_idx]

    _, val_loader, common_kwargs = make_loaders(
        cfg=cfg,
        data=data,
        rel=rel,
        train_pos=train_pos,
        val_pos=val_pos,
        train_stars=train_stars,
        val_stars=val_stars,
        train_years=train_years,
        val_years=val_years,
        device=device,
    )

    test_loader = LinkNeighborLoader(
        **{**common_kwargs, "shuffle": False},
        edge_label_index=(rel, test_pos),
        edge_label=test_stars,
        edge_label_time=test_years,
    )

    use_ensemble = bool(getattr(cfg, "ensemble", None) and cfg.ensemble.get("enabled", False))

    if not use_ensemble:
        set_seed(cfg.data.seed)
        run = None
        if cfg.wandb.enabled:
            if wandb is None:
                print("wandb not installed; set wandb.enabled=false or pip install wandb")
            else:
                mode = cfg.wandb.mode or os.environ.get("WANDB_MODE") or "online"
                run = wandb.init(
                    project=cfg.wandb.project,
                    name=cfg.wandb.run_name,
                    mode=mode,
                    config=OmegaConf.to_container(cfg, resolve=True),
                )

        model_save_path = to_absolute_path(cfg.train.model_path)
        ckpt = train_one(
            cfg=cfg,
            seed=cfg.data.seed,
            data=data,
            rel=rel,
            edge_attr_dim=edge_attr_dim,
            train_edge_attr=train_edge_attr,
            val_edge_attr=val_edge_attr,
            device=device,
            model_save_path=model_save_path,
            wandb_run=run,
        )

        if run is not None:
            run.finish()
        print("Saved best model to:", ckpt)
        return

    save_dir = Path(to_absolute_path(cfg.ensemble.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(cfg.data.seed) + int(getattr(cfg.ensemble, "seed_offset", 0))
    n_models = int(cfg.ensemble.n_models)

    ckpts = []
    for i in range(n_models):
        seed_i = base_seed + i

        run = None
        if cfg.wandb.enabled:
            if wandb is None:
                print("wandb not installed; set wandb.enabled=false or pip install wandb")
            else:
                mode = cfg.wandb.mode or os.environ.get("WANDB_MODE") or "online"
                name = f"{cfg.wandb.run_name}-seed{seed_i}" if cfg.wandb.run_name else f"seed{seed_i}"
                run = wandb.init(
                    project=cfg.wandb.project,
                    name=name,
                    mode=mode,
                    config=OmegaConf.to_container(cfg, resolve=True),
                )

        ckpt_path = str(save_dir / f"model_seed{seed_i}.pt")
        ckpt = train_one(
            cfg=cfg,
            seed=seed_i,
            data=data,
            rel=rel,
            edge_attr_dim=edge_attr_dim,
            train_edge_attr=train_edge_attr,
            val_edge_attr=val_edge_attr,
            device=device,
            model_save_path=ckpt_path,
            wandb_run=run,
        )
        ckpts.append(ckpt)

        if run is not None:
            run.finish()

    models = [load_model_from_ckpt(cfg, data, edge_attr_dim, device, p) for p in ckpts]

    val_metrics = eval_ensemble(models, val_loader, rel, val_edge_attr, device)
    print(
        f"Ensemble (val) rmse={val_metrics['rmse']:.4f} mae={val_metrics['mae']:.4f} "
        f"acc={val_metrics['acc']:.4f} f1_micro={val_metrics['f1_micro']:.4f} f1_macro={val_metrics['f1_macro']:.4f}"
    )
    print("Ensemble (val) uncertainty summary:", val_metrics["unc_summary"])

    test_metrics = eval_ensemble(models, test_loader, rel, test_edge_attr, device)
    print(
        f"Ensemble (test) rmse={test_metrics['rmse']:.4f} mae={test_metrics['mae']:.4f} "
        f"acc={test_metrics['acc']:.4f} f1_micro={test_metrics['f1_micro']:.4f} f1_macro={test_metrics['f1_macro']:.4f}"
    )
    print("Ensemble (test) uncertainty summary:", test_metrics["unc_summary"])

if __name__ == "__main__":
    print("Starting training...")
    main()
    print("Training complete.")