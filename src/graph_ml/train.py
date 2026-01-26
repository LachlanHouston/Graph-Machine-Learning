import os
import copy
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path

import torch
import torch.nn.functional as F
from warnings import filterwarnings
from tqdm import tqdm
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import f1_score

from graph_ml import data
from graph_ml.data import load_yelp_as_hetero, split_edge_indices_by_year, check_uniform_edge_attr_dim
from graph_ml.model import HeteroHGTStarPredictor
from graph_ml.utils import get_n_params, set_seed, build_num_neighbors_from_cfg, multilabel_f1_from_logits, log_confmatrix, ordinal_targets, FocalLoss, EarlyStopping

try:
    import wandb
except ImportError:
    wandb = None

filterwarnings("ignore")
use_amp = False

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Pretty-print the composed config
    print(OmegaConf.to_yaml(cfg, resolve=True))
    set_seed(cfg.data.seed)

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
        max_friends_per_user=50,
        use_text_edge_attr=cfg.data.use_text_edge_attr,
    )

    print(data.metadata())

    edge_attr_dim = check_uniform_edge_attr_dim(data, rel, ("user", "friends", "user"))

    train_idx, val_idx = split_edge_indices_by_year(
        data, rel=rel, boundary_year=cfg.data.year_cutoff, include_boundary_in_train=True
    )

    full_edge_attr = data[rel].edge_attr

    # Seed edge pairs (positives only)
    train_pos = data[rel].edge_index[:, train_idx]
    val_pos   = data[rel].edge_index[:, val_idx]

    # Star labels 1..5 for those edges
    train_stars = data[rel].edge_label[train_idx]   # [P_train]
    val_stars   = data[rel].edge_label[val_idx]     # [P_val]

    years = data[rel].time.view(-1)
    train_years = years[train_idx]
    val_years   = years[val_idx]

    num_neighbors = build_num_neighbors_from_cfg(cfg)

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
        time_attr='time',
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

    num_users = data["user"].num_nodes
    num_businesses = data["business"].num_nodes

    model = HeteroHGTStarPredictor(
        metadata=data.metadata(),
        node_feat_dim=data['user'].x.size(1),
        num_users=num_users,
        num_businesses=num_businesses,
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

    print('Number of Parameters for Total model:', get_n_params(model))
    optimizer = torch.optim.AdamW(
        params = model.parameters(),
        lr=cfg.train.lr,
        eps=1e-8
    )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.train.lr,
        eps=1e-8
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",          # we minimize val_rmse
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

    run = None
    if cfg.wandb.enabled:
        if wandb is None:
            print("wandb not installed; set wandb.enabled=false or pip install wandb")
        else:
            mode = cfg.wandb.mode or os.environ.get('WANDB_MODE') or 'online'
            run = wandb.init(
                project=cfg.wandb.project,
                name=cfg.wandb.run_name,
                mode=mode,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            wandb.watch(model, log="gradients")

    loss_fn = FocalLoss(gamma=2.0)
    train_step = 0

    epoch_bar = tqdm(range(1, cfg.train.n_epoch + 1), desc="Epochs", leave=True)
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0

        y_true_stars = []
        y_pred_stars = []

        batch_bar = tqdm(train_loader, desc="Train batches", leave=False)
        for batch in batch_bar:
            batch = batch.to(device)
            y_train = batch[rel].edge_label.to(torch.long)
            y_train = (y_train - 1).clamp_(0, 4)
            optimizer.zero_grad(set_to_none=True)

            edge_label_attr = full_edge_attr[batch[rel].input_id].to(device, non_blocking=True) if batch[rel].edge_attr is not None else None

            out = model(
                batch.x_dict,
                batch.edge_index_dict,
                edge_label_index=batch[rel].edge_label_index,
                label_edge_type=rel,
                edge_label_attr=edge_label_attr,
                edge_label_time=getattr(batch[rel], "edge_label_time", None),
            )

            loss = loss_fn(out, y_train)

            pred_cls = out.argmax(dim=1)
            preds = 1 + pred_cls

            y_true_stars.append(y_train + 1)
            y_pred_stars.append(preds)

            loss.backward()
            if cfg.train.clip and cfg.train.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
            optimizer.step()

            total_loss += loss.item()

            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            if run is not None and train_step % 10 == 0:
                wandb.log({'train/loss': loss.item()}, step=train_step)
            train_step += 1

        batch_bar.close()
        avg_loss = total_loss / max(1, len(train_loader))

        y_true_i = torch.cat(y_true_stars).int()
        y_pred_i = torch.cat(y_pred_stars).int()
        train_acc = (y_true_i == y_pred_i).float().mean().item()

        epoch_bar.set_postfix({'train_acc': f'{train_acc:.4f}'})

        model.eval()
        y_true_stars = [] 
        y_pred_stars = []
        val_logits_list = []
        val_targets_list = []
        val_loss_sum = 0.0
        se_sum = 0.0
        n_sum = 0

        with torch.inference_mode():
            for batch in tqdm(val_loader, desc="Val", leave=False, dynamic_ncols=True):
                batch = batch.to(device, non_blocking=True)
                y = batch[rel].edge_label.to(torch.long)
                y = (y - 1).clamp_(0, 4)

                out = model(
                    batch.x_dict,
                    batch.edge_index_dict,
                    edge_label_index=batch[rel].edge_label_index,
                    label_edge_type=rel,
                edge_label_attr=full_edge_attr[batch[rel].input_id] if batch[rel].edge_attr is not None else None,
                    edge_label_time=getattr(batch[rel], "edge_label_time", None),
                )

                val_loss_sum += loss_fn(out, y).item()

                pred_cls = out.argmax(dim=1)
                preds = 1 + pred_cls

                se_sum += ((pred_cls - y)**2).sum().item()
                n_sum += y.numel()
                y_true_stars.append(y + 1)
                y_pred_stars.append(preds)

                val_logits_list.append(ordinal_targets(preds).detach().cpu())

                t = ordinal_targets(y)
                val_targets_list.append(t.detach().cpu())

        val_loss = val_loss_sum / max(1, len(val_loader))
        val_rmse = (se_sum / max(1, n_sum)) ** 0.5

        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]["lr"]

        y_true = torch.cat(y_true_stars).float()
        y_pred = torch.cat(y_pred_stars).float()
        y_true_i = y_true.clamp(1, 5).round().int().cpu()
        y_pred_i = y_pred.clamp(1, 5).round().int().cpu()
        val_acc = (y_true_i == y_pred_i).float().mean().item()

        if run is not None and (epoch % cfg.wandb.conf_matrix == 0):
            log_confmatrix(run, y_true_i, y_pred_i, step=train_step, normalize=None, title="Val Confusion Matrix")

        f1_macro = f1_score(y_true_i.numpy(), y_pred_i.numpy(), average="macro")
        f1_micro = f1_score(y_true_i.numpy(), y_pred_i.numpy(), average="micro")

        if run is not None:
            wandb.log({
                'epoch': epoch,
                'lr': current_lr,
                'train/loss': avg_loss,
                'train/acc': train_acc,
                'val/loss': val_loss,
                'val/rmse': val_rmse,
                'val_acc': val_acc,
                'val/f1_micro': f1_micro,
                'val/f1_macro': f1_macro,
            }, step=train_step)

        epoch_bar.set_postfix({
            'train_loss': f'{avg_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_rmse': f'{val_rmse:.4f}',
            'val_acc': f'{val_acc:.4f}',
            'f1_micro': f'{f1_micro:.4f}',
            'f1_macro': f'{f1_macro:.4f}',
            'lr': f'{current_lr:.2e}',
        })

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered...")
            early_stopper.save_best_model(cfg.train.model_path)
            break

    if run is not None:
        run.finish()

if __name__ == "__main__":
    main()