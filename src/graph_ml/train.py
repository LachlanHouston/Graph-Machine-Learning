import os
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path

import torch
import torch.nn.functional as F
from warnings import filterwarnings
from tqdm import tqdm
from torch_geometric.loader import LinkNeighborLoader

from graph_ml.data import load_yelp_as_hetero, split_edge_indices_by_year
from graph_ml.model import HGTStarPredictor
from graph_ml.utils import get_n_params, set_seed

try:
    import wandb
except ImportError:
    wandb = None

filterwarnings("ignore")
use_amp = False

def ordinal_targets(y):
    B = y.size(0)
    k = torch.arange(1, 5, device=y.device).unsqueeze(0).expand(B, -1)
    return (y.unsqueeze(1) > k).float()

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Pretty-print the composed config
    print(OmegaConf.to_yaml(cfg, resolve=True))

    set_seed(cfg.data.seed)

    # Resolve any file system paths that should be relative to original CWD
    data_dir = to_absolute_path(cfg.data.data_dir)

    device = torch.device(f"cuda:{cfg.data.cuda}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    rel = ("user", "reviews", "business")

    data = load_yelp_as_hetero(
        data_dir=data_dir,
        max_reviews=cfg.data.max_reviews,
        min_review_len=cfg.data.min_review_len,
        seed=cfg.data.seed,
        cache=True,
        cache_subdir="processed",
        include_user_friends=True,
        max_friends_per_user=50,
        use_text_edge_attr=cfg.data.use_text_edge_attr,
    )

    print(data.metadata())

    for nt in data.node_types:
            if not hasattr(data[nt], "x"):
                data[nt].x = torch.arange(data[nt].num_nodes)

    num_nodes = {nt: data[nt].num_nodes for nt in data.node_types}

    train_idx, val_idx = split_edge_indices_by_year(
        data, rel=rel, boundary_year=2017, include_boundary_in_train=True
    )

    num_neighbors = {
        ("user","reviews","business"):   [20, 15, 10, 5],
        ("business","rev_reviews","user"): [6, 3, 0, 0],
        ("user","friends","user"):       [6, 4, 2, 0],
    }

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
    )

    stars = data[rel].edge_label
    years = data[rel].time.squeeze()
    train_label = stars[train_idx]
    val_label = stars[val_idx]

    train_loader = LinkNeighborLoader(
        **common_kwargs,
        edge_label_index=(rel, data[rel].edge_index[:, train_idx]),
        edge_label=train_label,
        edge_label_time=years[train_idx],
        time_attr="time",
    )

    val_loader = LinkNeighborLoader(
        **{**common_kwargs, "shuffle": False},
        edge_label_index=(rel, data[rel].edge_index[:, val_idx]),
        edge_label=val_label,
        edge_label_time=years[val_idx],
        time_attr="time",
    )

    edge_attr_dim = int(data[rel].edge_attr.size(-1)) if hasattr(data[rel], "edge_attr") else None

    model = HGTStarPredictor(
        metadata=data.metadata(),
        num_nodes=num_nodes,
        hidden_dim=cfg.model.n_hid,
        num_layers=cfg.model.n_layers,
        num_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        edge_attr_dim=edge_attr_dim,
        out_mode="ordinal",
    ).to(device)

    print('Number of Parameters for Total model:', get_n_params(model))
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer = torch.optim.AdamW(
        [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': cfg.train.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ],
        lr=cfg.train.lr,
        eps=1e-8
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

    y_train_all = data[rel].edge_label[train_idx].to(torch.long)
    with torch.no_grad():
        pos = torch.stack([(y_train_all > k).float().mean() for k in range(1, 5)])
    pos_w = ((1 - pos) / (pos + 1e-6)).to(device)

    best_val = float("inf")
    train_step = 0

    epoch_bar = tqdm(range(1, cfg.train.n_epoch + 1), desc="Epochs", leave=True)
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        steps = 0

        batch_bar = tqdm(total=cfg.train.n_batch, desc="Train batches", leave=False)
        for steps, batch in zip(range(cfg.train.n_batch), train_loader):
            batch = batch.to(device)
            y_train = batch[rel].edge_label.float()
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(
                    batch.edge_index_dict,
                    edge_label_index=batch[rel].edge_label_index,
                    label_edge_type=rel,
                    batch=batch,
                )
                t = ordinal_targets(y_train)
                loss = F.binary_cross_entropy_with_logits(out, t, pos_weight=pos_w)

            loss.backward()
            if cfg.train.clip and cfg.train.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
            optimizer.step()

            total_loss += loss.item()
            batch_bar.update(1)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            if run is not None and train_step % 10 == 0:
                wandb.log({'train/loss': loss.item()}, step=train_step)
            train_step += 1

        batch_bar.close()
        avg_loss = total_loss / max(1, steps + 1)

        model.eval()
        y_true, y_pred = [], []
        val_loss_sum = 0.0
        se_sum = 0.0
        n_sum = 0

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
            for batch in tqdm(val_loader, desc="Val", leave=False, dynamic_ncols=True):
                batch = batch.to(device, non_blocking=True)
                y = batch[rel].edge_label.float()
                out = model(
                    batch.edge_index_dict,
                    edge_label_index=batch[rel].edge_label_index,
                    label_edge_type=rel,
                    batch=batch,
                )
                t = ordinal_targets(y)
                val_loss_sum += F.binary_cross_entropy_with_logits(out, t, pos_weight=pos_w).item()
                probs = torch.sigmoid(out)
                preds = 1 + (probs > 0.5).sum(dim=1)
                se_sum += ((preds - y)**2).sum().item()
                n_sum += y.numel()
                y_true.append(y)
                y_pred.append(preds)

        val_loss = val_loss_sum / max(1, len(val_loader))
        val_rmse = (se_sum / max(1, n_sum)) ** 0.5
        y_true = torch.cat(y_true).float()
        y_pred = torch.cat(y_pred).float()
        y_true_i = y_true.clamp(1, 5).round().int().cpu()
        y_pred_i = y_pred.clamp(1, 5).round().int().cpu()
        val_acc = (y_true_i == y_pred_i).float().mean().item()

        if run is not None:
            wandb.log({'epoch': epoch, 'train/loss': avg_loss, 'val/loss': val_loss, 'val/rmse': val_rmse, 'val_acc': val_acc}, step=train_step)

        if val_rmse < best_val:
            best_val = val_rmse
            os.makedirs(os.path.dirname(cfg.train.model_path), exist_ok=True)
            torch.save(model.state_dict(), cfg.train.model_path)

        epoch_bar.set_postfix({'train_loss': f'{avg_loss:.4f}', 'val_loss': f'{val_loss:.4f}', 'val_rmse': f'{val_rmse:.4f}', 'val_acc': val_acc})
        print(y_true_i.tolist()[:20])
        print(y_pred_i.tolist()[:20])

    if run is not None:
        class_labels = ['1 star', '2 star', '3 star', '4 star', '5 star']
        y_true_idx = torch.clamp(y_true_i - 1, 0, 4).tolist()
        y_pred_idx = torch.clamp(y_pred_i - 1, 0, 4).tolist()
        wandb.log({
            "my_conf_mat_id": wandb.plot.confusion_matrix(
                probs=None,
                preds=y_pred_idx,
                y_true=y_true_idx,
                class_names=class_labels
            )
        }, step=train_step)
        run.finish()

if __name__ == "__main__":
    main()