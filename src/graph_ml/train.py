import os
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path

import torch
import torch.nn.functional as F
from warnings import filterwarnings
from tqdm import tqdm
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import f1_score

from graph_ml.data import load_yelp_as_hetero, split_edge_indices_by_year, check_uniform_edge_attr_dim
from graph_ml.model import HeteroHGTStarPredictor
from graph_ml.utils import get_n_params, set_seed, build_num_neighbors_from_cfg, multilabel_f1_from_logits, log_confmatrix

try:
    import wandb
except ImportError:
    wandb = None

filterwarnings("ignore")
use_amp = False

class FocalLoss(torch.nn.Module):
    """Implementation of the Focal loss function

        Args:
            weight: class weight vector to be used in case of class imbalance
            gamma: hyper-parameter for the focal loss scaling.
    """
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, outputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none', weight=self.weight) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1-pt)**self.gamma * ce_loss).mean() # mean over the batch
        return focal_loss

def ordinal_targets(y):
    B = y.size(0)
    k = torch.arange(1, 5, device=y.device).unsqueeze(0).expand(B, -1)
    return (y.unsqueeze(1) > k).float()

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
        min_review_len=cfg.data.min_review_len,
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
        data, rel=rel, boundary_year=2017, include_boundary_in_train=True
    )

    train_edge_attr = data[rel].edge_attr[train_idx]  # shape [P_train, 305]
    val_edge_attr   = data[rel].edge_attr[val_idx]    # shape [P_val, 305]

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

    model = HeteroHGTStarPredictor(
        metadata=data.metadata(),
        node_feat_dim=data['user'].x.size(1),
        hidden_dim=cfg.model.n_hid,
        num_layers=cfg.model.n_layers,
        num_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        num_classes=5,
        edge_attr_dim=edge_attr_dim,
        time_out=16,
    ).to(device)

    print('Number of Parameters for Total model:', get_n_params(model))
    optimizer = torch.optim.AdamW(
        params = model.parameters(),
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

    loss_fn = FocalLoss()

    best_val = float("inf")
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
                    edge_label_attr=val_edge_attr[batch[rel].input_id] if batch[rel].edge_attr is not None else None,
                    edge_label_time=getattr(batch[rel], "edge_label_time", None),
                )

                val_loss_sum += loss_fn(out, y).item()

                pred_cls = out.argmax(dim=1)
                preds = 1 + pred_cls

                se_sum += ((preds - y)**2).sum().item()
                n_sum += y.numel()
                y_true_stars.append(y + 1)
                y_pred_stars.append(preds)

                val_logits_list.append(ordinal_targets(preds).detach().cpu())

                t = ordinal_targets(y)
                val_targets_list.append(t.detach().cpu())

        val_loss = val_loss_sum / max(1, len(val_loader))
        val_rmse = (se_sum / max(1, n_sum)) ** 0.5

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
            wandb.log({'epoch': epoch, 'train/loss': avg_loss, 'train/acc': train_acc, 'val/loss': val_loss, 'val/rmse': val_rmse, 'val_acc': val_acc,
                       'val/f1_micro': f1_micro, 'val/f1_macro': f1_macro}, step=train_step)

        if val_rmse < best_val:
            print(f'Saving new best model with val_acc: {val_acc:.4f} and val_rmse: {val_rmse:.4f}')
            best_val = val_rmse
            os.makedirs(os.path.dirname(cfg.train.model_path), exist_ok=True)
            torch.save(model.state_dict(), cfg.train.model_path)

        epoch_bar.set_postfix({'train_loss': f'{avg_loss:.4f}', 'val_loss': f'{val_loss:.4f}', 'val_rmse': f'{val_rmse:.4f}', 'val_acc': f'{val_acc:.4f}',
                               'f1_micro': f'{f1_micro:.4f}', 'f1_macro': f'{f1_macro:.4f}'})

    if run is not None:
        run.finish()

if __name__ == "__main__":
    main()