import os
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from warnings import filterwarnings
filterwarnings("ignore")

from tqdm import tqdm
from torch_geometric.loader import LinkNeighborLoader

# Package modules
from graph_ml.data import load_yelp_as_hetero, split_edge_indices, split_edge_indices_by_year
from graph_ml.model import EdgeHGT, EdgeClassifier
from graph_ml.utils import args_print, get_n_params, set_seed

# WandB
try:
    import wandb
except ImportError:
    wandb = None
        
def main():
    parser = argparse.ArgumentParser(description='Training GNN')

    # System
    parser.add_argument('--data_dir', type=str, default='data/raw/',
                        help='The address of preprocessed graph.')
    parser.add_argument('--model_dir', type=str, default=f'models/run_{int(time.time())}.pt',
                        help='The address for storing the trained models.')
    parser.add_argument('--cuda', type=int, default=0, help='Available GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Data
    parser.add_argument("--max_reviews", type=int, default=250_000)
    parser.add_argument("--min_review_len", type=int, default=5)
    parser.add_argument("--use_text_edge_attr", action="store_true", default=False)
    parser.add_argument("--tfidf_max_tokens", type=int, default=5000)
    parser.add_argument("--svd_dim", type=int, default=64)
    parser.add_argument("--sample_depth", type=int, default=4, help="Number of hops")
    parser.add_argument("--sample_width", type=int, default=200, help="Neighbors per hop")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")

    # Model
    parser.add_argument('--n_hid', type=int, default=512, help='Hidden dim')
    parser.add_argument('--n_heads', type=int, default=8, help='Attention heads')
    parser.add_argument('--n_layers', type=int, default=4, help='GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout ratio')
    parser.add_argument('--n_epoch', type=int, default=3, help='Epochs')
    parser.add_argument('--n_batch', type=int, default=32, help='Batches (sampled graphs) per epoch')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient norm clipping')

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')

    # WandB
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='Graph Machine Learning', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb_mode', type=str, default=None, choices=[None, 'online', 'offline'],
                        help='wandb mode (default: env/WANDB_MODE or online)')
    
    args = parser.parse_args()
    args_print(args)
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    rel = ("user", "reviews", "business")

    data = load_yelp_as_hetero(args.data_dir, max_reviews=args.max_reviews, 
                               min_review_len=args.min_review_len, seed=args.seed, 
                               cache=True, cache_subdir="processed", 
                               include_user_friends=True, max_friends_per_user=50)
    
    num_classes = 1
    
    train_idx, val_idx = split_edge_indices_by_year(
        data, rel=rel, boundary_year=2017, include_boundary_in_train=True
    )


    num_neighbors = {
        ("user","reviews","business"): [16,16],
        ("business","rev_reviews","user"): [16,16],
        ("user","friends","user"): [8,8],   # much smaller
    }

    num_workers = max(1, os.cpu_count() - 1) if device.type == "cuda" else 0
    prefetch_factor = 2 if device.type == "cuda" else None
    common_kwargs = dict(
        data=data,
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
    )

    stars = data[rel].edge_label           # [E]
    years = data[rel].time.squeeze()

    train_label = stars[train_idx] # [Ntr, 2]
    val_label   = stars[val_idx]

    train_loader = LinkNeighborLoader(
        **common_kwargs,
        edge_label_index=(rel, data[rel].edge_index[:, train_idx]),
        edge_label= train_label,
        edge_label_time=years[train_idx],
        time_attr="time"
    )

    val_loader = LinkNeighborLoader(
        **{**common_kwargs, "shuffle": False},
        edge_label_index=(rel, data[rel].edge_index[:, val_idx]),
        edge_label= val_label,
        edge_label_time=years[val_idx],
        time_attr="time"
    )

    gnn = EdgeHGT(metadata=data.metadata(),
                          hidden_dim=args.n_hid,
                          num_layers=args.n_layers,
                          heads=args.n_heads,
                          dropout=args.dropout).to(device)
    
    gnn.set_num_nodes({
        "user": int(data["user"].num_nodes),
        "business": int(data["business"].num_nodes),
    })
    
    classifier = EdgeClassifier(hidden_dim=args.n_hid, num_classes=num_classes).to(device)

    model = torch.nn.Sequential(gnn, classifier).to(device)

    criterion = torch.nn.L1Loss()

    print('Number of Parameters for Total model: %d' % get_n_params(model))
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer = torch.optim.Adam(
        [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ],
        lr=args.lr,
        eps=1e-8
    )
    
    run = None
    if args.wandb:
        if wandb is None:
            print("wandb not installed; run `pip install wandb` or disable --wandb")
        else:
            mode = args.wandb_mode or os.environ.get('WANDB_MODE') or 'online'
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode=mode,
                config=vars(args),
            )
            wandb.watch(model, log="gradients")

    best_val = float("inf")
    train_step = 0

    # Training loop
    epoch_bar = tqdm(range(1, args.n_epoch + 1), desc="Epochs", leave=True)
    for epoch in epoch_bar:
        # Training
        model.train()
        total_loss = 0.0
        steps = 0

        batch_bar = tqdm(total=args.n_batch, desc="Train batches", leave=False)
        for steps, batch in zip(range(args.n_batch), train_loader):
            batch = batch.to(device)
            y = batch[rel].edge_label.float()     # target stars

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(batch).view(-1)
                loss = criterion(preds, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_bar.update(1)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            if run is not None and train_step % 10 == 0:
                wandb.log({'train/loss': loss.item()}, step=train_step)
            train_step += 1

            break

        batch_bar.close()
        avg_loss = total_loss / max(1, steps + 1)

        # Validation
        model.eval()
        y_true, y_pred = [], []
        val_loss = 0.0
        val_rmse = 0.0
        steps = 0

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            for batch in tqdm(val_loader, desc="Val", leave=False, dynamic_ncols=True):
                batch = batch.to(device, non_blocking=True)
                y = batch[rel].edge_label.float()     # target stars

                preds = model(batch).view(-1)

                val_loss += criterion(preds, y).item()
                val_rmse += torch.sqrt(F.mse_loss(preds, y)).item()
                steps += 1

                y_true.append(y)
                y_pred.append(preds)

        val_loss /= max(1, steps)
        val_rmse /= max(1, steps)

        y_true = torch.cat(y_true).float()
        y_pred = torch.cat(y_pred).float()

        # Ensure integers on CPU for metrics/logging
        y_true_i = y_true.round().long().cpu()
        y_pred_i = y_pred.clamp(1, 5).round().long().cpu()

        if val_rmse < best_val:
            best_val = val_rmse
            os.makedirs(os.path.dirname(args.model_dir), exist_ok=True)
            torch.save(model.state_dict(), args.model_dir)

        epoch_bar.set_postfix({'train_loss': f'{avg_loss:.4f}', 'val_loss': f'{val_loss:.4f}', 'val_rmse': f'{val_rmse:.4f}'})

        if run is not None:
            cm_plot = wandb.plot.confusion_matrix(
                preds=y_pred_i.tolist(),
                y_true=y_true_i.tolist(),
            )

            wandb.log({'epoch': epoch, 'train/loss': avg_loss, 'val/loss': val_loss, 'val/rmse': val_rmse, "val/confusion_matrix": cm_plot}, step=train_step)

if __name__ == "__main__":
    main()