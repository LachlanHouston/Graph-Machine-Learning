import sys
import platform
import multiprocessing as mp
from contextlib import nullcontext

import argparse
import time
import numpy as np
import random

import dill
import torch
import torch.nn as nn
import torch.nn.functional as F

from warnings import filterwarnings
filterwarnings("ignore")

from tqdm import tqdm
from sklearn.metrics import f1_score

from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from ogb.nodeproppred import Evaluator
# your package modules
from graph_ml.data import *
from graph_ml.model import *
from graph_ml.utils import graph_sample, prepare_data, multilabel_f1_from_logits

# WandB
try:
    import wandb
except ImportError:
    wandb = None

def main():
    parser = argparse.ArgumentParser(description='Training GNN')

    parser.add_argument('--data_dir', type=str, default='data/proc_Yelp_top300cats.pk',
                        help='The address of preprocessed graph.')
    parser.add_argument('--model_dir', type=str, default='models/test.pt',
                        help='The address for storing the trained models.')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the loss/acc curve')
    parser.add_argument('--cuda', type=int, default=0, help='Available GPU ID')
    parser.add_argument('--conv_name', type=str, default='hgt',
                        choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                        help='GNN filter (default: HGT)')
    parser.add_argument('--n_hid', type=int, default=512, help='Hidden dim')
    parser.add_argument('--n_heads', type=int, default=8, help='Attention heads')
    parser.add_argument('--n_layers', type=int, default=4, help='GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio')
    parser.add_argument('--sample_depth', type=int, default=4, help='Sampling depth (hops)')
    parser.add_argument('--sample_width', type=int, default=200, help='Neighbors per hop per type')
    parser.add_argument('--n_epoch', type=int, default=2, help='Epochs')
    parser.add_argument('--n_pool', type=int, default=0, help='Processes for sampling (set 0 to disable)')
    parser.add_argument('--n_batch', type=int, default=8, help='Batches (sampled graphs) per epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='Output papers per batch')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient norm clipping')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='Graph Machine Learning', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb_mode', type=str, default=None, choices=[None, 'online', 'offline'],
                        help='wandb mode (default: env/WANDB_MODE or online)')

    args = parser.parse_args()

    # Disable multiprocessing if not Linux for local debugging
    if platform.system() != "Linux":
        print(f"Warning: Multiprocessing disabled (n_pool set to 0) on {platform.system()}")
        args.n_pool = 0

    args_print(args)

    graph = dill.load(open(args.data_dir, 'rb'))
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    target_feature = 'business'
    target_nodes = np.arange(len(graph.node_feature[target_feature]))

    gnn = GNN(in_dim = len(graph.node_feature[target_feature][0]), \
          n_hid = args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
          num_types = len(graph.get_types()), num_relations = len(graph.get_meta_graph()) + 1,\
          prev_norm = True, last_norm = True, use_RTE = True
          )
    
    classifier = Classifier(args.n_hid, graph.y.shape[1])

    model = nn.Sequential(gnn, classifier).to(device)

    train_mask = getattr(graph, "train_mask", None)
    if train_mask is None and hasattr(graph, "split"):
        train_mask = graph.split.get("train", None)
    assert train_mask is not None, "Need a global train mask to compute class imbalance."

    Y_train = torch.as_tensor(graph.y[train_mask], dtype=torch.float32)
    P = Y_train.sum(dim=0)
    N = torch.tensor(Y_train.shape[0], dtype=torch.float32)

    eps = 1e-6
    pos_weight = (N - P) / (P + eps)

    # optional: clamp to avoid extreme explosions if a class is ultra-rare
    pos_weight = pos_weight.clamp(min=1.0, max=1e4)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    print('Number of Parameters for Total model: %d' % get_n_params(model))
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],     'weight_decay': 0.0}
        ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-06)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.05, anneal_strategy='linear', final_div_factor=10, max_lr = 5e-4, total_steps = args.n_batch * args.n_epoch + 1)
    
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
            wandb.log({
                "dataset/business_count": int(len(graph.node_feature['business'])),
            })

    if args.n_pool > 0:
        # Linux only; fork avoids pickling overhead
        ctx = mp.get_context("fork")
        pool = ctx.Pool(args.n_pool)
        jobs = prepare_data(pool, graph, target_feature, target_nodes, args, task_type="train")
    else:
        pool, jobs = None, None

    best_val = -float("inf")
    train_step = 0

    # Training loop
    st = time.time()
    epoch_bar = tqdm(range(1, args.n_epoch + 1), desc="Epochs")
    for epoch in epoch_bar:
        start_prep = time.time()
        if args.n_pool > 0:
            datas = [job.get() for job in jobs]
            # reuse pool to avoid churn; queue next epoch
            pool.close(); pool.join()
            ctx = mp.get_context("fork")
            pool = ctx.Pool(args.n_pool)
            jobs = prepare_data(pool, graph, target_feature, target_nodes, args, task_type="train", device=device.type)
        else:
            datas = [
                graph_sample(
                    graph, target_feature, args, random.randint(0, 2**31 - 1),
                    np.random.choice(target_nodes, args.batch_size, replace=False), device.type
                )
                for _ in range(args.n_batch)
            ]

        prep_time = time.time() - start_prep
        print(f"Epoch {epoch} — Data Preparation: {prep_time:.1f}s")

        epoch_labels = []
        for _, _, _, _, _, (_, _, _), ylabel in datas:
            y = np.asarray(ylabel, dtype=np.float32)  # [B, C]
            epoch_labels.append(y)
        if epoch_labels:
            Y = np.concatenate(epoch_labels, axis=0)  # [N, C]
            batch_label_mean = float(Y.mean())        # overall prevalence
            batch_label_std  = float(Y.mean(axis=0).std())  # per-class prevalence spread
            label_cardinality = float(Y.sum(axis=1).mean()) # avg labels per sample
        else:
            batch_label_mean = float("nan")
            batch_label_std  = float("nan")
            label_cardinality = float("nan")

        gnn.train(); classifier.train()
        running_loss = 0.0
        n_steps = 0

        val_logits_list = []
        val_targets_list = []

        batch_iter = tqdm(datas, desc=f"Epoch {epoch}", leave=False)
        for (node_feature, node_type, edge_time, edge_index, edge_type,
         (train_mask, valid_mask, test_mask), ylabel) in batch_iter:

            # targets for the output nodes (B, C)
            ylabel = torch.as_tensor(ylabel, dtype=torch.float32, device=device)

            # forward only for output nodes in the sampled batch
            node_rep = gnn(node_feature, node_type, edge_time, edge_index, edge_type)
            node_rep_batch = node_rep[:len(ylabel)]          # [B, D]
            logits = classifier(node_rep_batch)              # [B, C]

            train_logits = logits[train_mask]                # [B_train, C]
            train_targets = ylabel[train_mask]               # [B_train, C]

            loss = criterion(train_logits, train_targets)
            # backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # gradient clipping (helps with large graphs)
            if args.clip is not None and args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)

            running_loss += loss.item()
            n_steps += 1

            train_step += 1
            scheduler.step(train_step)  

            batch_iter.set_postfix({
                "loss": f"{running_loss / n_steps:.4f}"
            })

            # collect validation predictions from the same batch (cheap signal)
            if valid_mask.any():
                val_logits_list.append(logits[valid_mask].detach())
                val_targets_list.append(ylabel[valid_mask].detach())

        avg_train_loss = running_loss / max(1, n_steps)

        if val_logits_list:
            v_logits = torch.cat(val_logits_list, dim=0)     # [V, C]
            v_targets = torch.cat(val_targets_list, dim=0)   # [V, C]
            val_micro_f1 = multilabel_f1_from_logits(v_logits, v_targets, average="micro")
            val_macro_f1 = multilabel_f1_from_logits(v_logits, v_targets, average="macro")
        else:
            val_micro_f1 = float("nan")
            val_macro_f1 = float("nan")

        score = val_micro_f1 if not math.isnan(val_micro_f1) else -float("inf")
        if score > best_val:
            best_val = score
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "pos_weight": pos_weight,
                "best_val_micro_f1": best_val,
                "args": vars(args),
            }, args.model_dir)


        if run is not None:
            wandb.log({
                "epoch": epoch,
                "time/prep_s": prep_time,
                "train/loss": avg_train_loss,
                "valid/micro_f1": val_micro_f1,
                "valid/macro_f1": val_macro_f1,
                "labels/prevalence_mean": batch_label_mean,
                "labels/prevalence_std": batch_label_std,
                "labels/cardinality": label_cardinality,
                "lr": optimizer.param_groups[0]["lr"],
            })

        tqdm.write("")
        epoch_bar.set_postfix({
            "loss": f"{avg_train_loss:.4f}",
            "vF1_micro": f"{val_micro_f1:.4f}",
            "vF1_macro": f"{val_macro_f1:.4f}",
            "card": f"{label_cardinality:.2f}"
        })

        # reset start time for next epoch’s prep timing
        st = time.time()

    if pool:
        pool.close(); pool.join()

    if run is not None:
        run.finish()

    print(f"Best val micro-F1: {best_val:.4f} | saved: {args.model_dir}")

if __name__ == "__main__":
    if platform.system() == "Linux":
        mp.set_start_method("fork", force=True)
    else:
        mp.set_start_method("spawn", force=True)
    mp.freeze_support()
    main()