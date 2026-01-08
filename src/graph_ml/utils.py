import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from texttable import Texttable
import wandb

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def args_print(args):
    _dict = vars(args)
    t = Texttable() 
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def get_device(cuda: int) -> torch.device:
    if cuda >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda}")
    return torch.device("cpu")

def build_num_neighbors_from_cfg(cfg):
    """
    Turn cfg.data.num_neighbors (list of {src, rel, dst, hops})
    into { (src, rel, dst): [fanout_h1, fanout_h2, ...] }.
    Also validates that all relations define the same number of hops.
    """
    items = getattr(cfg.data, "num_neighbors", None)
    if items is None:
        return {}

    nn = {}
    hop_lengths = set()
    for it in items:
        src = str(it["src"])
        rel = str(it["rel"])
        dst = str(it["dst"])
        hops = list(it["hops"])
        nn[(src, rel, dst)] = hops
        hop_lengths.add(len(hops))

    # Ensure consistent number of hops across relations (required by PyG)
    if len(hop_lengths) > 1:
        raise ValueError(
            f"Inconsistent hops across relations: lengths = {sorted(hop_lengths)}. "
            "All 'hops' lists must have the same length (same num_hops)."
        )

    return nn

def multilabel_f1_from_logits(logits, targets, threshold=0.5, average="micro", eps=1e-8):
    """
    logits: [B, C], raw
    targets: [B, C], {0,1} floats
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(targets.dtype)

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)

    if average == "micro":
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return f1.item()
    elif average == "macro":
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1_c = 2 * precision * recall / (precision + recall + eps)
        return f1_c.mean().item()
    else:
        raise ValueError("average must be 'micro' or 'macro'")
    
def log_confmatrix(run, y_true_i, y_pred_i, step, normalize=None, title="Val Confusion Matrix"):
    """
    y_true_i, y_pred_i: int tensors in {1..5}
    normalize: None | 'true' | 'pred' | 'all' (sklearn options)
    Logs a Matplotlib confusion matrix figure to Weights & Biases.
    """
    if run is None:
        return

    class_labels = ['1 star', '2 star', '3 star', '4 star', '5 star']
    # keep labels explicit to pin row/col order
    labels = [1, 2, 3, 4, 5]

    cm = confusion_matrix(
        y_true=y_true_i.cpu().numpy(),
        y_pred=y_pred_i.cpu().numpy(),
        labels=labels,
        normalize=normalize,
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    # values_format depends on normalization
    values_format = '.2f' if normalize else 'd'
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format=values_format)
    ax.set_title(title + (f" (normalize={normalize})" if normalize else ""))
    fig.tight_layout()

    run.log({"val/confusion_matrix": wandb.Image(fig)}, step=step)
    plt.close(fig)