# scripts/metrics.py
import torch, numpy as np
from sklearn.metrics import ( accuracy_score, recall_score, f1_score, precision_score, confusion_matrix)

@torch.no_grad()
def collect_preds(model, loader, device, normalize_batch=None):
    # get predictions from the model for the given data loader.
    # If normalize_batch is provided, it will be applied to the input data.
    model.eval()
    all_p, all_y = [], []
    for batch in loader:        # batch can be either (X, y) or (X, L, y), size of 32
        if len(batch) == 2:
            X, y = batch
            X, y = X.to(device), y.to(device)
            if normalize_batch is not None:
                X = normalize_batch(X)
            logits = model(X)
        else:
            X, L, y = batch
            X, L, y = X.to(device), L.to(device), y.to(device)
            if normalize_batch is not None:
                X = normalize_batch(X)
            logits = model(X, L)
        all_p.append(logits.argmax(1).cpu().numpy())
        all_y.append(y.cpu().numpy())
    return np.concatenate(all_p), np.concatenate(all_y)


def compute_scores(y_true, y_pred):
    acc   = accuracy_score(y_true, y_pred)

    # macro (balanced, class-averaged)
    rec_macro  = recall_score(y_true, y_pred, average="macro",    zero_division=0)  # UAR
    f1_macro   = f1_score(   y_true, y_pred, average="macro",    zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)

    # weighted (support-weighted, matches your formula request)
    rec_weighted  = recall_score(   y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted   = f1_score(       y_true, y_pred, average="weighted", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    return {
        "acc": acc,
        "uar": rec_macro,  # keep same key for your scheduler/early stop
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,

        "f1_weighted": f1_weighted,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,

        "cm": cm,
    }