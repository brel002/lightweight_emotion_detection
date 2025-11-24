
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from scripts.splits import make_loaders
from scripts.dataset_ravdess import pad_pack as packed_pad_pack

#existing normalizer
from scripts.compute_norm_nan import get_feature_norm_frames_fromloaders

def retrain_trainval_then_test(
    *,
    dataset,                       # PackedRavdess instance
    splits: dict,                  # {"train": idxs, "val": idxs, "test": idxs}
    model_builder_fn,              # () -> nn.Module (already parameterized, will be .to(device) here)
    device: torch.device,
    best_epoch_saved: int,         # train this many epochs on train∪val
    args,                          # argparse.Namespace (lr, weight_decay, label_smoothing, etc.)
    collect_preds_fn,              # (model, loader, device, normalize_batch) -> (pred, truth)
    compute_scores_fn,             # (truth, pred) -> dict
    generator=None,
    spec_augmenter_tv=None,
    mod_drop_tv=None,
    model_name:str="gru",
):
    """
    Retrain on train + val for 'best_epoch_saved' epochs, then evaluate once on test.
    Uses the SAME μ/σ computation path as main training via get_feature_norm_frames_fromloaders.
    """
    # 1) Build train ∪ val + test loaders
    trainval_idxs = splits["train"] + splits["val"]
    tv_splits = {"train": trainval_idxs, "val": [], "test": splits["test"]}

    


    tv_loaders = make_loaders( dataset, 
                                tv_splits, 
                                batch_size=32, 
                                collate_fn=packed_pad_pack,
                                num_workers=4,  
                                pin_memory=torch.cuda.is_available(),
                                prefetch_factor=4,
                                generator=generator,
                                )




    # allways compute μ/σ only from the curretn train + val loader
    # mu_np, std_np, _ = get_feature_norm_frames_fromloaders({"train": tv_loaders["train"]})
    mu_np, std_np, _ = get_feature_norm_frames_fromloaders(tv_loaders)

    mu_tv  = torch.from_numpy(mu_np).to(device=device, dtype=torch.float32)
    std_tv = torch.from_numpy(std_np).to(device=device, dtype=torch.float32)

    def normalize_batch_tv(X: torch.Tensor) -> torch.Tensor:
        return (X - mu_tv) / (std_tv + 1e-8)


    # 3) Balanced class weights on train∪val
    tv_labels = [dataset.items[i].label_idx for i in tv_loaders["train"].dataset.indices]
    classes = np.arange(dataset.num_classes)

    y = np.array(tv_labels)
    
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    weights_tv = torch.tensor(w, dtype=torch.float32, device=device)

    # 4) Rebuild the same model and optimizer
    if model_name=="gru":
        model_tv: nn.Module = model_builder_fn().to(device)
    else:
        model_tv: nn.Module = model_builder_fn(  mu_tv=mu_tv,
                                                 std_tv=std_tv
                                             ).to(device)






    crit_tv = nn.CrossEntropyLoss(weight=weights_tv, label_smoothing=args.label_smoothing)

    if model_name=="mamba":
        opt_tv = torch.optim.AdamW(model_tv.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt_tv  = torch.optim.Adam(model_tv.parameters(), lr=args.lr, weight_decay=args.weight_decay)

   
    # 5) Train exactly best_epoch_saved epochs (no early stop on this train + val concatenated split)
    max_epochs = max(1, int(best_epoch_saved))
    model_tv.train()

    for _epoch in range(1, max_epochs + 1):
        model_tv.train()
        seen = correct = 0
        for X, lengths, y in tv_loaders["train"]:

            X = X.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if model_name=="gru":
                X = normalize_batch_tv(X)

            # in mamba, augs are done inside the model forward.

            if spec_augmenter_tv is not None:
                X = spec_augmenter_tv(X, lengths)
            if mod_drop_tv is not None:
                X = mod_drop_tv(X)



            opt_tv.zero_grad(set_to_none=True)
            logits = model_tv(X, lengths)
            loss = crit_tv(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_tv.parameters(), 1.0)
            opt_tv.step()
            correct += (logits.argmax(1) == y).sum().item()
            seen    += y.size(0)
        if _epoch == 1 or _epoch == max_epochs:
            print(f"[retrain] epoch {_epoch}/{max_epochs} train_acc={correct/max(seen,1):.3f}")

    model_tv.eval()

    # 6) Final test evaluation
    norm_fn = normalize_batch_tv if model_name == "gru" else None

    p_test_f, y_test_f = collect_preds_fn(model_tv, tv_loaders["test"], device, normalize_batch=norm_fn)  # mamba does normalization internally
    final_scores = compute_scores_fn(y_test_f, p_test_f)

    return final_scores, max_epochs
