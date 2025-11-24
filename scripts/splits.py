# scripts/splits.py
from typing import Dict, Sequence, Callable, Optional, Iterable
import torch  
from torch.utils.data import Dataset, Subset, DataLoader
from scripts.config import TRAIN_ACTORS, VAL_ACTORS, TEST_ACTORS,PACKED_FUSED_CONCAT_FEATURE_NORM_POOLED_NPZ
import numpy as np

def split_indices_by_actor_from_items( 
    dataset, 
    train_actors: Iterable[int], 
    val_actors: Iterable[int], 
    test_actors: Iterable[int],
) -> Dict[str, list]:
    # split dataset indices by actor ids into train, val, and test subsets.
    train_indices = [i for i,item in enumerate(dataset.items) if item.actor_id in train_actors]
    val_indices   = [i for i,item in enumerate(dataset.items) if item.actor_id in val_actors]
    test_indices  = [i for i,item in enumerate(dataset.items) if item.actor_id in test_actors]
    return {"train": train_indices, "val": val_indices, "test": test_indices}

# create DataLoaders for train/val/test
def make_loaders(
    dataset: Dataset,
    idx_splits: Dict[str, Sequence[int]],
    batch_size: int,
    collate_fn: Optional[Callable],
    num_workers: int = 0,
    pin_memory: bool = False,
    generator: Optional[torch.Generator] = None,
    prefetch_factor: int = 4,
    drop_last: bool = False,
) -> Dict[str, DataLoader]:
    loaders = {}
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False                  # using CPU: pin_memory is not needed

    for name, idxs in idx_splits.items():   # train, val, test
        # create a subset of the dataset for each split using split indices   
        subset = Subset(dataset, idxs)
        
        drop_last = (name == "train")  # only drop for train

        # populate dictionary with DataLoader object for each subset of data. 
        loaders[name] = DataLoader(         # use DataLoader to create batches of data, shuffle them, and parallelize dataiterations
            subset,
            batch_size=batch_size,
            shuffle=(name == "train"),      # for "train" set, randomize the order of samples each epoch
            collate_fn=collate_fn,          # how to stack samples into a batch
            num_workers=num_workers,        # number of parallel workers to load data
            pin_memory=pin_memory,          # if True (and using CUDA), DataLoader puts batches in pinned (page-locked) host memory to speed up CPU -> GPU copies.
            drop_last = drop_last,          # if True, drop the last batch if it is smaller than batch_size
            persistent_workers=(num_workers > 0),   # keep workers alive after the first epoch to speed up subsequent epochs            
            generator=generator,            # for reproducibility of the shuffled data loading order
            prefetch_factor=prefetch_factor,
        )
    return loaders



# for debugging and understanding the dataset distribution:summarize the split sizes and actors in each split


def _summarize_split(dataset, idxs, name):
    actors = sorted({dataset.items[i].actor_id for i in idxs})
    print(f"{name}: n={len(idxs)} | actors={actors[:8]}{'...' if len(actors)>8 else ''}")
    
def main():
   
    import torch
    from scripts.config import PROJECT_ROOT

    from scripts.dataset_ravdess import PackedRavdess, collate_mean_pool
    
    ds = PackedRavdess(PROJECT_ROOT, variant="fused_concat")


    ################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("feature_dir =", ds.feature_dir)
    print("seq_dim =", ds.seq_dim)

    # # load one file to be sure
    # import numpy as np, os
    # sample_path = ds.items[0].path
    # arr = np.load(sample_path)
    # print("sample file:", sample_path, "shape:", arr.shape)

    # load stats
    stats = np.load(PACKED_FUSED_CONCAT_FEATURE_NORM_POOLED_NPZ)
    mu  = torch.from_numpy(stats["mean"]).float().to(device)
    std = torch.from_numpy(stats["std"]).float().to(device)
    print("mu len =", mu.numel())

    # FAIL FAST if modality doesn’t match
    assert mu.numel() == ds.seq_dim, (
        f"Normalization dim {mu.numel()} != dataset dim {ds.seq_dim}. "
        f"feature_dir={ds.feature_dir}"
    )

    ######################


    splits = split_indices_by_actor_from_items(ds, TRAIN_ACTORS, VAL_ACTORS, TEST_ACTORS)
    print("Split sizes:", {k: len(v) for k, v in splits.items()})

    g = torch.Generator().manual_seed(42)
    loaders = make_loaders(ds, splits, batch_size=32, collate_fn=collate_mean_pool,
                           num_workers=0, pin_memory=True, generator=g)

    for X, y in loaders["train"]:
        print("Batch X:", X.shape, "y:", y.shape)
        break

    _summarize_split(ds, splits["train"], "train")
    _summarize_split(ds, splits["val"],   "val")
    _summarize_split(ds, splits["test"],  "test")

if __name__ == "__main__":
    # On Windows + multiprocessing, this guard is REQUIRED to avoid re-execution.
    # It also prevents double prints when importing this module from elsewhere.
    main()