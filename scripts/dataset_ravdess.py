# scripts/dataset_packed.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from scripts.config import BACKBONE_NAME, OUTPUTS_DIR

Variant = Literal["audio", "video", "fused_concat", "fused_interleaved"]

def actor_id_from_filename(filename: str) -> int:
    # Works for both ".mp4" and ".npy" stems: "01-xx-yy-...-AA"
    stem = Path(filename).stem
    parts = stem.split("-")
    return int(parts[-1])  # last token is actor id

@dataclass(frozen=True)
class Item:
    idx: int
    label_idx: int
    actor_id: int

class PackedRavdess(Dataset):
    """
    Dataset over packed feature arrays saved by extract_fuse_features_mnv2_two_tower_pack.py.
    Loads arrays with mmap to avoid huge RAM spikes.
    Keeps a .items list so scripts/splits.split_indices_by_actor_from_items works unchanged.

    variant:
      - "audio"            -> FEATURE_DIR / "audio_features.npy"          [N, Ta, 1280]
      - "video"            -> FEATURE_DIR / "video_features.npy"          [N, Tv, 1280]
      - "fused_concat"     -> FEATURE_DIR / "fused_features_concat.npy"            [N, Tf, 2560]
      - "fused_interleaved"-> FEATURE_DIR / "fused_features_interleaved.npy"       [N, 2*Tf, 1280]
    """
    def __init__(self, project_root: Path, variant: Variant = "fused_concat", backbone = BACKBONE_NAME, dtype: Optional[str] = None):
        self.root = Path(project_root).resolve()
        self.variant: Variant = variant
        self.feature_dir = OUTPUTS_DIR  / "features" / backbone

   #     from scripts.config import LABELS_CSV, FEATURE_DIR
        import pandas as pd, numpy as np, os

        # choose file
        name = {
            "audio": "audio_features.npy",
            "video": "video_features.npy",
            "fused_concat": "fused_features_concat.npy",
            "fused_interleaved": "fused_features_interleaved.npy",
        }[variant]

        self.data_path = self.feature_dir / name
        if not self.data_path.exists():
            raise FileNotFoundError(f"Missing {self.data_path}. Run the pack extractor first.")

        # memory-map to keep RAM small
        self.arr = np.load(self.data_path, mmap_mode="r")  # shape [N, T, D]
        self.N, self.T, self.D = self.arr.shape

        # labels + file order
        self.labels_csv = self.feature_dir / "ravdess_labels.csv" 
        df = pd.read_csv(self.labels_csv)  # columns: file, label

        print("LABELS_CSV =", self.labels_csv)
        print("FEATURE_DIR =", self.feature_dir)


        # Enforce order using files.npy when present
        files_npy = self.feature_dir / "files.npy"
        if files_npy.exists():
            files = np.load(files_npy, allow_pickle=True,mmap_mode="r").tolist()
            
            # reindex df to this order
            order = pd.Index(files, name="file")          # name it!
            df = (
                df.set_index("file")
                .reindex(order)                         # preserves name
                .reset_index()                          # column will be "file"
            )
            

        # label mapping
        labels = sorted(df["label"].unique())
        self.label_to_idx = {lab: i for i, lab in enumerate(labels)}
        self.idx_to_label = {i: lab for lab, i in self.label_to_idx.items()}

        # items list aligned with array order
        self.items = []
        for i, row in df.iterrows():
            actor = actor_id_from_filename(row["file"])
            y = self.label_to_idx[row["label"]]
            self.items.append(Item(idx=i, label_idx=y, actor_id=actor))

        if len(self.items) != self.N:
            raise RuntimeError(f"Array length N={self.N} does not match labels count {len(self.items)}.")

        # Optional on-load dtype conversion (e.g., float16->float32 view)
        self.cast_to = dtype

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        # slice the memmapped array -> np.ndarray view (still lazy)
        x = self.arr[i]  # [T, D]
        
        if self.cast_to == "float32" and x.dtype != np.float32:
            x = x.astype(np.float32, copy=True)
        else:
            x = np.ascontiguousarray(x).copy()    # ensure writable

        x = torch.from_numpy(x).float()
        y = torch.tensor(self.items[i].label_idx, dtype=torch.long)
        return x, y
    





    @property
    def num_classes(self):
        return len(self.label_to_idx)

    @property
    def seq_dim(self):
        return self.D

    @property
    def time_steps(self):
        return self.T

# Collates identical to dataset_ravdess for drop-in use
def collate_mean_pool(batch):
    xs, ys = zip(*batch)
    pooled = [x.mean(dim=0) for x in xs]    # [D]
    X = torch.stack(pooled, dim=0)          # [B, D]
    y = torch.stack(ys, dim=0)              # [B]
    return X, y

# convert variable-length sequences into a single padded tensor
def pad_pack(batch, pad_value=0.0):
    xs, ys = zip(*batch)  # xs: list of [T_i, D] tensors
    lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    X = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_value)  # [B, maxT, D]
    ys = torch.tensor(ys, dtype=torch.long)
    return X, lengths, ys           # X: [B, maxT, D], lengths:[B], ys:[B]