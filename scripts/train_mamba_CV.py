# scripts/train_mamba_CV.py


import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ---- Standard library ----
from pathlib import Path
import argparse
import json
import time
import secrets
import random
# import hashlib

# ---- Third-party ----
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import recall_score
# from sklearn.model_selection import train_test_split   

from mamba_ssm import Mamba  

# ---- Local imports ----
from scripts.dataset_ravdess import PackedRavdess, collate_mean_pool, pad_pack
from scripts.config import (
    PROJECT_ROOT, STORAGE_ROOT, OUTPUTS_DIR,
    TRAIN_ACTORS, VAL_ACTORS, TEST_ACTORS,
    MODELS_DIR, BACKBONE_NAME, FEATURE_DIR,
)
from scripts.splits import split_indices_by_actor_from_items, make_loaders
from scripts.compute_norm_nan import get_feature_norm_frames_fromloaders
from scripts.log_results import ResultsLogger, count_params, file_size_mb
from scripts.finalise_train_val import retrain_trainval_then_test
from typing import Optional





class ModalityDropout(nn.Module):
    """
    Randomly zero EITHER the audio slice or the video slice per sample with prob ~p each.
    Keeps training robust when one stream is weak/missing.
    """
    def __init__(self, audio_dim: int, p: float = 0.10):
        super().__init__()
        assert 0.0 <= p <= 0.5, "Pick p in [0, 0.5] so 2p<=1"
        self.audio_dim = audio_dim
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        B, T, D = x.shape
        A = self.audio_dim
        V = D - A
        x_v, x_a = x[..., :V], x[..., -A:]

        # Per-sample choice: [0,1) -> drop audio if <p; drop video if in [p,2p); else keep
        choice = torch.rand(B, device=x.device)
        drop_a = (choice < self.p).view(B, 1, 1)
        drop_v = ((choice >= self.p) & (choice < 2*self.p)).view(B, 1, 1)

        x_v = x_v * (~drop_v).float()
        x_a = x_a * (~drop_a).float()
        return torch.cat([x_v, x_a], dim=-1)


# Normalization (frame-level)
class FrameNormalizer(nn.Module):
    def __init__(self, mu, std, device: torch.device):
        super().__init__()
        # stats = np.load(norm_npz_path)
        # mu = torch.tensor(stats["mean"], dtype=torch.float32, device=device)   # [D]
        # std = torch.tensor(stats["std"], dtype=torch.float32, device=device) # [D]
        self.register_buffer("mu", mu.view(1, 1, -1))
        self.register_buffer("std", std.view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        return (x - self.mu) / (self.std + 1e-8)

class MaskedAttnPool(nn.Module):
    def __init__(self, in_dim: int, hidden: int, p_drop: float = 0.2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )
        self.post_drop = nn.Dropout(p_drop)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-9):
        # mask: [B,T] bool
        logits = logits.masked_fill(~mask, float('-inf'))
        # softmax; rows with all -inf become NaN -> zero them, then renormalize on valid support
        alpha = torch.softmax(logits, dim=dim)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = alpha * mask.float()
        denom = alpha.sum(dim=dim, keepdim=True).clamp_min(eps)
        return alpha / denom

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        scores = self.attn(x).squeeze(-1)                      # [B,T]
        mask = (torch.arange(T, device=x.device).unsqueeze(0)  # [B,T] bool
                < lengths.unsqueeze(1))
        alpha = self._masked_softmax(scores, mask, dim=1)      # [B,T]
        ctx = (alpha.unsqueeze(-1) * x).sum(dim=1)             # [B,D]
        return self.post_drop(ctx)

class MaskedMeanPool(nn.Module):
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        mask = (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))  # bool [B,T]
        summed = (x * mask.unsqueeze(-1).float()).sum(dim=1)                           # [B,D]
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0).float()                   # [B,1]
        return summed / denom


class MaskedTopKAttnPool(nn.Module):
    def __init__(self, in_dim: int, hidden: int, frac: float = 0.35, min_k: int = 6, p_drop: float = 0.0):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )
        self.frac = frac
        self.min_k = max(1, int(min_k))
        self.post_drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        device = x.device
        mask = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1))  # bool [B,T]

        scores = self.scorer(x).squeeze(-1).masked_fill(~mask, float('-inf'))        # [B,T]

        # per-row k, clamped to [1, T] so topk is always well-defined
        k = (lengths.float() * self.frac).ceil().clamp_min(self.min_k).clamp_max(T).long()  # [B]
        maxk = int(k.max().item())

        # topk once with maxk, then per-row keep only first k[b]
        topk_vals, topk_idx = torch.topk(scores, k=maxk, dim=1)  # [B,maxk], [B,maxk]
        ar = torch.arange(maxk, device=device).unsqueeze(0)       # [1,maxk]
        use = (ar < k.unsqueeze(1)).float()                       # [B,maxk]

        topk_x = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D))  # [B,maxk,D]
        pooled = (topk_x * use.unsqueeze(-1)).sum(dim=1) / use.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.post_drop(pooled)


class AudioOnlySpecAugment(nn.Module):
    def __init__(self,
                 audio_dim: int = 1280,
                 freq_mask_ratio: float = 0.15,
                 time_mask_ratio: float = 0.20,
                 num_f: int = 1,
                 num_t: int = 1,
                 per_sample_freq: bool = True,
                 inplace: bool = False,
                 allow_zero: bool = True):  # allow zero-width masks (matches original spirit)
        super().__init__()
        self.A = int(audio_dim)
        self.fr = float(freq_mask_ratio)
        self.tr = float(time_mask_ratio)
        self.nf = int(num_f)
        self.nt = int(num_t)
        self.per_sample_freq = bool(per_sample_freq)
        self.inplace = bool(inplace)
        self.allow_zero = bool(allow_zero)

    def forward(self, X: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # X: [B, T, D] where last A dims are audio
        if not self.training:
            return X
        if not self.inplace:
            X = X.clone()

        # separate video and audio from concatenated features
        B, T, D = X.shape
        A = self.A
        if A <= 0 or A > D:
            return X  # safety

        V = X[..., :D - A]
        A_ = X[..., -A:]  # audio slice [B, T, A]

        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=X.device)

        # time masking  
        if self.tr > 0.0 and self.nt > 0:
            for b in range(B):
                Ti = int(lengths[b].item())
                if Ti <= 0:
                    continue
                max_t = int(Ti * self.tr)  # no floor to 1; let it be 0 to allow skip
                if max_t <= 0:
                    continue  # do nothing if ratio too small for this sequence
                for _ in range(self.nt):
                    if self.allow_zero:
                        # t in [0, max_t]
                        t = int(torch.randint(0, max_t + 1, (), device=X.device).item())
                        if t == 0:
                            continue
                    else:
                        # t in [1, max_t]
                        if max_t < 1:
                            continue
                        t = int(torch.randint(1, max_t + 1, (), device=X.device).item())

                    # start index
                    if Ti - t <= 0:
                        t0 = 0
                    else:
                        t0 = int(torch.randint(0, Ti - t + 1, (), device=X.device).item())
                    A_[b, t0:t0 + t, :] = 0

        # frequency masking 
        if self.fr > 0.0 and self.nf > 0:
            max_f = int(A * self.fr)  # can be 0
            if max_f > 0:
                if self.per_sample_freq:
                    for _ in range(self.nf):
                        # widths per sample
                        if self.allow_zero:
                            f = torch.randint(0, max_f + 1, (B,), device=X.device)
                        else:
                            f = torch.randint(1, max_f + 1, (B,), device=X.device)
                        # skip where zero
                        if (f > 0).any():
                            f0_max = (A - f).clamp_min(0)
                            # uniform start per sample in [0, f0_max]
                            # torch.randint with per-sample upper bounds is awkward; use rand trick
                            f0 = torch.floor(torch.rand(B, device=X.device) * (f0_max + 1)).long()
                            for b in range(B):
                                fb = int(f[b].item())
                                if fb <= 0:
                                    continue
                                f0b = int(f0[b].item())
                                A_[b, :, f0b:f0b + fb] = 0
                else:
                    for _ in range(self.nf):
                        if self.allow_zero:
                            f = int(torch.randint(0, max_f + 1, (), device=X.device).item())
                            if f == 0:
                                continue
                        else:
                            f = int(torch.randint(1, max_f + 1, (), device=X.device).item())
                        f0 = int(torch.randint(0, A - f + 1, (), device=X.device).item())
                        A_[:, :, f0:f0 + f] = 0

        return torch.cat([V, A_], dim=-1)


class TimeDropout(nn.Module):
    # Drop a random contiguous time span (all features) with probability p.
    def __init__(self, p: float = 0.10, max_frac: float = 0.10):
        super().__init__()
        self.p = p
        self.max_frac = max_frac
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        B, T, D = x.shape
        for b in range(B):
            Lb = int(lengths[b].item()) if torch.is_tensor(lengths[b]) else int(lengths[b])
            if Lb > 1 and torch.rand(1, device=x.device).item() < self.p:
                L = max(1, int(self.max_frac * Lb))
                L = min(L, Lb)
                start = torch.randint(0, Lb - L + 1, (1,), device=x.device).item()
                x[b, start:start+L, :] = 0.0
        return x


    
class TemporalDWConvStem(nn.Module):
    #    Lightweight temporal front-end: depthwise temporal convs and pointwise projection.
    #    Adds local temporal inductive bias before the sequence mixer.
    
    def __init__(self, d_in: int, d_model: int, k1: int = 5, k2: int = 5, p_drop: float = 0.1):
        super().__init__()
        self.dw1 = nn.Conv1d(d_in, d_in, kernel_size=k1, padding=k1//2, groups=d_in)
#       self.dw2 = nn.Conv1d(d_in, d_in, kernel_size=k2, padding=k2//2, dilation=2, groups=d_in)
        d2 = 2  # keep dilation
        pad2 = (k2 - 1) * d2 // 2              # 'same' for odd k2
        self.dw2 = nn.Conv1d(d_in, d_in, kernel_size=k2, padding=pad2, dilation=d2, groups=d_in)
        self.pw  = nn.Conv1d(d_in, d_model, kernel_size=1)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):  # x: [B,T,D_in]
        x = x.transpose(1, 2)          # [B,D_in,T]
        x = self.dw1(x)
        x = self.act(x)
        x = self.dw2(x)
        x = self.pw(x).transpose(1, 2) # [B,T,d_model]
        x = self.norm(x)
        return self.drop(x)
    
class ModalityLayerNorm(nn.Module):
    def __init__(self, video_dim: int, audio_dim: int, audio_at_end: bool = True):
        super().__init__()
        self.audio_at_end = audio_at_end
        self.lv = nn.LayerNorm(video_dim)
        self.la = nn.LayerNorm(audio_dim)
    def forward(self, x):  # [B,T,D]
        B,T,D = x.shape
        A = self.la.normalized_shape[0]
        V = D - A
        v, a = x[..., :V], x[..., -A:]
        v, a = self.lv(v), self.la(a)
        return torch.cat([v,a], dim=-1)
    
class ModalityWiseTIN(nn.Module):
    def __init__(self, d_audio: int, d_video: int, use_affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.d_audio = d_audio
        self.d_video = d_video
        self.eps = eps
        self.use_affine = use_affine
        if use_affine:
            # learnable scale/shift per feature dim (audio, video)
            self.gamma_a = nn.Parameter(torch.ones(d_audio))
            self.beta_a  = nn.Parameter(torch.zeros(d_audio))
            self.gamma_v = nn.Parameter(torch.ones(d_video))
            self.beta_v  = nn.Parameter(torch.zeros(d_video))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor):
        # x: [B,T,D], mask: [B,T] in {0,1}
        B, T, D = x.shape
        Da, Dv = self.d_audio, self.d_video
        xa, xv = x[..., :Da], x[..., Da:Da+Dv]          # split A/V
        m = mask.unsqueeze(-1).float()                  # [B,T,1]
        denom = lengths.clamp_min(1).view(B,1,1).float()

        # audio
        ma = (xa * m).sum(dim=1, keepdim=True) / denom
        va = ((xa - ma)**2 * m).sum(dim=1, keepdim=True) / denom
        xa = (xa - ma) / (va.add(self.eps).sqrt())
        # video
        mv = (xv * m).sum(dim=1, keepdim=True) / denom
        vv = ((xv - mv)**2 * m).sum(dim=1, keepdim=True) / denom
        xv = (xv - mv) / (vv.add(self.eps).sqrt())

        # re-mask padded frames back to zero
        xa = xa * m
        xv = xv * m

        if self.use_affine:
            xa = xa * self.gamma_a.view(1,1,Da) + self.beta_a.view(1,1,Da)
            xv = xv * self.gamma_v.view(1,1,Dv) + self.beta_v.view(1,1,Dv)

        return torch.cat([xa, xv], dim=-1)

class FusedTIN(nn.Module):
    def __init__(self, d_audio: int, d_video: int, use_affine: bool = True, eps: float = 1e-5):
        super().__init__() 
        self.d_audio = d_audio
        self.d_video = d_video       
        self.eps = eps
        self.use_affine = use_affine
        if use_affine:
            # learnable scale/shift
            self.gamma = nn.Parameter(torch.ones(d_audio + d_video))
            self.beta  = nn.Parameter(torch.zeros(d_audio + d_video))

       

    def forward(self, x: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor):
        # x: [B,T,D], mask: [B,T] in {0,1}
        B, T, D = x.shape
        expected_D = self.d_audio + self.d_video
        if D != expected_D:
            raise ValueError(f"FusedTIN: got D={D}, expected {expected_D} (d_audio+d_video)")

        mask_t = mask.unsqueeze(-1).float()                  # [B,T,1]

        # for each clip (64 frames), compute the mean/variance across its valid time steps( valid frames),
        # for every feature dim d, then normalize that clip’s entire sequence of features accordingly.

        # get number of valid frames per-utterance (clip) in the batch,  to avoid divide-by-zero:
        # lengths hold B (32) lenghts: length of each clip
        denom = lengths.clamp_min(1).view(B,1,1).float()    # [B,1,1]

        # get masked per-utterance mean over time ( only valid frames): mean of each feature [D] over valid frames of a clip, for all clips (32) in the batch
        mean_t = (x * mask_t).sum(dim=1, keepdim=True) / denom  # [B,1,D]

        # get masked per-utterance variance over time: variance of each feature [D] over valid frames of a clip, for all clips (32) in the batch
        var_t   = ((x - mean_t)**2 * mask_t).sum(dim=1, keepdim=True) / denom  # [B,1,D]

        # normalize each utterance’s time series ( each clip's frames)
        # gives each utterance( clip) zero mean and unit variance across time, for each feature dim, using only its valid frames.
        x = (x - mean_t) / (var_t.add(self.eps).sqrt())

        # keep padded frames strictly zero
        x = x * mask_t

        if self.use_affine:
            # affine per feature dim, apply learned scale/shift
            x = x * self.gamma.view(1,1,D) + self.beta.view(1,1,D)

        # re-mask once more in case affine touched pads (paranoia-safe)
        x = x * mask_t

        return x


    

class MambaStack(nn.Module):
    # Tiny Mamba/Mamba2 stack with pre-norm residuals.
    # Expects input [B,T,D_in]; projects to d_model; returns [B,T,d_model].
    def __init__(self, d_in: int, d_model: int = 256, n_layers: int = 4, p_drop: float = 0.2, bypass_proj: bool = False):
        super().__init__()
        # Try Mamba2 first, then Mamba (v1). If neither found, raise with an install hint.
        self.kind = None
        MambaLayer = None
        try:
            from mamba_ssm.torch.mamba2 import Mamba2  # type: ignore
            MambaLayer = Mamba2
            self.kind = "mamba2"
            print("Using MAMBA2 sequence model.")
        except Exception:
            try:
                from mamba_ssm import Mamba  # type: ignore
                class MambaWrapper(nn.Module):
                    def __init__(self, d_model):
                        super().__init__()
                        self.m = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                    def forward(self, x):
                        return self.m(x)
                MambaLayer = MambaWrapper
                self.kind = "mamba1"
                print("Using MAMBA (v1) sequence model.")
            except Exception:
                raise RuntimeError(
                    "No Mamba implementation found. Please install mamba-ssm:"
                    "  pip install mamba-ssm  # (or the CUDA wheel per your system)"
                )

        self.proj_in  = nn.Identity() if bypass_proj else nn.Linear(d_in, d_model)
        self.block_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.layers   = nn.ModuleList([MambaLayer(d_model) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(p_drop)
        # ReZero-style residual scale to stabilize early training
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D_in]
        x = self.proj_in(x)
        for ln, layer in zip(self.block_norms, self.layers):
            y = layer(ln(x))
            x = x + self.res_scale * y
        x = self.out_norm(x)
        x = self.drop(x)
        return x


# Full model: Norm → (SpecAug/ModDrop) → MambaStack → AttnPool → Head
class AVMambaClassifier(nn.Module):
    def __init__(self, 
                 d_in: int, 
                 n_classes: int, 
                 d_model: int = 256, 
                 n_layers: int = 4,
                 p_drop: float = 0.2, 
                 audio_dim: int | None = None, 
                 video_dim: int | None = None,
                 use_specaug=False, 
                 use_moddrop=False,
                 pool_type: str = 'mean', 
                 use_dwstem: bool = True, 
                 use_tin: str = "off",      # "off" | "permod" | "fused"
                 tin_affine: bool = True,
                 mu:float = 1.0, 
                 std:float = 0.0, 
                 device: Optional[torch.device] = None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm      = FrameNormalizer(mu,std, self.device)
        self.specaug   = AudioOnlySpecAugment(audio_dim=audio_dim,freq_mask_ratio=0.15, time_mask_ratio=0.25) if use_specaug else nn.Identity()
        self.moddrop   = ModalityDropout(audio_dim=audio_dim, p=0.10) if use_moddrop else nn.Identity()
        self.use_dwstem = use_dwstem
        if self.use_dwstem:
            self.stem = TemporalDWConvStem(d_in, d_model, k1=5, k2=5, p_drop=p_drop)
            self.encoder = MambaStack(d_in=d_model, d_model=d_model, n_layers=n_layers, p_drop=p_drop, bypass_proj=True)
        else:
            self.encoder = MambaStack(d_in=d_in, d_model=d_model, n_layers=n_layers, p_drop=p_drop)

        
        self.pool = (
            MaskedMeanPool() if pool_type == 'mean' else
            MaskedAttnPool(in_dim=d_model, hidden=d_model, p_drop=p_drop) if pool_type == 'attn' else
            MaskedTopKAttnPool(in_dim=d_model, hidden=d_model, frac=0.35, min_k=6, p_drop=0.0)  # 'topk'
        )
        self.out_drop  = nn.Dropout(p_drop)
        self.head      = nn.Linear(d_model, n_classes)

        self.use_tin = use_tin  # "off" | "permod" | "fused"
        if self.use_tin == "permod":
            self.tin = ModalityWiseTIN(audio_dim, video_dim, use_affine=tin_affine)
        elif self.use_tin == "fused":
            self.tin = FusedTIN(audio_dim, video_dim, use_affine=tin_affine)
    
    
    # dataset-norm → TIN → augs (train only) → encoder → masked pool
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        B, T, D = x.shape       # T: number of time steps (64 frames) , can be padded; B: batch size (32), D: extracted A+V features

        # encode frames: tell the model which frames in each clip (in the batch of 32) are real and which are just padding so they should be ignored.
        mask = (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).float()  # [B,T]


        ############
        if self.training and (torch.rand(()) < 0.01):  # sample 1% of steps
            with torch.no_grad():
                print(f"[dbg] T={T}, lengths[min..max]={lengths.min().item()}..{lengths.max().item()},"
                    f" mask_valid_mean={(mask.mean().item()):.3f}")

        # guardrail: if some batch has all-zero mask (bad), pretend all frames valid
        if (mask.sum(dim=1) == 0).any():
            mask = torch.ones_like(mask)
        ##################


        # 1) dataset-level normlization with stats from 'train' split
        x = self.norm(x)                         # dataset μ/σ (train-only stats)
        x = x * mask.unsqueeze(-1)               # zero pads before any op

        
        # for each clip (64 frames), compute the mean/variance across its valid time steps( valid frames),
        # for every feature dim d, then normalize that clip’s entire sequence of features accordingly.
        # 2 options:
        # - per-modality TIN: separate norm for audio and video dims
        # - fused TIN: single norm over all dims
        if self.use_tin == "permod":
            x = self.tin(x, mask, lengths)
        elif self.use_tin == "fused":
            x = self.tin(x, mask, lengths) 
        
        # 2) augs

        # resets all padded frames to exact zeros, so augs start from a clean slate of valid frames only
        x = x * mask.unsqueeze(-1)              
 
        if self.training and self.moddrop is not None:
            x = self.moddrop(x)

        if self.training and self.specaug is not None:                  
            if isinstance(self.specaug, AudioOnlySpecAugment):
                x = self.specaug(x, lengths)
            else:
                x = self.specaug(x)


        # reset padded frames to zero again after augs  
        x = x * mask.unsqueeze(-1)


        # # resets all padded frames to exact zeros, no matter what augs did
        # x = x * mask.unsqueeze(-1)

        # 3) encode 
        # get x into the right width/shape. 
        # and (optionally) add a tiny bit of local preprocessing before Mamba.
        x_proj = self.stem(x) if self.use_dwstem else x

        # mamba stack: [B,T,d_model], create a contextualised sequence of frame-level features of size d_model.
        z = self.encoder(x_proj)
        
        # again, reset all padded frames to zeros, for safety
        z = z * mask.unsqueeze(-1)

        # 4) masked pool over time
        c = self.pool(z, lengths)   # [B,d_model], pooled clip-level features
        c = self.out_drop(c)        # dropout before head
        return self.head(c)         # [B,n_classes]




from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix

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




def main(model_name: str):
    parser = argparse.ArgumentParser(description="Train Mamba sequence model on fused AV features (RAVDESS).")
    parser.add_argument("--backbone", type=str, choices=["mobilenetv2_100",
                                                        "efficientnet_b0", 
                                                        "mobilenetv3_small_100", 
                                                        "mobilenetv3_large_100",
                                                        "mobilenetv4_conv_small_050.e3000_r224_in1k",
                                                        "mobilevit_s.cvnets_in1k",
                                                        "deit_tiny_patch16_224",
                                                        "tf_efficientnetv2_s.in21k_ft_in1k",
                                                        "efficientnet_b1"], default=None,
                                                        help="If omitted, uses BACKBONE_NAME from scripts.config")

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)


    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--pool", type=str, choices=["attn","mean","topk"], default="mean")

    # TIN ( temporal instance normalization) options
    parser.add_argument("--tin", type=str, choices=["off","permod","fused"], default="off")   # per modality or fused
    parser.add_argument("--tin-affine",action="store_true")

    # augs
    parser.add_argument("--use-specaug", action="store_true",default=False)
    parser.add_argument("--use-moddrop", action="store_true",default=False)

    parser.add_argument("--seed", type=int, default=42) 

    # CV
    # location of files with folds and splits indexes ( actors)
    parser.add_argument("--folds-json", type=str, default=None,  help="Path to 5-CV folds JSON. Each item may include 'val' (required), and optionally 'test' indexes.")
    # ablation phases 0, 1,2,3 pass it as 0: train and evaluate on single fixed fold
    parser.add_argument("--fold-idx", type=int, default=None,    help="Which fold index to run (0-based). If omitted, runs ALL folds sequentially (aggregation mode).")
    # ablation phases 0, 1,2,3  use this flag so that best models are not evaluated on test split. Not needed for ablation hyperparams sweeps
    parser.add_argument("--skip-test", action="store_true", help="Skip test evaluation/logging (useful for hyperparam sweeps).")
    
    parser.add_argument("--early-metric", type=str, choices=["acc","uar"], default="acc", help="Metric used for ReduceLROnPlateau and early stopping (default: acc).")
    parser.add_argument("--uar-floor", type=float, default=0.0,  help="Minimum acceptable UAR during early-stop selection (guardrail). Set 0.0 to disable.")
    parser.add_argument("--uar-warmup", type = int, default = 6)
    parser.add_argument("--report-json", type=str, default=None,  help="If set, write a per-fold and aggregate summary JSON here in CV mode.")


    parser.add_argument("--asap-epochs", type=int, default=6)
    parser.add_argument("--asap-cutoff", type=float, default=0.45)

    #--ablation-phase=0  
    parser.add_argument("--ablation-phase", type=int, default=0, choices=[0,1,2,3,4,5,6], help="For logging only")

    parser.add_argument("--retrain-trainval", action="store_true", default=False)
                        
    args = parser.parse_args()

    backbone = args.backbone or BACKBONE_NAME



    MODELS_DIR  = STORAGE_ROOT / "models" / backbone        # best saved models
    FEATURE_DIR = OUTPUTS_DIR  / "features" / backbone
    EXPER_DIR   = OUTPUTS_DIR  / "experiments" / backbone   # runs results     
    EXPER_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    RESULTS_MAMBA_CSV       = EXPER_DIR / "mamba_results.csv"    
    RESULTS_MAMBA_CV_AGG_CSV= EXPER_DIR / "mamba_CV_results.csv"                # 5 fold CV and average over folds



    print(f"[cfg] features={FEATURE_DIR}")
    print(f"[cfg] models={MODELS_DIR}  results={RESULTS_MAMBA_CSV}")


    # get feature dimensions from meta.json file, created by extractor
    meta_path = FEATURE_DIR / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found at: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    VIDEO_DIM = int(meta["video_feature_dim"])
    AUDIO_DIM = int(meta["audio_feature_dim"])
    FUSE_T_STEPS = int(meta.get("fuse_steps", 64))
    CONCAT_DIM = int(meta.get("concat_feature_dim", VIDEO_DIM + AUDIO_DIM))

    print(f"[meta] backbone={meta.get('backbone','?')} "
        f"V={VIDEO_DIM} A={AUDIO_DIM} concat={CONCAT_DIM} T={FUSE_T_STEPS}")



    dataset = PackedRavdess(PROJECT_ROOT, variant="fused_concat",backbone=backbone)  
    gen = torch.Generator().manual_seed(args.seed)
    
    in_dim = dataset.seq_dim
    num_classes = dataset.num_classes

    def set_global_seed(seed: int):
 #       import os, random, numpy as np, torch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Strict determinism (optional; slower, but bulletproof):
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


    # train validate test on single fold
    def run_single_fold(train_actors, val_actors, test_actors, cv_foldID=1):

        fold_seed = args.seed + int(cv_foldID)
        set_global_seed(fold_seed)
        gen_fold = torch.Generator().manual_seed(fold_seed)
        
        splits = split_indices_by_actor_from_items(dataset, train_actors, val_actors, test_actors)
        print("Split sizes:", {k: len(v) for k, v in splits.items()})


        # Create dictionary with DataLoaders for train, val, test sets
        loaders = make_loaders(
            dataset, 
            splits,
            batch_size=args.batch_size,
            collate_fn=pad_pack,
            num_workers=4,                 # <= reduce worker RAM 
            pin_memory=torch.cuda.is_available(),
                                    prefetch_factor=4,
                                    generator=gen_fold,
        )
        

     

        print("Using MAMBA sequence model.")   

        # get stats from 'train' split only
        mu_np, std_np, _ = get_feature_norm_frames_fromloaders(loaders)
        mu  = torch.from_numpy(mu_np).to(device=device, dtype=torch.float32)
        std = torch.from_numpy(std_np).to(device=device, dtype=torch.float32)    
        print("mu len =", mu.numel())

        n_classes = dataset.num_classes
        d_in = dataset.seq_dim               #ds.input_dim  # fused D
        model = AVMambaClassifier(
            d_in=d_in, 
            n_classes=n_classes, 
            d_model=args.d_model, 
            n_layers=args.layers,
            p_drop=args.dropout, 
            audio_dim=AUDIO_DIM, 
            video_dim=VIDEO_DIM,
            mu = mu,
            std=std, 
            device=device,
            pool_type=args.pool,                  
            use_tin=args.tin,                     
            tin_affine=args.tin_affine,                    
            use_specaug =args.use_specaug,
            use_moddrop =args.use_moddrop,
        ).to(device)

      
        # get labels' weights for CrossEntropyLoss
        train_labels = [dataset.items[i].label_idx for i in loaders["train"].dataset.indices]
        classes = np.arange(dataset.num_classes)
        y = np.array(train_labels)

        # get class weights for unbalanced RAVDESS datasets
        use_weights = (not args.no_class_weights)
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y)  # = N/(K*n_c)
        weights = torch.tensor(w, dtype=torch.float32, device=device)
       
        # use weights for each class with CrossEntropyLoss
        crit = nn.CrossEntropyLoss(weight=weights.to(device) if use_weights else None, label_smoothing=args.label_smoothing)


        # optimizer + scheduler
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # if args.use_swa:
        #     from torch.optim.swa_utils import AveragedModel, SWALR
        #     swa_model = AveragedModel(model)
        #     swa_start = args.swa_start if args.swa_start > 0 else max(1, int(0.7 * args.epochs))
        #     swa_scheduler = SWALR(opt, swa_lr=args.lr * 0.5)
        #     swa_updates = 0

        # steady the last epochs
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

   

        

        t1 = time.time()

        # training loop with Accuracy early metric + UAR guardrail
        # early stopping setup, initialised once
        epochs_no_improve = 0
        patience = args.patience

        best_saved = -1.0
        best_seen  = -1.0
        best_epoch_saved = 0
        best_epoch_seen  = 0
        best_seen_state  = None

        # get unique filename for 'best metrics' checkpoint file
        ts  = time.strftime("%Y%m%d-%H%M%S")      # wall-clock second
        pid = os.getpid()                         # process id
        rnd = secrets.token_hex(6)                # 12 hex chars (48-bit)

        name = f"mamba_f{ts}_{pid}_{rnd}.pt"
        ckpt_path = MODELS_DIR / name


        for epoch in range(1, args.epochs+1):
            
            # train
            model.train()
            seen = correct = total_loss = 0
            for X, lengths, y in loaders["train"]:
                
                X = X.to(device)
                lengths = lengths.to(device)
                y = y.to(device)
                
                opt.zero_grad(set_to_none=True)

                in_X, in_L = X, lengths

                logits = model(in_X, in_L)

                # Cross Entropy loss
                base_loss = crit(logits, y)

                loss = base_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                

                b = y.size(0)
                total_loss += loss.item() * b
                correct    += (logits.argmax(1) == y).sum().item()
                seen       += b

            train_loss = total_loss / max(seen, 1)
            train_acc  = correct / max(seen, 1)
            print(f"Epoch {epoch:02d} train loss {train_loss:.4f} acc {train_acc:.3f}", end=" | ")


            
            # val
            model.eval()
            seen = correct = total_loss = 0
            y_true, y_pred = [], []
            with torch.no_grad():
                for X, lengths, y in loaders["val"]:
                    X, lengths, y = X.to(device), lengths.to(device), y.to(device)

                    logits = model(X, lengths)
                    loss = crit(logits, y)

                    b = y.size(0)
                    total_loss += loss.item() * b
                    correct  += (logits.argmax(1) == y).sum().item()
                    seen       += b

                    y_true.extend(y.tolist())
                    y_pred.extend(logits.argmax(1).tolist())
                    

            val_loss = total_loss / max(seen, 1) 
            val_acc = correct / max(seen, 1)                                         

            val_precw = precision_score(y_true, y_pred, average="weighted", zero_division=0) 
            val_uar = recall_score(y_true, y_pred, average="macro", zero_division=0)
            

            # get metric for early stopping
            early_val = val_acc if args.early_metric == "acc" else val_uar

            print(f"Epoch {epoch:02d} | val loss {val_loss:.4f} acc {val_acc:.3f} UAR {val_uar:.3f} prec_w {val_precw:.3f}")
            
            sched.step(early_val)

            # UAR guardrail: ignore floor during warmup, enforce after
            if args.uar_floor <= 0.0:
                uar_ok = True
            else:
                if epoch < args.uar_warmup:
                    uar_ok = True                           # warmup: don't gate saving
                else:
                    uar_ok = (val_uar >= args.uar_floor)



            # always track best_seen (ignores guardrail)
            if early_val > best_seen + 1e-4:
                best_seen = early_val
                best_epoch_seen = epoch
                best_seen_state = {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "in_dim": in_dim, "num_classes": num_classes,
                }

            if uar_ok and (early_val > best_saved + 1e-4):
                    print(f"Validation acc improved from {best_saved:.4f} to {early_val:.4f}. Saving model.")
                    torch.save({
                        "model": model.state_dict(),
                        "in_dim": in_dim,
                        "num_classes": num_classes,
                    }, ckpt_path)
                    best_saved = early_val
                    best_epoch_saved = epoch
                    epochs_no_improve = 0

            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping after {patience} epochs with no improvement.")
                    break




            # ASAP (use best_seen)
            if epoch == args.asap_epochs and best_seen < args.asap_cutoff:
                print(f"[asap] stopping: best {args.early_metric} {best_seen:.3f} < {args.asap_cutoff}")
                break    

        
        # after all epochs:
        print(f"Best validation {args.early_metric} (saved)={best_saved:.3f} at epoch {best_epoch_saved}; "
                f"best-seen={best_seen:.3f} at epoch {best_epoch_seen}")
        
        print(f"[{time.strftime('%T')}] training took {time.time()-t1:.1f}s")
        
        # Fallback if UAR floor blocked all saves
        if not ckpt_path.exists():
            if best_seen_state is None:
                raise RuntimeError("Training finished with no progress; nothing to save.")
            print("[info] no checkpoint passed UAR floor; saving best-seen weights instead.")
            torch.save(best_seen_state, ckpt_path)
        


        # (D) collect best VAL scores; optionally TEST if not args.skip-test
        import inspect
        kw = {}
        if "weights_only" in inspect.signature(torch.load).parameters:
            kw["weights_only"] = True
        ckpt = torch.load(ckpt_path, map_location=device, **kw)
        model.load_state_dict(ckpt["model"]); 
        model.to(device)

        n_params  = count_params(model)
        model_size = file_size_mb(ckpt_path)
        
        


        # val scores
        p_val_ckpt, y_val_ckpt = collect_preds(model, loaders["val"], device, normalize_batch=None)
        val_scores = compute_scores(y_val_ckpt, p_val_ckpt)
        

        print(
        f"VAL:  acc={val_scores['acc']:.3f}, "
        f"UAR(macro)={val_scores['uar']:.3f}, "
        f"prec_w={val_scores['precision_weighted']:.3f}, "
        f"rec_w={val_scores['recall_weighted']:.3f}, "
        f"f1_w={val_scores['f1_weighted']:.3f} "
        f"| prec_m={val_scores['precision_macro']:.3f}, rec_m={val_scores['recall_macro']:.3f}, f1_m={val_scores['f1_macro']:.3f}"
        )

        
        # test scores
        test_scores = None
        if not args.skip_test and len(loaders["test"].dataset) > 0:
            p_test, y_test = collect_preds(model, loaders["test"], device, normalize_batch=None)
            if p_test is not None:
                test_scores = compute_scores(y_test, p_test)
        else:
            # did not run on test splits
            print("[info] empty TEST split for this fold; skipping test eval.")

            test_scores = {     "acc": 0.0, 
                                "uar": 0.0,    

                                "f1_macro": 0.0,                                 
                                "precision_macro": 0.0,
                                "recall_macro": 0.0, 

                                "precision_weighted": 0.0,
                                "recall_weighted": 0.0, 
                                "f1_weighted": 0.0,
                            }

        print(
            f"TEST: acc={test_scores['acc']:.3f}, "
            f"UAR(macro)={test_scores['uar']:.3f}, "
                f"prec_w={test_scores['precision_weighted']:.3f}, "
                f"rec_w={test_scores['recall_weighted']:.3f}, "
                f"f1_w={test_scores['f1_weighted']:.3f} "
                f"| prec_m={test_scores['precision_macro']:.3f}, rec_m={test_scores['recall_macro']:.3f}, f1_m={test_scores['f1_macro']:.3f}"
            )

        # delete checkpoint file during hyper params sweep. Keep after final CV evaluation
        if args.ablation_phase < 4:               
            try:
                if ckpt_path: ckpt_path.unlink(missing_ok=True)
                print(f"deleted checkpoint file {ckpt_path}")
            except Exception:
                pass

        ########################################################################
        #
        # FINALIZE: Retrain the best model on train + val splits, then test once
        #
        ########################################################################

        retrain_test_scores=None
        if not args.retrain_trainval:
            retrain_test_scores = { "acc": 0.0, 
                                    "uar": 0.0,    

                                    "f1_macro": 0.0,                                 
                                    "precision_macro": 0.0,
                                    "recall_macro": 0.0, 

                                    "precision_weighted": 0.0,
                                    "recall_weighted": 0.0, 
                                    "f1_weighted": 0.0,
                                }
        print(f"test length: {len(loaders['test'].dataset)}, best_epoch_saved: {best_epoch_saved}, retrain_trainval: {getattr(args, 'retrain_trainval', False)} ")

        if getattr(args, "retrain_trainval", False) and len(loaders["test"].dataset) > 0 and best_epoch_saved > 0:
            
            print(f"\n[retrain] Retrain on TRAIN + VAL using best_epoch_saved = {best_epoch_saved}")

            # in CV, we iterate through multiple folds so cv_foldID is set.
            # setattr(args, "fold_idx", cv_foldID)

            def _model_builder(mu_tv, std_tv):
                return AVMambaClassifier(
                    d_in=d_in, 
                    n_classes=n_classes, 
                    d_model=args.d_model, 
                    n_layers=args.layers,
                    p_drop=args.dropout, 
                    audio_dim=AUDIO_DIM, 
                    video_dim=VIDEO_DIM,
                    mu= mu_tv,             # recalculated on train+val
                    std=std_tv,             # recalculated on train+val
                    device=device,
                    pool_type=args.pool,                  
                    use_tin=args.tin,                     
                    tin_affine=args.tin_affine,                    
                    use_specaug =None,
                    use_moddrop =None,
                ).to(device)

            
            #use a **different** but deterministic seed
            retr_seed = args.seed + 1000 + int(cv_foldID)
            set_global_seed(retr_seed)
            gen_retrain = torch.Generator().manual_seed(retr_seed)


            # Run finalize: cached μ/σ via get_feature_norm_frames_fromloaders inside the helper
            retrain_test_scores, trainval_epochs = retrain_trainval_then_test(
                dataset=dataset,
                splits=splits,
                model_builder_fn=_model_builder,
                device=device,
                best_epoch_saved=best_epoch_saved,
                args=args,  # contains lr, weight_decay, label_smoothing, backbone, fold_idx
                collect_preds_fn=collect_preds,
                compute_scores_fn=compute_scores,
                generator=gen_retrain,                
                model_name="mamba"

            )


            print(
                    f"[retrained][TEST] acc={retrain_test_scores['acc']:.3f}, "
                    f"UAR={retrain_test_scores['uar']:.3f}, "
                    f"prec_w={retrain_test_scores['precision_weighted']:.3f}, "
                    f"rec_w={retrain_test_scores['recall_weighted']:.3f}, "
                    f"f1_w={retrain_test_scores['f1_weighted']:.3f} "
                    f"| prec_m={retrain_test_scores['precision_macro']:.3f}, "
                    f"rec_m={retrain_test_scores['recall_macro']:.3f}, "
                    f"f1_m={retrain_test_scores['f1_macro']:.3f}"
                )

        split_train = sorted({dataset.items[i].actor_id for i in splits["train"]})
        split_val = sorted({dataset.items[i].actor_id for i in splits["val"]})
        split_test = sorted({dataset.items[i].actor_id for i in splits["test"]})




        logger = ResultsLogger(RESULTS_MAMBA_CSV)
        logger.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": args.backbone,
            "model": "mamba",
            "seed": SEED,
            "num_params": n_params,
            "model_size_mb": model_size,

            "val_acc": round(val_scores["acc"], 4),
            "val_uar": round(val_scores["uar"], 4),

            "val_f1_macro": round(val_scores["f1_macro"], 4),
            "val_precision_macro": round(val_scores["precision_macro"], 4),
            "val_recall_macro": round(val_scores["recall_macro"], 4),

            "val_f1_weighted": round(val_scores["f1_weighted"],4),
            "val_precision_weighted": round(val_scores["precision_weighted"],4),
            "val_recall_weighted": round(val_scores["recall_weighted"],4),
            

            "test_acc": round(test_scores["acc"], 4),
            "test_uar": round(test_scores["uar"], 4),

            "test_f1_macro": round(test_scores["f1_macro"], 4),
            "test_precision_macro": round(test_scores["precision_macro"], 4),
            "test_recall_macro": round(test_scores["recall_macro"], 4),

            "test_f1_weighted": round(test_scores["f1_weighted"],4),
            "test_precision_weighted": round(test_scores["precision_weighted"],4),
            "test_recall_weighted": round(test_scores["recall_weighted"],4),

            "retrain_test_acc": round(retrain_test_scores["acc"], 4),
            "retrain_test_uar": round(retrain_test_scores["uar"], 4),

            "retrain_test_f1_macro": round(retrain_test_scores["f1_macro"], 4),
            "retrain_test_precision_macro": round(retrain_test_scores["precision_macro"], 4),
            "retrain_test_recall_macro": round(retrain_test_scores["recall_macro"], 4),

            "retrain_test_f1_weighted": round(retrain_test_scores["f1_weighted"], 4),
            "retrain_test_precision_weighted": round(retrain_test_scores["precision_weighted"], 4),
            "retrain_test_recall_weighted": round(retrain_test_scores["recall_weighted"], 4),




            "notes": f"features={FEATURE_DIR.name}",
            
            "d_model": args.d_model,
            "dropout": args.dropout,
            "layers": args.layers, 

            "epochs": args.epochs,    
            "patience": args.patience,
            "batch_size": args.batch_size,

            "lr": args.lr,
            "weight_decay": args.weight_decay,

            "label_smoothing": args.label_smoothing,
            "no_class_weights": args.no_class_weights,
            "pool": args.pool,

            "tin": args.tin,
            "tin_affine": args.tin_affine,

            "use_specaug": args.use_specaug,
            "use_moddrop": args.use_moddrop,

            "seed": args.seed,

    

            "audio_dim": AUDIO_DIM_RUNTIME, 
            "video_dim": VIDEO_DIM,
   

            "cv_foldID": cv_foldID,
            "train_actors": "-".join(str(x) for x in split_train),  
            "val_actors": "-".join(str(x) for x in split_val),
            "test_actors": "-".join(str(x) for x in split_test),


            "ablation_phase":args.ablation_phase

        })

        return val_scores, test_scores, retrain_test_scores, splits
        


    #***************************************************************************************
    #
    #       end of run_single_fold()
    #
    # ***************************************************************************************

    print("feature_dir =", dataset.feature_dir)
    print("seq_dim =", dataset.seq_dim)

    # load one file to be sure
   # import numpy as np, os
    x0, y0 = dataset[0]
    print("sample file shape:", x0.shape, "label:", y0.item())
    print(tuple(x0.shape), int(y0))  # e.g., (64, 2560), 3



    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = dataset.seq_dim
    num_classes = dataset.num_classes
    print("Using Mamaba sequence model.")
    print("input_dim =", in_dim, "num_classes =", num_classes)


    # check backbone consistency
    AUDIO_DIM_RUNTIME = in_dim - VIDEO_DIM  

    assert AUDIO_DIM_RUNTIME == AUDIO_DIM, \
        f"Dim mismatch: runtime audio={AUDIO_DIM_RUNTIME}, meta audio={AUDIO_DIM}"
    assert in_dim == VIDEO_DIM + AUDIO_DIM, \
        f"Seq dim {in_dim} != V({VIDEO_DIM})+A({AUDIO_DIM})"
    

    # Cross Validation: build collection of splits for each fold
    if args.folds_json is not None:
        # CV driver: get fold splits from JSON
        if not os.path.isfile(args.folds_json):
            raise ValueError(f"Folds JSON file not found: {args.folds_json}")
        
        ALL_ACTORS = sorted({it.actor_id for it in dataset.items})

        fold_specs = []
        folds_path = Path(args.folds_json)
        folds = json.load(open(folds_path, "r"))

        if args.fold_idx is not None:
            folds = [folds[args.fold_idx]]  # single-fold run (best for sweeps)


        # for each fold, build fold's splits indexes
        for i, fold in enumerate(folds):
            val  = set(fold.get("val", []))
            test = set(fold.get("test", []))   # may be empty in sweeps

            # safety checks
            overlap = val & test
            if overlap:
                raise ValueError(f"Fold {i}: actors overlap in val & test: {sorted(overlap)}")

            # build TRAIN = ALL_ACTORS − (VAL ∪ TEST)
            train = sorted(list(set(ALL_ACTORS) - (val | test)))

            # add current fold's splits to the collection
            fold_specs.append({
                "train": train,
                "val":   sorted(list(val)),
                "test":  sorted(list(test)),
            })

            # optional: quick sanity print
            print(f"Fold {i}: train={len(train)} val={len(val)} test={len(test)}")

    else:
        # single split from config        
        fold_specs = [{
            "train": TRAIN_ACTORS,
            "val":   VAL_ACTORS,
            "test":  TEST_ACTORS,
        }]
        print(f"Single split: train={len(TRAIN_ACTORS)} val={len(VAL_ACTORS)} test={len(TEST_ACTORS)}")


    # run acros folds: 
    #      run wiht single fold for ablation or 
    #      run accros 5 folds for final test metrics ( optional)
    #
    #   note: when running over the fold's splits: train -> val 
    #      there is option to test best model on test
    #      there is option to re-train best model on train+val and test on test


    all_val, all_test, all_retrain_test = [], [],[]

    #append val and test scores, from each fold run

    for k, spec in enumerate(fold_specs):
        print(f"\n========== FOLD {k+1}/{len(fold_specs)} ==========")

        # train-> val -> test (optional test ) -> retrain best on train+val -> test ( optional retrain-test)
        val_scores, test_scores, retrain_test_scores, _ = run_single_fold(spec["train"], spec["val"], spec["test"],cv_foldID=k)
        
        # scores on val
        all_val.append(val_scores)
        
        # scores from best model on test split
        if not args.skip_test: all_test.append(test_scores)

         # scores from best model retrained on train + val and tested on test
        if args.retrain_trainval: all_retrain_test.append(retrain_test_scores)  



        # calculate aggregate values across folds, for all metrics except "cm", from supplied list of dicts ( all_val or all_test)
    def cv_agg(list_of_scores_dicts):
        #import numpy as np
        m_keys = list(list_of_scores_dicts[0].keys())
        out = {}

        # iterate over all metric keys:  "acc", "uar", "f1_macro", "precision_macro", "recall_macro", "f1_weighted", "precision_weighted", "recall_weighted", "cm"
        for metric_key in m_keys:     
            if metric_key == "cm":   # skip confusion matrices in aggregate
                continue
            
            # get values from all folds, skip non-numeric
            vals = [fold_scores[metric_key] for fold_scores in list_of_scores_dicts if isinstance(fold_scores[metric_key], (int,float,np.floating))]
            if not vals: continue

            # compute mean and stddev, for each metric
            out[metric_key+"_mean"] = float(np.mean(vals))
            out[metric_key+"_std"]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return out


    # CV logging
    if len(fold_specs) > 1:



        # get valuation aggregated metrics
        agg_val  = cv_agg(all_val)
        print(f"\n[CV] VAL aggregate: "
            f"Acc {agg_val['acc_mean']:.3f}±{agg_val['acc_std']:.3f} | "
            f"UAR {agg_val['uar_mean']:.3f}±{agg_val['uar_std']:.3f} | "
            f"F1_m {agg_val['f1_macro_mean']:.3f}±{agg_val['f1_macro_std']:.3f} |"
            f"Prec_w {agg_val['precision_weighted_mean']:.3f}±{agg_val['precision_weighted_std']:.3f} | "
            f"Recall_w {agg_val['recall_weighted_mean']:.3f}±{agg_val['recall_weighted_std']:.3f} ")
        
        agg_test = {}
        if all_test:
            # get test aggregated metrics
            agg_test = cv_agg(all_test)
            print(f"\n[CV] TEST aggregate: "
                f"Acc {agg_test['acc_mean']:.3f}±{agg_test['acc_std']:.3f} | "
                f"UAR {agg_test['uar_mean']:.3f}±{agg_test['uar_std']:.3f} | "
                f"F1_m {agg_test['f1_macro_mean']:.3f}±{agg_test['f1_macro_std']:.3f} |"
                f"Prec_w {agg_test['precision_weighted_mean']:.3f}±{agg_test['precision_weighted_std']:.3f} | "
                f"Recall_w {agg_test['recall_weighted_mean']:.3f}±{agg_test['recall_weighted_std']:.3f}")
        else:
            agg_test = {
                "acc_mean":0.0,
                "acc_std":0.0,
                "uar_mean":0.0,
                "uar_std":0.0,

                "f1_macro_mean":0.0,
                "f1_macro_std":0.0,
                "precision_weighted_mean":0.0,
                "precision_weighted_std":0.0,
                "recall_weighted_mean":0.0,
                "recall_weighted_std":0.0,
            }


        agg_retrain_test = {}
        if all_retrain_test:
            # get aggregated metrics
            agg_retrain_test = cv_agg(all_retrain_test)
            print(f"\n[CV] RETRAIN TEST aggregate: "
                f"Acc {agg_retrain_test['acc_mean']:.3f}±{agg_retrain_test['acc_std']:.3f} | "
                f"UAR {agg_retrain_test['uar_mean']:.3f}±{agg_retrain_test['uar_std']:.3f} | "
                f"F1_m {agg_retrain_test['f1_macro_mean']:.3f}±{agg_retrain_test['f1_macro_std']:.3f} |"
                f"Prec_w {agg_retrain_test['precision_weighted_mean']:.3f}±{agg_retrain_test['precision_weighted_std']:.3f} | "
                f"Recall_w {agg_retrain_test['recall_weighted_mean']:.3f}±{agg_retrain_test['recall_weighted_std']:.3f}")
        else:
            agg_retrain_test = {
                "acc_mean":0.0,
                "acc_std":0.0,
                "uar_mean":0.0,
                "uar_std":0.0,

                "f1_macro_mean":0.0,
                "f1_macro_std":0.0,
                "precision_weighted_mean":0.0,
                "precision_weighted_std":0.0,
                "recall_weighted_mean":0.0,
                "recall_weighted_std":0.0,
            }



        # log aggregated CV results
        logger = ResultsLogger(RESULTS_MAMBA_CV_AGG_CSV)   

        logger.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": args.backbone,
            "model": "mamba",
            # best model on validation split    
            "val_agg_acc_mean": round(agg_val["acc_mean"],4),
            "val_agg_acc_std": round(agg_val["acc_std"],4),
            "val_agg_uar_mean": round(agg_val["uar_mean"],4),
            "val_agg_uar_std": round(agg_val["uar_std"],4),
            "val_agg_f1_macro_mean": round(agg_val["f1_macro_mean"],4),
            "val_agg_f1_macro_std": round(agg_val["f1_macro_std"],4),
            "val_agg_prec_w_mean":  round(agg_val["precision_weighted_mean"],4),
            "val_agg_prec_w_std":  round(agg_val["precision_weighted_std"],4),
            "val_agg_recall_w_mean":  round(agg_val["recall_weighted_mean"],4),
            "val_agg_recall_w_std":  round(agg_val["recall_weighted_std"],4),

            # best model tested on test split
            "test_agg_acc_mean": round(agg_test["acc_mean"],4),
            "test_agg_acc_std": round(agg_test["acc_std"],4),
            "test_agg_uar_mean": round(agg_test["uar_mean"],4),
            "test_agg_uar_std": round(agg_test["uar_std"],4),
            "test_agg_f1_macro_mean": round(agg_test["f1_macro_mean"],4),
            "test_agg_f1_macro_std": round(agg_test["f1_macro_std"],4),
            "test_agg_prec_w_mean":  round(agg_test["precision_weighted_mean"],4),
            "test_agg_prec_w_std":  round(agg_test["precision_weighted_std"],4),
            "test_agg_recall_w_mean":  round(agg_test["recall_weighted_mean"],4),
            "test_agg_recall_w_std":  round(agg_test["recall_weighted_std"],4),

            # results from best model retrained on train + validation splits and teste on test split
            "retrain_test_agg_acc_mean": round(agg_retrain_test["acc_mean"],4),
            "retrain_test_agg_acc_std": round(agg_retrain_test["acc_std"],4),
            "retrain_test_agg_uar_mean": round(agg_retrain_test["uar_mean"],4),
            "retrain_test_agg_uar_std": round(agg_retrain_test["uar_std"],4),
            "retrain_test_agg_f1_macro_mean": round(agg_retrain_test["f1_macro_mean"],4),
            "retrain_test_agg_f1_macro_std": round(agg_retrain_test["f1_macro_std"],4),
            "retrain_test_agg_prec_w_mean":  round(agg_retrain_test["precision_weighted_mean"],4),
            "retrain_test_agg_prec_w_std":  round(agg_retrain_test["precision_weighted_std"],4),
            "retrain_test_agg_recall_w_mean":  round(agg_retrain_test["recall_weighted_mean"],4),
            "retrain_test_agg_recall_w_std":  round(agg_retrain_test["recall_weighted_std"],4),

            "seed": SEED,
            "notes": f"features={FEATURE_DIR.name}",
            
            "d_model": args.d_model,
            "dropout": args.dropout,
            "layers": args.layers, 

            "epochs": args.epochs,    
            "patience": args.patience,
            "batch_size": args.batch_size,

            "lr": args.lr,
            "weight_decay": args.weight_decay,

            "label_smoothing": args.label_smoothing,
            "no_class_weights": args.no_class_weights,
            "pool": args.pool,

            "tin": args.tin,
            "tin_affine": args.tin_affine,

            "use_specaug": args.use_specaug,
            "use_moddrop": args.use_moddrop,

            "seed": args.seed,

    

            "audio_dim": AUDIO_DIM_RUNTIME, 
            "video_dim": VIDEO_DIM,
   



            "ablation_phase":args.ablation_phase


            })

#########################################

if __name__ == "__main__":
    main("mamba")
