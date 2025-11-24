# scripts/train_sequence.py
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
from scripts.dataset_ravdess import PackedRavdess, collate_mean_pool as packed_collate_mean_pool , pad_pack as packed_pad_pack

from scripts.config import (PROJECT_ROOT, STORAGE_ROOT,OUTPUTS_DIR,TRAIN_ACTORS, VAL_ACTORS, 
                            TEST_ACTORS,BACKBONE_NAME,
                            # VIDEO_DIM,RESULTS_GRU_CSV,
                            #RESULTS_GRU_ABL1_CSV, RESULTS_GRU_ABL2_CSV, RESULTS_GRU_ABL3_CSV,
                            # RESULTS_GRU_CV_AGG_CSV,PACKED_FUSED_CONCAT_FEATURE_NORM_FRAMES_NPZ, EXPER_DIR, MODELS_DIR, FEATURE_DIR ,AUDIO_DIM,
                            )
from sklearn.utils.class_weight import compute_class_weight
from scripts.splits import split_indices_by_actor_from_items, make_loaders
import argparse
import numpy as np, torch, random
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score
import os, torch
import json
from math import isfinite
import time, os, secrets
from scripts.compute_norm_nan import get_feature_norm_frames_fromloaders
from scripts.log_results import ResultsLogger, count_params, file_size_mb, benchmark_latency_cpu
from scripts.finalise_train_val import retrain_trainval_then_test


class ModalityDropout(nn.Module):
    """
    Randomly zero the audio slice or the video slice per sample with prob ~p each.
    
    """
    def __init__(self, audio_dim: int, p: float = 0.10):
        super().__init__()
        assert 0.0 <= p <= 0.5, "Pick p in [0, 0.5] so 2p<=1"   
        self.audio_dim = audio_dim
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        
        # separate video and audio from concatenated features
        B, T, D = x.shape
        A = self.audio_dim
        V = D - A
        x_v, x_a = x[..., :V], x[..., -A:]

        # decide which modality to drop for each sample in the batch:
        #  -> drop audio if <p; drop video if in [p,2p); else keep
        #   drop_a adn  drop_v are mutualy exclusive
        choice = torch.rand(B, device=x.device)                             # get B ( batch size) random floats from interval [0,1)
        drop_a = (choice < self.p).view(B, 1, 1)                            # boolean tensor, with True at sample indexes where to drop the audio
        drop_v = ((choice >= self.p) & (choice < 2*self.p)).view(B, 1, 1)   # boolean tensor, with True at sample indexes wehre to drop the video

        #apply dropout to video features
        x_v = x_v * (~drop_v).float()

        #apply dropout to audio features
        x_a = x_a * (~drop_a).float()

        # concatenate them back together
        return torch.cat([x_v, x_a], dim=-1)
    
class AudioOnlySpecAugment(nn.Module):
    def __init__(self,
                 audio_dim: int = 1280,
                 freq_mask_ratio: float = 0.15,
                 time_mask_ratio: float = 0.20,
                 num_f: int = 1,
                 num_t: int = 1,
                 per_sample_freq: bool = True,
                 inplace: bool = False,
                 allow_zero: bool = True):  # allow zero-width masks 
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


class AudioOnlySpecAugment_Old(nn.Module):
    def __init__(self, audio_dim=1280, freq_mask_ratio=0.15, time_mask_ratio=0.20, num_f=1, num_t=1,per_sample_freq: bool = True, inplace: bool = False,):
        super().__init__()
        self.A = audio_dim 
        self.fr = freq_mask_ratio 
        self.tr = time_mask_ratio
        self.nf = num_f 
        self.nt = num_t
        self.per_sample_freq = bool(per_sample_freq)
        self.inplace = bool(inplace)
        
    def forward(self, X, lengths=None):  # X: [B,T,D]; B=number of samples in batch, T=number of time frames, D= number of concatenated features
        if not self.training: 
            return X
        
        if not self.inplace:
            X = X.clone()

         # separate video and audio from concatenated features
        B,T,D = X.shape;
        A = self.A
        V = X[...,:D-A]
        A_ = X[...,-A:]
        
        # per-sample time masks within Ti length
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=X.device)
        


        # Time masking of audio:
        # - for each audio sample, determine the maximum allowable mask length based on a percentage of the sequence's duration.
        # - randomly chooses an actual length (t) and a random starting point (t0 ) within the signal. 
        # - zero out all audio features for that specific time span, effectively removing the audio information for that moment in time.
        for b in range(B):
            # get true length of b-th sequene, ignoring the padding
            Ti = int(lengths[b].item())
            if Ti <= 0: continue
            
            # get maximum width of time mask
            max_t = max(1, int(Ti*self.tr))

            # loop over number of masks to apply (1)
            for _ in range(self.nt):
                # get random width of the mask ( up to maximum width)
                t = int(torch.randint(1, max_t + 1, (), device=X.device).item())
                
                # if mask length is the same as sequence length, start must be 0
                if Ti - t <= 0:
                    t0 = 0
                else:
                    # get random start time index. Ensure that mask cannot extend past the true length of sample
                    t0 = int(torch.randint(0, Ti - t + 1, (), device=X.device).item())
                    # apply mask and zero all audio features, accros time span of t0 annd mask width.

                A_[b, t0:t0+t, :] = 0



        # feature masking of audio: 
        # - zeroes out contiguous feature channels inside the audio slice
        # - gets a different feature band ( mask width) per sample
        if self.fr > 0 and self.nf > 0:
            # get maximum width of feature mask
            max_f = int(A * self.fr)

            if max_f > 0:
                if self.per_sample_freq:
                    # Per-sample bands → better diversity
                    # loop over number of masks to apply (1)
                    for _ in range(self.nf):
                        
                        #randomly select width of the mask, for each sample.  [B]
                        f = torch.randint(1, max_f + 1, (B,), device=X.device)

                        # get maximum possible starting index for each sample
                        f0_max = (A - f).clamp_min(0)
                    

                        # get random starting feature index, one for each sample in range: [0,f0_max]
                        f0 = torch.floor(torch.rand(B, device=X.device) * (f0_max + 1)).long()

                        # apply mask per sample in the batch
                        for b in range(B):
                                # get the mask width for the current sample b
                                fb = int(f[b].item())
                                if fb <= 0:
                                    continue
                                # gets the starting index for the current sample b
                                f0b = int(f0[b].item())
                                A_[b, :, f0b:f0b + fb] = 0

                else:
                    # one band shared across the whole batch
                    for _ in range(self.nf):
                        f = int(torch.randint(1, max_f + 1, (), device=X.device).item())
                        f0 = int(torch.randint(0, A - f + 1, (), device=X.device).item())
                        A_[:, :, f0:f0 + f] = 0

        return torch.cat([V,A_], dim=-1)
    

class AVProjectFuse(nn.Module):    
    # Project audio and video to the same dim (d), then learn a per-step gate g in (0,1)
    # to fuse: z = g*za + (1-g)*zv. Very small but reduces noise from a bad modality.
    
    def __init__(self, a_in: int, v_in: int, d: int = 256, p: float = 0.1, channel_wise: bool = False):
        super().__init__()
        self.aproj = nn.Sequential(nn.Linear(a_in, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p))
        self.vproj = nn.Sequential(nn.Linear(v_in, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p))
        self.channel_wise = channel_wise
        self.gate = nn.Linear(2 * d, d if channel_wise else 1)
        nn.init.zeros_(self.gate.bias)

    def forward(self, A: torch.Tensor, V: torch.Tensor):
        # A,V: [B, T, a_in/v_in]
        za = self.aproj(A)                          # [B,T,d]
        zv = self.vproj(V)                          # [B,T,d]
        g  = torch.sigmoid(self.gate(torch.cat([za, zv], dim=-1)))  # [B,T,d] or [B,T,1]
        z  = g * za + (1.0 - g) * zv                # [B,T,d]
        return z, za, zv, g

    
    
class GRUAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,          # e.g., 2560 (concat) or 1280 (interleaved)
        hidden: int,             # GRU hidden size per direction
        num_classes: int,
        p_drop: float = 0.2,
        proj_dim: int | None = 256,   # e.g., 192/256; None = no projection
        num_layers: int = 1,
        bidirectional: bool = True,
        use_gate: bool = False,      # enable AVProjectFuse
        audio_dim: int | None = None,# required when use_gate=True
        fuse_dim: int = 256,         # width after fusion
        channel_wise_gate: bool = False,
        fuse_mode: str = "concat", 
        pool_type: str = "attn", 
        top_k: int = 5
                       
    ):
        super().__init__()
        self.use_gate = use_gate
        self.fuse_mode = fuse_mode
        self.audio_dim = audio_dim
        self.video_dim = input_dim - (audio_dim or 0)
        self.pool_type = pool_type
        self.topk = top_k

        # Always pre-normalize the raw fused features (2560) BEFORE any gating.       
        self.pre_norm = nn.Identity()

        # Dropout to apply after fusion/projection
        self.in_drop = nn.Dropout(p_drop)

        if use_gate:
            assert audio_dim is not None and audio_dim > 0, "audio_dim required when use_gate=True"
            # project audio and video to the same width, concatenate them with sigmoid at each T
            self.fuser = AVProjectFuse(a_in=audio_dim, v_in=self.video_dim, d=fuse_dim,
                                       p=p_drop, channel_wise=channel_wise_gate)


            # Post-fusion dimensionality seen by GRU (before optional proj):
            post_fuse_dim = (fuse_dim if fuse_mode == "replace" else (input_dim + fuse_dim))
        else:
            post_fuse_dim = input_dim


       

        # final projection to GRU input (small, fast)
        if proj_dim is None:
            self.in_proj = nn.Identity()
            gru_in = post_fuse_dim
        else:
            self.in_proj = nn.Sequential(nn.Linear(post_fuse_dim, proj_dim),
                                         nn.LayerNorm(proj_dim))
            gru_in = proj_dim


        self.gru = nn.GRU(
            input_size=gru_in,              # <-- MUST match in_proj output width
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if num_layers < 2 else p_drop,  # GRU's internal dropout works only if num_layers>1
        )

        self.attn = nn.Sequential(
            nn.Linear(hidden * (2 if bidirectional else 1), hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),  # score per timestep
        )
        self.attn_drop = nn.Dropout(p_drop)
        self.out_drop = nn.Dropout(p_drop)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), num_classes)

    def forward(self, X, lengths):

        X = self.pre_norm(X)

        if self.use_gate:
            D = X.size(-1); A = self.audio_dim; V = D - A
            x_v = X[..., :V]          # [B,T, video_dim]
            x_a = X[..., -A:]         # [B,T, audio_dim]
            z, _, _, _ = self.fuser(x_a, x_v)
            X = z if self.fuse_mode == "replace" else torch.cat([X, z], dim=-1)
        
        # Dropout then projection to the GRU input width
        X = self.in_drop(X)
        X = self.in_proj(X)

        # If all lengths are equal (fixed T from extractor), skip pack/unpack for a small speedup.
        if torch.all(lengths == lengths[0]):
            # Fast path (no padding to skip)
            out, _ = self.gru(X)                       # [B, T, 2H]
            T = X.size(1)
            mask = torch.ones(X.size(0), T, dtype=torch.bool, device=X.device)
        else:
            # Variable-length path (skip padded timesteps)
            packed = nn.utils.rnn.pack_padded_sequence(
                X, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed, batch_first=True, total_length=X.size(1)  # keep T consistent with input
            )
            T = out.size(1)
            mask = torch.arange(T, device=out.device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, T]

        # Use a very negative number to mask out padded steps (and avoid -inf in fp16/bf16)
        very_neg = -1e4 if out.dtype in (torch.float16, torch.bfloat16) else torch.finfo(out.dtype).min

        if self.pool_type == "mean":
            # masked mean over time
            m = mask.unsqueeze(-1).float()               # [B,T,1]
            denom = m.sum(dim=1).clamp_min(1.0)          # [B,1]
            context = (out * m).sum(dim=1) / denom       # [B,2H]
            return self.head(self.out_drop(context))

        elif self.pool_type == "topk":
            # per-timestep logits
            # mask invalid positions before selecting top-k
            logits_t = self.head(self.out_drop(out))     # [B,T,C]
            scores   = logits_t.max(dim=-1).values       # [B,T]
            scores   = scores.masked_fill(~mask, very_neg)

            k = min(self.topk, T)
            topk_idx = scores.topk(k, dim=1).indices     # [B,k]
            # which of the selected indices are actually valid?
            sel_valid = mask.gather(1, topk_idx)         # [B,k] bool
            idx_exp   = topk_idx.unsqueeze(-1).expand(-1, -1, logits_t.size(-1))  # [B,k,C]
            topk_logits = logits_t.gather(1, idx_exp)    # [B,k,C]
            topk_logits = topk_logits * sel_valid.unsqueeze(-1).float()
            denom = sel_valid.sum(dim=1).unsqueeze(-1).clamp_min(1.0)             # [B,1]
            logits = topk_logits.sum(dim=1) / denom                                # [B,C]
            return logits

        else:  # "attn" (default)
            # mask scores before softmax
            scores = self.attn(out).squeeze(-1)          # [B,T]
            scores = scores.masked_fill(~mask, very_neg)
            alpha  = torch.softmax(scores, dim=1)        # [B,T]
            alpha  = self.attn_drop(alpha).unsqueeze(-1)
            alpha  = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
            context = (alpha * out).sum(dim=1)           # [B,2H]
            return self.head(self.out_drop(context))

############ end class GRUAttentionClassifier(nn.Module):




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

    if not all_p:   # empty split
        return None, None
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



def main():

    import os, torch, re
    n = int(os.getenv("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)
    print(f"[threads] torch_intra={torch.get_num_threads()} inter={torch.get_num_interop_threads()}")

    with open("/proc/self/status") as f:
        m = re.search(r"Cpus_allowed_list:\s*(.*)", f.read())
        print(f"[cpus_allowed] {m.group(1) if m else '?'}")
    print(f"[env] OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')} MKL_NUM_THREADS={os.getenv('MKL_NUM_THREADS')}")

    parser = argparse.ArgumentParser(description="Train GRU sequence model on fused AV features (RAVDESS).")

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



    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--p-drop", type=float, default=0.25)
    
    parser.add_argument("--mod-drop", type=float, default=0.1)          # in ModalityDropout augmenter
   
    parser.add_argument("--freq-mask-ratio",type=float, default=0.00)   # in AudioOnlySpecAugment augmenter, masks audio features
    parser.add_argument("--time-mask-ratio",type=float, default=0.00)   # in AudioOnlySpecAugment augmenter, masks audio time slices
    parser.add_argument("--num_t",type=int, default=0)
    parser.add_argument("--num_f",type=int, default=0)
    

    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--proj-dim", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42) 

    # not used in ablation scripts: --use-gate is not set: default = false
    parser.add_argument("--use-gate", action="store_true", default=False)
    parser.add_argument("--fuse-dim", type=int, default=256)        
    parser.add_argument("--channel-wise-gate", action="store_true", default=False)
    parser.add_argument("--fuse-mode", type=str, choices=["concat","replace"], default="concat")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.015)

    parser.add_argument("--variant", type=str, choices=["fused_concat","fused_interleaved"], default="fused_concat")

    parser.add_argument("--pool-type", type=str, choices=["mean","topk","attn"], default="attn")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--margin", type=float, default=0.00)

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
    parser.add_argument("--ablation-phase", type=int, default=0, choices=[0,1,2,3,4], help="For logging only")

    parser.add_argument("--retrain-trainval", action="store_true", default=False)
    
    args = parser.parse_args()

    backbone = args.backbone or BACKBONE_NAME



    MODELS_DIR  = STORAGE_ROOT / "models" / backbone        # best saved models
    FEATURE_DIR = OUTPUTS_DIR  / "features" / backbone
    EXPER_DIR   = OUTPUTS_DIR  / "experiments" / backbone   # runs results     
    EXPER_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    RESULTS_GRU_CSV       = EXPER_DIR / "gru_results.csv"    
    RESULTS_GRU_CV_AGG_CSV= EXPER_DIR / "gru_CV_results.csv"                # 5 fold CV and average over folds



    print(f"[cfg] features={FEATURE_DIR}")
    print(f"[cfg] models={MODELS_DIR}  results={RESULTS_GRU_CSV}")


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






    if args.proj_dim == 0:
        args.proj_dim = None

    t0 = time.time()

    dataset = PackedRavdess(PROJECT_ROOT, variant=args.variant,backbone=backbone)  
    gen = torch.Generator().manual_seed(args.seed)



    # train validate test on single fold
    def run_single_fold(train_actors, val_actors, test_actors, cv_foldID=1):
        
        # (A) build splits + loaders
        splits = split_indices_by_actor_from_items(dataset, train_actors, val_actors, test_actors)
        print("Split sizes:", {k: len(v) for k, v in splits.items()})

        loaders = make_loaders( dataset, 
                                splits, 
                                batch_size=32, 
                                collate_fn=packed_pad_pack,
                                num_workers=4,  
                                pin_memory=torch.cuda.is_available(),
                                prefetch_factor=4,
                                generator=gen,
                                )
        


        model = GRUAttentionClassifier(
            input_dim=in_dim,
            hidden=args.hidden,
            num_classes=num_classes,
            p_drop=args.p_drop,
            num_layers=args.num_layers,                
            bidirectional=True,
            proj_dim=args.proj_dim,                     # no projection, directly use LayerNormed input
            use_gate=args.use_gate,                     # enable the gate
            audio_dim=AUDIO_DIM_RUNTIME,                # audio is the LAST slice, concatenated to video by extractor
            fuse_dim=args.fuse_dim,                     # output dim of the fuser (GRU input size)
            channel_wise_gate=args.channel_wise_gate,   # try True later for a tiny extra boost
            fuse_mode=args.fuse_mode,
            pool_type=args.pool_type,
            top_k=args.top_k                            # used with pool_type="topk"
        ).to(device)



        # get labels' weights for CrossEntropyLoss
        train_labels = [dataset.items[i].label_idx for i in loaders["train"].dataset.indices]
        classes = np.arange(dataset.num_classes)
        y = np.array(train_labels)

        # get class weights for unbalanced RAVDESS datasets
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y)  # = N/(K*n_c)
        weights = torch.tensor(w, dtype=torch.float32, device=device)
        
        # use weights for each class with CrossEntropyLoss
        crit = nn.CrossEntropyLoss(weight=weights.to(device),label_smoothing=args.label_smoothing)  # optionally use label smoothing

        # get optimizer
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)    
        
        #  get scheduler for steady the last epochs
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)



        mu_np, std_np, _ = get_feature_norm_frames_fromloaders(loaders)
        mu  = torch.from_numpy(mu_np).to(device=device, dtype=torch.float32)
        std = torch.from_numpy(std_np).to(device=device, dtype=torch.float32)
        ###############

        print("mu len =", mu.numel())

        def normalize_batch(X):
            return (X - mu) / (std + 1e-8)

        # FAIL FAST if modality doesn’t match
        assert mu.numel() == dataset.seq_dim, (
            f"Normalization dim {mu.numel()} != dataset dim {dataset.seq_dim}. "
            f"feature_dir={dataset.feature_dir}"
        )
    
        t1 = time.time()
        
        # (C) training loop with Accuracy early metric + UAR guardrail
        # early stopping setup, initialised once
        best = -1.0
        best_epoch = 0
        epochs_no_improve = 0
        patience = 15

        best_saved = -1.0
        best_seen  = -1.0
        best_epoch_saved = 0
        best_epoch_seen  = 0
        best_seen_state  = None

        # get unique filename for 'best metrics' checkpoint file
        ts  = time.strftime("%Y%m%d-%H%M%S")      # wall-clock second
        pid = os.getpid()                         # process id
        rnd = secrets.token_hex(6)                # 12 hex chars (48-bit)
        name = f"gru_f{cv_foldID}_{ts}_{pid}_{rnd}.pt"
        ckpt_path = MODELS_DIR / name
        
        # Augmentations    
        spec_augmenter = AudioOnlySpecAugment(audio_dim=AUDIO_DIM_RUNTIME, 
                                              freq_mask_ratio=args.freq_mask_ratio, 
                                              time_mask_ratio=args.time_mask_ratio,
                                              num_f=args.num_f, num_t=args.num_t, 
                                              inplace=False).to(device)
        
        mod_drop = ModalityDropout(audio_dim=AUDIO_DIM_RUNTIME, p=args.mod_drop).to(device)
        
        for epoch in range(1, 51):
            
            # train
            model.train()
            seen = correct = total_loss = 0
            for X, lengths, y in loaders["train"]:

                X = X.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)



                # Augs — train only           

                X = normalize_batch(X)
                X = spec_augmenter(X,lengths)   # sets to 0 (≈ mean after z-score)
                X = mod_drop(X)                 # randomly drop one modality per sample

                opt.zero_grad(set_to_none=True)
                logits = model(X, lengths)


                # Cross Entropy loss
                loss = crit(logits, y)


                loss.backward()

                # clip gradients to avoid exploding gradients
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
                    X = normalize_batch(X)

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
        model.load_state_dict(ckpt["model"]) 
        model.to(device)

        n_params  = count_params(model)
        model_size = file_size_mb(ckpt_path)


        # val scores
        p_val, y_val = collect_preds(model, loaders["val"], device, normalize_batch=normalize_batch)
        val_scores = compute_scores(y_val, p_val)

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
            p_test, y_test = collect_preds(model, loaders["test"], device, normalize_batch=normalize_batch)
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
            retrain_test_scores = {   "acc": 0.0, 
                                    "uar": 0.0,    

                                    "f1_macro": 0.0,                                 
                                    "precision_macro": 0.0,
                                    "recall_macro": 0.0, 

                                    "precision_weighted": 0.0,
                                    "recall_weighted": 0.0, 
                                    "f1_weighted": 0.0,
                                }


        if getattr(args, "retrain_trainval", False) and len(loaders["test"].dataset) > 0 and best_epoch_saved > 0:
            print(f"\n[retrain] Retrain on TRAIN + VAL using best_epoch_saved = {best_epoch_saved}")

            # Make sure the finalize module can distinguish cache keys per fold.
            # set args.fold_idx = cv_foldID (restored afterward) here and set it back later
            # _had_fold_idx = hasattr(args, "fold_idx")
            # _prev_fold_idx = getattr(args, "fold_idx", None)


            # # in CV, we iterate through multiple folds so cv_foldID is set.
            # setattr(args, "fold_idx", cv_foldID)

            # Model factory: build EXACT same model with same hparams as above
            def _model_builder():
                return GRUAttentionClassifier(
                    input_dim=in_dim,
                    hidden=args.hidden,
                    num_classes=num_classes,
                    p_drop=args.p_drop,
                    num_layers=args.num_layers,
                    bidirectional=True,
                    proj_dim=args.proj_dim,
                    use_gate=args.use_gate,
                    audio_dim=AUDIO_DIM_RUNTIME,
                    fuse_dim=args.fuse_dim,
                    channel_wise_gate=args.channel_wise_gate,
                    fuse_mode=args.fuse_mode,
                    pool_type=args.pool_type,
                    top_k=args.top_k,
                ).to(device)
            

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
                generator=gen,
                spec_augmenter_tv=spec_augmenter,
                mod_drop_tv=mod_drop,
                model_name="gru"
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

         

#####################################################





        split_train = sorted({dataset.items[i].actor_id for i in splits["train"]})
        split_val = sorted({dataset.items[i].actor_id for i in splits["val"]})
        split_test = sorted({dataset.items[i].actor_id for i in splits["test"]})


        # log results
        logger = ResultsLogger(RESULTS_GRU_CSV)   


        logger.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": args.backbone,
            "model": "gru",
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
            
            "hidden": args.hidden,
            "p_drop": args.p_drop,
            "num_layers": args.num_layers, 

            "mod_drop": args.mod_drop,    
            "freq_mask_ratio": args.freq_mask_ratio,
            "time_mask_ratio": args.time_mask_ratio,

            "proj_dim": args.proj_dim,               
            "use_gate": args.use_gate,    

            "audio_dim": AUDIO_DIM_RUNTIME, 
            "fuse_dim": args.fuse_dim,    

            "seed": args.seed,             
            "channel_wise_gate": args.channel_wise_gate ,

            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "fuse_mode": args.fuse_mode,
            "variant": args.variant,

            "pool_type": args.pool_type,
            "top_k": args.top_k,
            "margin": args.margin,

            "cv_foldID": cv_foldID,
            "train_actors": "-".join(str(x) for x in split_train),  
            "val_actors": "-".join(str(x) for x in split_val),
            "test_actors": "-".join(str(x) for x in split_test),

            "proj_dim_effective": args.proj_dim if args.proj_dim is not None else "none",
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
    import numpy as np, os
    x0, y0 = dataset[0]
    print("sample file shape:", x0.shape, "label:", y0.item())
    print(tuple(x0.shape), int(y0))  # e.g., (64, 2560), 3

    SEED = args.seed #42  #7 # 123
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = dataset.seq_dim
    num_classes = dataset.num_classes
    print("Using GRU sequence model.")
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
        import numpy as np
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
        logger = ResultsLogger(RESULTS_GRU_CV_AGG_CSV)   

        logger.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),

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


            "backbone": args.backbone,
            "model": "gru",
            "seed": SEED,


            "notes": f"features={FEATURE_DIR.name}",
            
            "hidden": args.hidden,
            "p_drop": args.p_drop,
            "num_layers": args.num_layers, 

            "mod_drop": args.mod_drop,    
            "freq_mask_ratio": args.freq_mask_ratio,
            "time_mask_ratio": args.time_mask_ratio,

            "proj_dim": args.proj_dim,               
            "use_gate": args.use_gate,    

            "audio_dim": AUDIO_DIM_RUNTIME, 
            "fuse_dim": args.fuse_dim,    

            "seed": args.seed,             
            "channel_wise_gate": args.channel_wise_gate ,

            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "fuse_mode": args.fuse_mode,
            "variant": args.variant,

            "pool_type": args.pool_type,
            "top_k": args.top_k,
            "margin": args.margin,

            "proj_dim_effective": args.proj_dim if args.proj_dim is not None else "none",
            "ablation_phase":args.ablation_phase


            })



if __name__ == "__main__":    
    main()