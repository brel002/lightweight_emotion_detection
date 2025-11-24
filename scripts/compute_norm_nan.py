
# scripts/compute_norm4.py
# has features to safely deal with nans

from pathlib import Path
import torch
import numpy as np

from scripts.splits import split_indices_by_actor_from_items, make_loaders
from scripts.config import (PROJECT_ROOT, TRAIN_ACTORS, VAL_ACTORS, TEST_ACTORS,
                            
                            # normalization stats from packed feature fies

                            # pooled features: used with MLP
                            PACKED_FUSED_CONCAT_FEATURE_NORM_POOLED_NPZ,
                            PACKED_FUSED_INTERLEAVED_FEATURE_NORM_POOLED_NPZ,
                            PACKED_AUDIO_FEATURE_NORM_POOLED_NPZ,
                            PACKED_VIDEO_FEATURE_NORM_POOLED_NPZ,

                            # frames features: used with sequence models like MAMBA, GRU
                            PACKED_FUSED_CONCAT_FEATURE_NORM_FRAMES_NPZ ,
                            PACKED_FUSED_INTERLEAVED_FEATURE_NORM_FRAMES_NPZ,
                            PACKED_AUDIO_FEATURE_NORM_FRAMES_NPZ,
                            PACKED_VIDEO_FEATURE_NORM_FRAMES_NPZ
)


import argparse
from scripts.dataset_ravdess import PackedRavdess, collate_mean_pool as packed_collate_mean_pool , pad_pack as packed_pad_pack

def get_feature_norm_frames_fromloaders(loaders):
    """
    Mean/Std over all valid frames X:[B,T,D] on TRAIN split only,
    ignoring padding and NaN/Inf per-dimension.
    """

    with torch.no_grad():
        sum_vec   = None
        sqsum_vec = None
        cnt_vec   = None

        for X, lengths, _ in loaders["train"]:     # X: [B, T, D]
            X = X.to(dtype=torch.float64)
            B, T, D = X.shape
            if sum_vec is None:
                sum_vec   = torch.zeros(D, dtype=torch.float64)
                sqsum_vec = torch.zeros(D, dtype=torch.float64)
                cnt_vec   = torch.zeros(D, dtype=torch.float64)

            # Valid-frame mask from lengths
            t_idx = torch.arange(T, device=X.device).unsqueeze(0)           # [1, T]
            pad_mask = (t_idx < lengths.unsqueeze(1)).unsqueeze(-1)         # [B, T, 1] True on valid frames

            # Finite mask
            fin_mask = torch.isfinite(X)                                    # [B, T, D]

            # Combined mask: valid frames AND finite values
            m = pad_mask & fin_mask                                         # [B, T, D]

            # Zero-out everything else for sums
            Xz = X.masked_fill(~m, 0.0)

            sum_vec   += Xz.sum(dim=(0, 1))                                  # [D]
            sqsum_vec += (Xz * Xz).sum(dim=(0, 1))                           # [D]
            cnt_vec   += m.sum(dim=(0, 1), dtype=torch.float64)              # [D]

        valid = cnt_vec > 0
        mu  = torch.zeros_like(sum_vec)
        var = torch.zeros_like(sum_vec)
        mu[valid]  = sum_vec[valid]   / cnt_vec[valid]
        var[valid] = sqsum_vec[valid] / cnt_vec[valid] - mu[valid].pow(2)

        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)
        std[~valid] = 1.0
        mu[~valid]  = 0.0

    return mu.to(torch.float32).cpu().numpy(), std.to(torch.float32).cpu().numpy(), int(cnt_vec.sum().item())




def get_feature_norm_frames(ds, collate_fn=packed_pad_pack):
    """
    Mean/Std over all valid frames X:[B,T,D] on TRAIN split only,
    ignoring padding and NaN/Inf per-dimension.
    """
    splits  = split_indices_by_actor_from_items(ds, TRAIN_ACTORS, VAL_ACTORS, TEST_ACTORS)
    loaders = make_loaders(ds, splits, batch_size=32, collate_fn=collate_fn, num_workers=0)

    with torch.no_grad():
        sum_vec   = None
        sqsum_vec = None
        cnt_vec   = None

        for X, lengths, _ in loaders["train"]:     # X: [B, T, D]
            X = X.to(dtype=torch.float64)
            B, T, D = X.shape
            if sum_vec is None:
                sum_vec   = torch.zeros(D, dtype=torch.float64)
                sqsum_vec = torch.zeros(D, dtype=torch.float64)
                cnt_vec   = torch.zeros(D, dtype=torch.float64)

            # Valid-frame mask from lengths
            t_idx = torch.arange(T, device=X.device).unsqueeze(0)           # [1, T]
            pad_mask = (t_idx < lengths.unsqueeze(1)).unsqueeze(-1)         # [B, T, 1] True on valid frames

            # Finite mask
            fin_mask = torch.isfinite(X)                                    # [B, T, D]

            # Combined mask: valid frames AND finite values
            m = pad_mask & fin_mask                                         # [B, T, D]

            # Zero-out everything else for sums
            Xz = X.masked_fill(~m, 0.0)

            sum_vec   += Xz.sum(dim=(0, 1))                                  # [D]
            sqsum_vec += (Xz * Xz).sum(dim=(0, 1))                           # [D]
            cnt_vec   += m.sum(dim=(0, 1), dtype=torch.float64)              # [D]

        valid = cnt_vec > 0
        mu  = torch.zeros_like(sum_vec)
        var = torch.zeros_like(sum_vec)
        mu[valid]  = sum_vec[valid]   / cnt_vec[valid]
        var[valid] = sqsum_vec[valid] / cnt_vec[valid] - mu[valid].pow(2)

        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)
        std[~valid] = 1.0
        mu[~valid]  = 0.0

    return mu.to(torch.float32).cpu().numpy(), std.to(torch.float32).cpu().numpy(), int(cnt_vec.sum().item())



def get_feature_norm_pooled(ds, collate_fn=packed_collate_mean_pool):
    """
    Mean/Std over pooled features X:[B,D] on TRAIN split only,
    ignoring NaN/Inf per-dimension. Returns float32 arrays.
    """
    splits  = split_indices_by_actor_from_items(ds, TRAIN_ACTORS, VAL_ACTORS, TEST_ACTORS)
    loaders = make_loaders(ds, splits, batch_size=32, collate_fn=collate_fn, num_workers=0)

    with torch.no_grad():
        sum_vec   = None           # float64 accumulators for precision
        sqsum_vec = None
        cnt_vec   = None           # per-dim finite counts: number of finite values per dimension (for example: dim = 1280 for audio or video, 2560 for fused; extracted by mobilenetv2)

        for X, _ in loaders["train"]:         # X: [B, D]; B is batch size
            X = X.to(dtype=torch.float64)     # accumulate in float64
            if sum_vec is None:
                D = X.shape[1]
                sum_vec   = torch.zeros(D, dtype=torch.float64)
                sqsum_vec = torch.zeros(D, dtype=torch.float64)
                cnt_vec   = torch.zeros(D, dtype=torch.float64)


            # safe accumulation:  ignoring NaNs/Infs
            # create finite mask: matrix of size [B,D] with True for finite values
            fin = torch.isfinite(X)           

            # zero-out non-finite values, for sums
            Xz  = X.masked_fill(~fin, 0.0)    

            # D is size of feature dimension: audio or video or fused( concatenated)
            # Get sum per dimension, for each sample in batch of size B. result is vector of size D. 
            # Accumulate over batches.
            sum_vec   += Xz.sum(dim=0)                       # [D]
            sqsum_vec += (Xz * Xz).sum(dim=0)                # [D]

            # get count of finite values per dimension, accumulate over batches. finite value is one that is not nan or inf.
            cnt_vec   += fin.sum(dim=0, dtype=torch.float64) # [D] 

        # Avoid divide-by-zero per-dim: 
        valid = cnt_vec > 0                 # bool vector of size D, indicating 
        mu  = torch.zeros_like(sum_vec)
        var = torch.zeros_like(sum_vec)

        # Compute mean and variance only for valid dimensions
        mu[valid]  = sum_vec[valid]   / cnt_vec[valid]
        var[valid] = sqsum_vec[valid] / cnt_vec[valid] - mu[valid].pow(2)

        # Numerical safety
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)
        # For dims with no valid samples, fall back to std=1.0
        std[~valid] = 1.0
        mu[~valid]  = 0.0

    # Return float32 on CPU
    return mu.to(torch.float32).cpu().numpy(), std.to(torch.float32).cpu().numpy(), int(cnt_vec.sum().item())






def main():
    import pandas as pd, numpy as np, os

    

    print(f"Using packed features")


    ds = PackedRavdess(PROJECT_ROOT, variant="fused_concat")                
    
    mu, std, n = get_feature_norm_pooled(ds,collate_fn=packed_collate_mean_pool)
    np.savez(PACKED_FUSED_CONCAT_FEATURE_NORM_POOLED_NPZ, mean=mu, std=std, n=n)
    print(f"Saved normalization stats for pooled fused_concat features to {PACKED_FUSED_CONCAT_FEATURE_NORM_POOLED_NPZ}")

    mu, std, n = get_feature_norm_frames(ds,collate_fn=packed_pad_pack)
    np.savez(PACKED_FUSED_CONCAT_FEATURE_NORM_FRAMES_NPZ, mean=mu, std=std, n=n)
    print(f"Saved normalization stats for frames fused_concat features to {PACKED_FUSED_CONCAT_FEATURE_NORM_FRAMES_NPZ}")
    
        
    
    # ds = PackedRavdess(PROJECT_ROOT, variant="fused_interleaved") 

    # mu, std, n = get_feature_norm_pooled(ds,collate_fn=packed_collate_mean_pool)
    # np.savez(PACKED_FUSED_INTERLEAVED_FEATURE_NORM_POOLED_NPZ, mean=mu, std=std, n=n)
    # print(f"Saved normalization stats for pooled fused_interleaved features to {PACKED_FUSED_INTERLEAVED_FEATURE_NORM_POOLED_NPZ}")

    # mu, std, n = get_feature_norm_frames(ds,collate_fn=packed_pad_pack)
    # np.savez(PACKED_FUSED_INTERLEAVED_FEATURE_NORM_FRAMES_NPZ, mean=mu, std=std, n=n)
    # print(f"Saved normalization stats for frames fused_interleaved features to {PACKED_FUSED_INTERLEAVED_FEATURE_NORM_FRAMES_NPZ}")

    
    ds = PackedRavdess(PROJECT_ROOT, variant="audio")   

    mu, std, n = get_feature_norm_pooled(ds,collate_fn=packed_collate_mean_pool)
    np.savez(PACKED_AUDIO_FEATURE_NORM_POOLED_NPZ , mean=mu, std=std, n=n)
    print(f"Saved normalization stats for pooled audio features to {PACKED_AUDIO_FEATURE_NORM_POOLED_NPZ}")

    mu, std, n = get_feature_norm_frames(ds,collate_fn=packed_pad_pack)
    np.savez(PACKED_AUDIO_FEATURE_NORM_FRAMES_NPZ , mean=mu, std=std, n=n)
    print(f"Saved normalization stats for frames audio features to {PACKED_AUDIO_FEATURE_NORM_FRAMES_NPZ}")





    ds = PackedRavdess(PROJECT_ROOT, variant="video")   

    mu, std, n = get_feature_norm_pooled(ds,collate_fn=packed_collate_mean_pool)
    np.savez(PACKED_VIDEO_FEATURE_NORM_POOLED_NPZ , mean=mu, std=std, n=n)
    print(f"Saved normalization stats for pooled video features to {PACKED_VIDEO_FEATURE_NORM_POOLED_NPZ}")


    mu, std, n = get_feature_norm_frames(ds,collate_fn=packed_pad_pack)
    np.savez(PACKED_VIDEO_FEATURE_NORM_FRAMES_NPZ  , mean=mu, std=std, n=n)
    print(f"Saved normalization stats for frames video features to {PACKED_VIDEO_FEATURE_NORM_FRAMES_NPZ}")


if __name__ == "__main__":
    main()

# stats from packed features
# python -m scripts.compute_norm




  
