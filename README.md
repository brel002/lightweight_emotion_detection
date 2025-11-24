# Lightweight Emotion Detection
```markdown
# Emotion Detection Model on ML Cluster

This repository contains scripts and instructions for running emotion detection experiments on the ML cluster using the RAVDESS dataset.

---

## Setup

1. **Connect to the cluster:**
   ```bash
   ssh <username>@foscsmlprd02
   cd /home/brel002/emotion_detection/
   ```

---

## GRU Example

### 1. Feature Extraction ( with efficientnet_b0 BACKBONE )
Edit `scripts/run_extract_EN.sbatch`:
- Uncomment line 69:
  ```bash
  MOD="scripts.extract_fuse_features_generic_timm_V4_24VideoTSTEPS"
  ```
- Comment out line 70:
  ```bash
  # MOD="scripts.extract_fuse_features_generic_timm_V4_32VideoTSTEPS"
  ```

Run the extractor:
```bash
sed -i 's/\r$//' scripts/run_extract_EN.sbatch
sbatch scripts/run_extract_EN.sbatch
tail -f /home/brel002/emotion_detection/outputs/logs/extract_<BID>.log
```

### 2. Check Extracted Features
Features are saved at:
```
/data/brel002/emotion_storage/outputs/features/<BACKBONE>
```

### 3. Run Ablation Experiments
- It runs over all 9 backbones' extracted features. Comment out backbones you do not want.
  
```bash
sed -i 's/\r$//' scripts/ablate_gru_all.sh
sbatch scripts/ablate_gru_all.sh
tail -n 200 -f /data/brel002/emotion_storage/logs/output_<QID>.txt
tail -n 200 -f /data/brel002/emotion_storage/logs/error_<QID>.txt
```

### 4. Check Results
Results are saved at:
```
/data/brel002/emotion_storage/outputs/experiments/<BACKBONE>
```
- `gru_results.csv`: Results from all phases.
- `gru_CV_results.csv`: Aggregated results from the final test phase.

### 5. Check Best GRU Result
Check the sorted results:
```
/data/brel002/emotion_storage/outputs/experiments/efficientnet_b0/gru_CV_results_T24.csv
```
Sort by `retrain_test_agg_acc_mean` (descending).

### 6. Best GRU model: checkpoints on each fold
( files are too big to upload to storage/models/efficientnet_b0)

on ML cluster:

/data/brel002/emotion_storage/models/efficientnet_b0/gru_f0_20251018-024158_2050509_62325429b617.pt
/data/brel002/emotion_storage/models/efficientnet_b0/gru_f1_20251018-024250_2050509_1d24a834d9f6.pt
/data/brel002/emotion_storage/models/efficientnet_b0/gru_f2_20251018-024341_2050509_08019a6fbdd3.pt
/data/brel002/emotion_storage/models/efficientnet_b0/gru_f3_20251018-024425_2050509_0d16493a8eea.pt
/data/brel002/emotion_storage/models/efficientnet_b0/gru_f4_20251018-024500_2050509_7800f7799cf2.pt


### 7. Replicate Best GRU Model
```bash
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_gru_CV \
  --backbone=efficientnet_b0 \
  --hidden=384 \
  --num-layers=1 \
  --p-drop=0.1 \
  --proj-dim=0 \
  --fuse-mode=concat \
  --fuse-dim=256 \
  --pool-type=attn \
  --seed=42 \
  --ablation-phase=4 \
  --mod-drop=0.1 \
  --freq-mask-ratio=0.0 \
  --time-mask-ratio=0.0 \
  --lr=0.001 \
  --weight-decay=0.0001 \
  --label-smoothing=0.02 \
  --margin=0 \
  --folds-json=scripts/ravdess_5cv_val.json \
  --uar-floor=0.56 \
  --uar-warmup=6 \
  --early-metric=acc
```

---

## Mamba Example

### 1. Feature Extraction ( with efficientnet_b0 BACKBONE )
```bash
sed -i 's/\r$//' scripts/run_extract_EN.sbatch
sbatch scripts/run_extract_EN.sbatch
tail -f /home/brel002/emotion_detection/outputs/logs/extract_<BID>.log
```

### 2. Check Extracted Features
Features are saved at:
```
/data/brel002/emotion_storage/outputs/features/<BACKBONE>
```

### 3. Run Ablation Experiments
- It runs over all 9 backbones' extracted features. Comment out backbones you do not want.
  
```bash
sed -i 's/\r$//' scripts/ablate_mamba_all.sh
sbatch scripts/ablate_mamba_all.sh
tail -n 200 -f /data/brel002/emotion_storage/logs/output_<QID>.txt
tail -n 200 -f /data/brel002/emotion_storage/logs/error_<QID>.txt
```

### 4. Run Fine-Tuning Experiments
```bash
sed -i 's/\r$//' scripts/ablate_mamba_fine_tuning.sh
sbatch scripts/ablate_mamba_fine_tuning.sh
tail -n 200 -f /data/brel002/emotion_storage/logs/output_<QID>.txt
tail -n 200 -f /data/brel002/emotion_storage/logs/error_<QID>.txt
```

### 5. Check Results
Results are saved at:
```
/data/brel002/emotion_storage/outputs/experiments/<BACKBONE>
```
- `mamba_results.csv`: Results from all phases.
- `mamba_CV_results.csv`: Aggregated results from the final test phase.

### 6. Check Best Mamba Result
Check the sorted results:
```
/data/brel002/emotion_storage/outputs/experiments/efficientnet_b0/mamba_CV_results_T32.csv
```
Sort by `retrain_test_agg_acc_mean` (descending).

### 7. Best Mamba model: checkpoints on each fold
files are uploaded to: storage/models/efficientnet_b0

on ML cluster:

/data/brel002/emotion_storage/models/efficientnet_b0/mamba_f20251026-110705_1092033_387969945d20.pt
/data/brel002/emotion_storage/models/efficientnet_b0/mamba_f20251026-110737_1092033_a2596c6b35a7.pt
/data/brel002/emotion_storage/models/efficientnet_b0/mamba_f20251026-110813_1092033_79980189c172.pt
/data/brel002/emotion_storage/models/efficientnet_b0/mamba_f20251026-110850_1092033_e1b68eedbd95.pt
/data/brel002/emotion_storage/models/efficientnet_b0/mamba_f20251026-110913_1092033_6e82ef52dc61.pt

### 8. Replicate Best Mamba Model
```bash
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV \
  --backbone=efficientnet_b0 \
  --d-model=256 \
  --layers=2 \
  --dropout=0.25 \
  --pool=attn \
  --label-smoothing=0.05 \
  --tin=off \
  --lr=0.001 \
  --weight-decay=0.001 \
  --epochs=40 \
  --patience=10 \
  --batch-size=32 \
  --folds-json=scripts/ravdess_5cv_val_test.json \
  --seed=42 \
  --early-metric=acc \
  --uar-floor=0.56 \
  --uar-warmup=6 \
  --ablation-phase=5 \
  --retrain-trainval
```

---

## Notes
- Replace `<BID>` and `<QID>` with actual job IDs.
- Ensure paths and filenames are updated according to your environment.
- To queue feature extracts from all backbones, run

```bash
sed -i 's/\r$//' scripts/run_extract_EN.sbatch
sbatch scripts/run_extract_EN.sbatch
sed -i 's/\r$//' scripts/run_extract_EN_B1.sbatch
sbatch scripts/run_extract_EN_B1.sbatch
sed -i 's/\r$//' scripts/run_extract_EN_V2.sbatch
sbatch scripts/run_extract_EN_V2.sbatch
sed -i 's/\r$//' scripts/run_extract_MNV2.sbatch
sbatch scripts/run_extract_MNV2.sbatch
sed -i 's/\r$//' scripts/run_extract_MNV3_large.sbatch
sbatch scripts/run_extract_MNV3_large.sbatch
sed -i 's/\r$//' scripts/run_extract_MNV3_small.sbatch
sbatch scripts/run_extract_MNV3_small.sbatch
sed -i 's/\r$//' scripts/run_extract_MNV4.sbatch 
sbatch scripts/run_extract_MNV4.sbatch
sed -i 's/\r$//' scripts/run_extract_MViT.sbatch
sbatch scripts/run_extract_MViT.sbatc
sed -i 's/\r$//' scripts/run_extract_DeiT_Tiny.sbatch
sbatch scripts/run_extract_DeiT_Tiny.sbatch
```
(make sure that all .sbatch files run the same extractor version: have the same line uncommented:
MOD="scripts.extract_fuse_features_generic_timm_V4_32VideoTSTEPS"
OR
MOD="scripts.extract_fuse_features_generic_timm_V4_24VideoTSTEPS"
)

