#!/bin/bash
#SBATCH --job-name=emotion_test
#SBATCH --output=/data/brel002/emotion_storage/logs/output_%j.txt  # %j = job ID
#SBATCH --error=/data/brel002/emotion_storage/logs/error_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00  # Optional: max runtime
#SBATCH --mem=32G  # Optional: memory limit

# Change to working directory
cd /home/brel002/emotion_detection/

# Initialize and activate conda
# eval "$(conda shell.bash hook)"  # Modern method (preferred)
# conda activate emotion_detection

# Verify environment (optional, for debugging)
echo "Python: $(which python)"
# echo "Conda env: $CONDA_DEFAULT_ENV"


# Run the ablation scripts: Mamba model

# /data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=attn --label-smoothing=0.02 --tin=off --lr=0.001 --weight-decay=0.0005 --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=5 --retrain-trainval
# /data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=attn --label-smoothing=0.02 --tin=off --lr=0.001 --weight-decay=0.001   --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=5 --retrain-trainval
# /data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=attn --label-smoothing=0.05 --tin=off --lr=0.001 --weight-decay=0.0005 --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=5 --retrain-trainval
# /data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=attn --label-smoothing=0.05 --tin=off --lr=0.001 --weight-decay=0.001   --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=5 --retrain-trainval

# /data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=topk --label-smoothing=0.02 --tin=off --lr=0.001 --weight-decay=0.0005 --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=5 --retrain-trainval
# /data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=topk --label-smoothing=0.02 --tin=off --lr=0.001 --weight-decay=0.001   --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=5 --retrain-trainval
# /data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=topk --label-smoothing=0.05 --tin=off --lr=0.001 --weight-decay=0.0005 --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=5 --retrain-trainval
# /data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=topk --label-smoothing=0.05 --tin=off --lr=0.001 --weight-decay=0.001   --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=5 --retrain-trainval

# /data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=attn --label-smoothing=0.02 --tin=off --use-moddrop --lr=0.001 --weight-decay=0.001 --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=5 --retrain-trainval
# /data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=attn --label-smoothing=0.05 --tin=off --use-moddrop --lr=0.001 --weight-decay=0.001 --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=5 --retrain-trainval
# /data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=topk --label-smoothing=0.02 --tin=off --use-moddrop --lr=0.001 --weight-decay=0.001 --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=5 --retrain-trainval
# /data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=topk --label-smoothing=0.05 --tin=off --use-moddrop --lr=0.001 --weight-decay=0.001 --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=5 --retrain-trainval

# 1) Same config, but pick checkpoints by UAR (may lift test UAR)
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=attn --label-smoothing=0.05 --tin=off --lr=0.001 --weight-decay=0.001 --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=uar --uar-floor=0.56 --uar-warmup=6 --ablation-phase=6 --retrain-trainval
# 2) Switch to Top-K pooling (same selection metric = acc)
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=topk --label-smoothing=0.05 --tin=off --lr=0.001 --weight-decay=0.001 --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=6 --retrain-trainval
# 3) Top-K + select-by-UAR (sometimes best for balanced recall)
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=topk --label-smoothing=0.05 --tin=off --lr=0.001 --weight-decay=0.001 --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=uar --uar-floor=0.56 --uar-warmup=6 --ablation-phase=6 --retrain-trainval
# 4) Slightly lower label smoothing (0.03)
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=attn --label-smoothing=0.03 --tin=off --lr=0.001 --weight-decay=0.001 --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=6 --retrain-trainval
# 5) Slightly lower weight decay (7.5e-4)
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.train_mamba_CV --backbone=efficientnet_b0 --d-model=256 --layers=2 --dropout=0.25 --pool=attn --label-smoothing=0.05 --tin=off --lr=0.001 --weight-decay=0.00075 --epochs=40 --patience=10 --batch-size=32 --folds-json=scripts/ravdess_5cv_val_test.json --seed=42 --early-metric=acc --uar-floor=0.56 --uar-warmup=6 --ablation-phase=6 --retrain-trainval






# Note: to run only Mamba ablation for a specific backbone, uncomment the relevant lines above and comment out the rest.