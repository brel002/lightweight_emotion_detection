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

# 1) backbone: efficientnet_b0
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph0 --backbone=efficientnet_b0
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph1 --backbone=efficientnet_b0 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph2 --backbone=efficientnet_b0 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph3 --backbone=efficientnet_b0 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_test --backbone=efficientnet_b0 --metric=val_acc --top-k=10 --retrain-trainval

# 2) backbone: efficientnet_b1
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph0 --backbone=efficientnet_b1
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph1 --backbone=efficientnet_b1 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph2 --backbone=efficientnet_b1 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph3 --backbone=efficientnet_b1 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_test --backbone=efficientnet_b1 --metric=val_acc --top-k=10 --retrain-trainval

# 3) backbone: tf_efficientnetv2_s.in21k_ft_in1k
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph0 --backbone=tf_efficientnetv2_s.in21k_ft_in1k
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph1 --backbone=tf_efficientnetv2_s.in21k_ft_in1k --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph2 --backbone=tf_efficientnetv2_s.in21k_ft_in1k --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph3 --backbone=tf_efficientnetv2_s.in21k_ft_in1k --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_test --backbone=tf_efficientnetv2_s.in21k_ft_in1k --metric=val_acc --top-k=10 --retrain-trainval

# 4) backbone: deit_tiny_patch16_224
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph0 --backbone=deit_tiny_patch16_224
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph1 --backbone=deit_tiny_patch16_224 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph2 --backbone=deit_tiny_patch16_224 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph3 --backbone=deit_tiny_patch16_224 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_test --backbone=deit_tiny_patch16_224 --metric=val_acc --top-k=10 --retrain-trainval


# 5) backbone: mobilenetv2_100
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph0 --backbone=mobilenetv2_100
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph1 --backbone=mobilenetv2_100 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph2 --backbone=mobilenetv2_100 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph3 --backbone=mobilenetv2_100 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_test --backbone=mobilenetv2_100 --metric=val_acc --top-k=10 --retrain-trainval

# 6) backbone: mobilenetv3_large_100
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph0 --backbone=mobilenetv3_large_100
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph1 --backbone=mobilenetv3_large_100 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph2 --backbone=mobilenetv3_large_100 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph3 --backbone=mobilenetv3_large_100 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_test --backbone=mobilenetv3_large_100 --metric=val_acc --top-k=10 --retrain-trainval

# 7) backbone: mobilenetv3_small_100
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph0 --backbone=mobilenetv3_small_100
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph1 --backbone=mobilenetv3_small_100 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph2 --backbone=mobilenetv3_small_100 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph3 --backbone=mobilenetv3_small_100 --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_test --backbone=mobilenetv3_small_100 --metric=val_acc --top-k=10 --retrain-trainval

# 8) backbone: mobilenetv4_conv_small_050.e3000_r224_in1k
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph0 --backbone=mobilenetv4_conv_small_050.e3000_r224_in1k
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph1 --backbone=mobilenetv4_conv_small_050.e3000_r224_in1k --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph2 --backbone=mobilenetv4_conv_small_050.e3000_r224_in1k --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph3 --backbone=mobilenetv4_conv_small_050.e3000_r224_in1k --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_test --backbone=mobilenetv4_conv_small_050.e3000_r224_in1k --metric=val_acc --top-k=10 --retrain-trainval


# 9) backbone: mobilevit_s.cvnets_in1k
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph0 --backbone=mobilevit_s.cvnets_in1k
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph1 --backbone=mobilevit_s.cvnets_in1k --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph2 --backbone=mobilevit_s.cvnets_in1k --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_ph3 --backbone=mobilevit_s.cvnets_in1k --metric=val_acc --top-k=10
/data/brel002/conda/envs/emotion_detection/bin/python -u -m scripts.run_mamba_ablate_test --backbone=mobilevit_s.cvnets_in1k --metric=val_acc --top-k=10 --retrain-trainval

# Note: to run only Mamba ablation for a specific backbone, uncomment the relevant lines above and comment out the rest.