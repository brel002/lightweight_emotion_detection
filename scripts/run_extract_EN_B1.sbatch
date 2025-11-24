#!/usr/bin/env bash
#SBATCH --job-name=ravdess_extract_cpu
#SBATCH --chdir=/home/brel002/emotion_detection
#SBATCH --output=/home/brel002/emotion_detection/outputs/logs/extract_%j.log
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --requeue

set -Eeuo pipefail

# exact interpreter
PYTHON="/data/brel002/conda/envs/emotion_extract2/bin/python"

# backbone to run (change as needed or export via sbatch --export=BACKBONE=...)
#BACKBONE="mobilenetv3_large_100"
#BACKBONE="mobilenetv4_conv_small_050.e3000_r224_in1k"
#BACKBONE="mobilenetv3_small_100"

#BACKBONE="efficientnet_b0"

#BACKBONE="efficientnetv2_s"
BACKBONE="efficientnet_b1"
#BACKBONE="mobilenetv2_100"
#BACKBONE="mobilevit_s.cvnets_in1k"

#BACKBONE="deit_tiny_patch16_224"

# ensure extractor runs on CPU only
export CUDA_VISIBLE_DEVICES=""
export TORCH_DEVICE="cpu"

# storage & imports
export ED_STORAGE="/data/brel002/emotion_storage"
export PYTHONPATH="/home/brel002/emotion_detection:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# threading: match cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

export HF_HOME="$ED_STORAGE/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$ED_STORAGE/torch_hub"
export HF_HUB_DISABLE_SYMLINKS=1
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"


# don’t unset LD_LIBRARY_PATH; OpenCV/FFmpeg may need it for CPU decode
# unset LD_LIBRARY_PATH || true   # <- leave commented

mkdir -p outputs/logs

echo "[INFO] CWD:    $(pwd)"
echo "[INFO] Python: $PYTHON"
"$PYTHON" -V

# quick visibility check
$PYTHON - <<'PY'
from scripts.config import DATA_DIR
import glob, os
print("DATA_DIR =", DATA_DIR)
print("mp4 count =", len(glob.glob(os.path.join(str(DATA_DIR),'**','*.mp4'), recursive=True)))
PY


# run current extractor version
#MOD="scripts.extract_fuse_features_generic_timm_V4_24VideoTSTEPS"
MOD="scripts.extract_fuse_features_generic_timm_V4_32VideoTSTEPS"

# run as a module
srun --cpu-bind=cores --unbuffered "$PYTHON" -u -m "$MOD" --backbone="${BACKBONE}" --device=cpu
