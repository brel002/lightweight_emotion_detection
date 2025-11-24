# scripts/extract_fuse_features_generic_timm.py
# Hybrid patch: keep the new TIMM plumbing & argparse, but
#   * revert audio pipeline to baseline-safe path (uint8 PCEN/Δ/ΔΔ + plain Resize -> Normalize)
#   * remove aspect-preserving pad for audio (avoid big constant regions)
#   * default VIDEO_T_STEPS back to 32 (AUDIO 64, FUSE 64)
#   * expose --audio-steps / --video-steps / --fuse-steps flags
#   * write rich meta.json (emb_dim_model from dummy forward; emb_dim from output; per-stream dims)

import os
import cv2
import json
import torch
import librosa
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from PIL import Image
import argparse

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms.v2 as Tv2
from moviepy.editor import VideoFileClip

# Project paths & defaults
from scripts.config import (DATA_DIR, OUTPUTS_DIR, BACKBONE_NAME)

# ------------------------------
# Defaults (can be overridden by CLI)
# ------------------------------
TARGET_SR_DEFAULT = 16000
AUDIO_WIN_S_DEFAULT = 3.0
AUDIO_N_MELS_DEFAULT = 64
AUDIO_T_STEPS_DEFAULT = 64
VIDEO_T_STEPS_DEFAULT = 24   
FUSE_T_STEPS_DEFAULT  = 64
FMIN_DEFAULT, FMAX_DEFAULT = 50.0, 7600.0
TOP_DB_DEFAULT = 25.0

AUDIO_CACHE = OUTPUTS_DIR / "audio_cache_wav"
AUDIO_CACHE.mkdir(parents=True, exist_ok=True)

# --- preferred audio source: official 03-*.wav in same Actor_* folder (or separate root) ---
# AUDIO_ONLY_ROOT = os.environ.get("RAVDESS_WAV_ROOT", "")  # optional, leave empty to use same dir as mp4

def find_official_wav_for(mp4_path: Path) -> Path | None:
    stem = mp4_path.stem  # e.g., 01-01-04-01-01-01-01
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    
    # build the name of corresponding .wav file, from the name of current .mp4 file
    parts[0] = "03"                             # RAVDESS modality: 01=AV, 02=V-only, 03=A-only
    wav_name = "-".join(parts) + ".wav"

    # get .wav from the same folder as the mp4 (Actor_XX)
    cand1 = mp4_path.with_suffix("").with_name(wav_name)
    if cand1.exists():
        return cand1

    # 2) optional separate audio-only root, preserving Actor_XX subdir
    # if AUDIO_ONLY_ROOT:
    #     cand2 = Path(AUDIO_ONLY_ROOT) / mp4_path.parent.name / wav_name
    #     if cand2.exists():
    #         return cand2

    return None


# ------------------------------
# Backbone factory
# ------------------------------

def make_backbone(name: str, device: torch.device):
    """Create pooled-embedding backbone and transforms for VIDEO.
       Audio uses a custom transform pipeline coordinated below.
    """
    model = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
    model.eval().to(device)

    # Source-of-truth embedding size via dummy forward
    in_h, in_w = (model.default_cfg or {}).get("input_size", (3, 224, 224))[-2:]
    with torch.no_grad():
        y = model(torch.zeros(1, 3, in_h, in_w, device=device))
    emb_dim = int(y.shape[-1])

    # Deterministic eval transform for VIDEO frames
    cfg = resolve_data_config({}, model=model)
    video_tf = create_transform(**cfg, is_training=False)   # Resize -> CenterCrop -> ToTensor -> Normalize

    # AUDIO transform is defined inside extract_audio_features (uses same cfg mean/std & plain Resize)
    return model, emb_dim, cfg, video_tf

# ------------------------------
# Utils
# ------------------------------

def time_resize(feat: np.ndarray, T_out: int) -> np.ndarray:
    if feat.shape[0] == T_out:
        return feat.astype(np.float32)
    T_in = feat.shape[0]
    idx = np.linspace(0, T_in - 1, num=T_out)
    idx0 = np.floor(idx).astype(int)
    idx1 = np.clip(idx0 + 1, 0, T_in - 1)
    a = idx - idx0
    return ((1 - a)[:, None] * feat[idx0] + a[:, None] * feat[idx1]).astype(np.float32)


def extract_audio_from_video(video_path: Path, audio_path: Path):
    if audio_path.exists():
        return
    clip = VideoFileClip(str(video_path))
    clip.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
    clip.close()


def voiced_center_crop(y: np.ndarray, sr: int, target_len_s: float, top_db: float):
    
    # get voiced regions (speech-like energy) in the waveform: finds regions where the signal is above (ref_db - top_db).
    intervals = librosa.effects.split(y, top_db=top_db)
    
    # get targeted segment length in samples
    target_len = int(target_len_s * sr)

    #  if nothing is voiced (e.g., very quiet or silence), take the first target_len samples,
    #  then pad/trim to exactly target_len.
    if len(intervals) == 0:
        return librosa.util.fix_length(y[:target_len], size=target_len)

     # Pick the longest voiced interval.
    lens = intervals[:, 1] - intervals[:, 0]        # lengths of each interval
    i = int(np.argmax(lens))                        # index of the longest
    vstart, vend = intervals[i]                     # start/end samples of that interval
    
    # get midpoint of the longest voiced interval
    vcenter = (vstart + vend) // 2

    # center a window of target_len around vcenter.
    half = target_len // 2
    start = max(0, vcenter - half)
    end = start + target_len

    # if the window runs past the end of y, shift it left so it fits.
    if end > len(y):
        start = max(0, len(y) - target_len)
        end = len(y)
    seg = y[start:end]

    # extract and ensure *exact* length via padding/truncation.
    return librosa.util.fix_length(seg, size=target_len)


def _to_uint8_img(x: np.ndarray) -> np.ndarray:
    # Baseline-safe per-image scaling to [0,255]
    x = x.astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - np.median(x)) / (np.std(x) + 1e-6)
    mn, mx = x.min(), x.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - mn) / (mx - mn)
    return (x * 255.0).clip(0, 255).astype(np.uint8)

# ------------------------------
# Audio features (baseline-style image pipeline)
# ------------------------------

@torch.no_grad()
def extract_audio_features(
    audio_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    cfg: dict,
    *,
    target_sr: int,
    audio_win_s: float,
    audio_n_mels: int,
    audio_t_steps: int,
    fmin: float,
    fmax: float,
    top_db: float,
    use_preemph: bool = True,
):
    # 'robust' load + voiced crop
    try:
        y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True, res_type="soxr_hq")
    except Exception:
        y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True, res_type="kaiser_best")
    
    y = y.astype(np.float32, copy=False)

    # first-order pre-emphasis filter with coefficient 0.97
    # gently boosts high frequencies and sharp transients (consonants/onsets), which often helps time-frequency features.
    if use_preemph and y.size >= 2:
        y = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)
    
    # center on longest voiced region (then pad/trim to exact length)
    y = voiced_center_crop(y, sr, target_len_s=audio_win_s, top_db=top_db)
    if y.size < int(0.2 * sr):
        y = np.pad(y, (0, int(0.2 * sr) - y.size + 1), mode="constant")

    # Mel -> PCEN (+ Δ/ΔΔ)
    n_fft = max(256, int(round(0.025 * sr)))
    hop   = max(80,  int(round(0.010 * sr)))
    fmax_used = min(fmax, sr // 2)

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=audio_n_mels,
        fmin=fmin, fmax=fmax_used, power=1.0
    ).astype(np.float32, copy=False)

    pcen = librosa.pcen(S, sr=sr, hop_length=hop, eps=1e-6).astype(np.float32, copy=False)
    d1   = librosa.feature.delta(pcen, order=1).astype(np.float32, copy=False)
    d2   = librosa.feature.delta(pcen, order=2).astype(np.float32, copy=False)

    pcen = np.nan_to_num(pcen, nan=0.0, posinf=1e6, neginf=-1e6)
    d1   = np.nan_to_num(d1,   nan=0.0, posinf=1e6, neginf=-1e6)
    d2   = np.nan_to_num(d2,   nan=0.0, posinf=1e6, neginf=-1e6)

    img3 = np.stack([_to_uint8_img(pcen), _to_uint8_img(d1), _to_uint8_img(d2)], axis=-1)  # H×W×3 (uint8)

    # slice along time and convert to PIL
    T = img3.shape[1]
    bounds = np.linspace(0, T, num=audio_t_steps + 1, dtype=int)
    segs = []
    for t in range(audio_t_steps):
        l, r = bounds[t], bounds[t + 1]
        seg = img3[:, l:r, :]
        if seg.shape[1] == 0:
            seg = img3[:, max(0, T - 1):T, :]
        segs.append(Image.fromarray(seg))

    # transform & embed (plain Resize, no pad; then Normalize per cfg)
    H, W = cfg['input_size'][1], cfg['input_size'][2]
    audio_tf = Tv2.Compose([
        Tv2.ToImage(),
        Tv2.ToDtype(torch.float32, scale=True),
        Tv2.Resize((H, W), antialias=True),
        Tv2.Normalize(mean=cfg['mean'], std=cfg['std']),
    ])

    # use pretrained backbone to extract and embed the features
    X = torch.stack([audio_tf(s) for s in segs], dim=0).to(device)  # [T,3,H,W]
    feats = model(X)
    if feats.ndim != 2:
        raise RuntimeError(f"Expected [T,D], got {tuple(feats.shape)}")
    emb = feats.detach().cpu().numpy().astype(np.float32)
    return np.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)

# ------------------------------
# Video features
# ------------------------------

def extract_video_features(video_path: Path, model, video_transform, device, video_t_steps: int) -> np.ndarray:
    
    # get all frames
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return None

    # sample video_t_steps frames
    idxs = np.linspace(0, len(frames) - 1, num=video_t_steps, dtype=int)
    pils = []

    # convert to RGB
    for i in idxs:
        frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        pils.append(Image.fromarray(frame_rgb))

    # use pretrained backbone to extract and embed the features
    with torch.inference_mode():
        X = torch.stack([video_transform(p) for p in pils], dim=0).to(device)  # [T,3,H,W]
        v_emb = model(X).detach().cpu().numpy().astype(np.float32)            # [T, emb_dim]
    return np.nan_to_num(v_emb, nan=0.0, posinf=1e6, neginf=-1e6)

# ------------------------------
# Fusion
# ------------------------------

def fuse_concat(video_feat: np.ndarray, audio_feat: np.ndarray, T_out: int) -> np.ndarray:
    v = time_resize(video_feat, T_out)
    a = time_resize(audio_feat, T_out)
    return np.concatenate([v, a], axis=1).astype(np.float32)

# ------------------------------
# Main
# ------------------------------

def pick_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Extract Audio, Video and Fused features from RAVDESS (hybrid patch).")

    parser.add_argument("--backbone", type=str,
                        default=None,
                        help="TIMM backbone (default: from scripts.config BACKBONE_NAME)")

    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                        help="cpu=cpu only, cuda=force CUDA, auto=CUDA if available else CPU")

    # Steps & audio options
    parser.add_argument("--audio-steps", type=int, default=AUDIO_T_STEPS_DEFAULT)
    parser.add_argument("--video-steps", type=int, default=VIDEO_T_STEPS_DEFAULT)
    parser.add_argument("--fuse-steps",  type=int, default=FUSE_T_STEPS_DEFAULT)
    parser.add_argument("--target-sr", type=int, default=TARGET_SR_DEFAULT)
    parser.add_argument("--audio-win-s", type=float, default=AUDIO_WIN_S_DEFAULT)
    parser.add_argument("--audio-n-mels", type=int, default=AUDIO_N_MELS_DEFAULT)
    parser.add_argument("--fmin", type=float, default=FMIN_DEFAULT)
    parser.add_argument("--fmax", type=float, default=FMAX_DEFAULT)
    parser.add_argument("--top-db", type=float, default=TOP_DB_DEFAULT)
    parser.add_argument("--no-preemph", action="store_true", help="Disable pre-emphasis in audio pipeline")

    parser.add_argument("--sample-limit", type=int, default=None, help="For quick tests; limit number of clips")

    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"[INFO] Using device: {device}")

    backbone = args.backbone or BACKBONE_NAME
    model, emb_dim_model, cfg, video_transform = make_backbone(backbone, device)
    print(f"[INFO] Backbone={backbone} | emb_dim_model(dummy)={emb_dim_model}")

    # IO paths for this backbone
    feature_dir = OUTPUTS_DIR / "features" / backbone
    feature_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = feature_dir / "ravdess_labels.csv"

    # Enumerate files (actor '01' multimodal mp4s only, per your convention)
    mp4_files = glob(os.path.join(str(DATA_DIR), "**", "*.mp4"), recursive=True)
    mp4_files = [p for p in mp4_files if os.path.basename(p).split('-')[0] == '01']
    mp4_files.sort()
    if args.sample_limit:
        mp4_files = mp4_files[:args.sample_limit]
    print(f"Processing {len(mp4_files)} multimodal mp4 files.")

    ravdess_label_map = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
    }

    audio_all, video_all, fused_c_all = [], [], []
    labels, files = [], []

    for path in mp4_files:
        filename = os.path.basename(path)
        video_path = Path(path)
        print(f"\nProcessing: {filename}")
        emotion_code = filename.split('-')[2]
        emotion_label = ravdess_label_map.get(emotion_code, "unknown")
        try:
            
            # get .wav file, if located in the same folder as .mp4
            official_wav = find_official_wav_for(video_path)

            if official_wav is not None:
                wav_path = official_wav
                print(f"  using official WAV: {wav_path.name}")
            else:
                # fallback: extract audio from the AV mp4 into cache (exactly what you do today)
                wav_cache = AUDIO_CACHE / (filename.replace('.mp4', '.wav'))
                if not wav_cache.exists():
                    extract_audio_from_video(video_path, wav_cache)
                wav_path = wav_cache

            # --- Audio ---
            audio_feat = extract_audio_features(
                wav_path, model, device, cfg,
                target_sr=args.target_sr,
                audio_win_s=args.audio_win_s,
                audio_n_mels=args.audio_n_mels,
                audio_t_steps=args.audio_steps,
                fmin=args.fmin, fmax=args.fmax, top_db=args.top_db,
                use_preemph=(not args.no_preemph),
            )

            
            # --- Video ---
            video_feat = extract_video_features(video_path, model, video_transform, device, args.video_steps)
            if video_feat is None:
                print("  Skipping (no frames).")
                continue

            # --- Fuse ---
            fused_c = fuse_concat(video_feat, audio_feat, args.fuse_steps)

            print(f"  Audio: {audio_feat.shape} | Video: {video_feat.shape} | Concat: {fused_c.shape}")

            audio_all.append(audio_feat.astype(np.float32))
            video_all.append(video_feat.astype(np.float32))
            fused_c_all.append(fused_c.astype(np.float32))

            labels.append((filename, emotion_label))
            files.append(filename)

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    if not audio_all:
        print("No samples processed.")
        return

    audio_arr = np.stack(audio_all, axis=0)
    video_arr = np.stack(video_all, axis=0)
    fused_c_arr = np.stack(fused_c_all, axis=0)

    # Save arrays
    np.save(feature_dir / "audio_features.npy", audio_arr)
    np.save(feature_dir / "video_features.npy", video_arr)
    np.save(feature_dir / "fused_features_concat.npy", fused_c_arr)

    # Labels & index
    df = pd.DataFrame(labels, columns=["file", "label"])
    df.to_csv(labels_csv, index=False)
    np.save(feature_dir / "files.npy", np.array(files))

    print("\n✅ Packed arrays saved.")
    print(f"  audio_features.npy              shape={audio_arr.shape}")
    print(f"  video_features.npy              shape={video_arr.shape}")
    print(f"  fused_features_concat.npy       shape={fused_c_arr.shape}")
    print(f"  labels.csv, files.npy written to {feature_dir}")

    # Meta (emb_dim is per-stream output dim from this backbone)
    meta = {
        "backbone": backbone,
        "emb_dim_model": int(emb_dim_model),
        "emb_dim": int(audio_arr.shape[-1]),
        "audio_steps": int(args.audio_steps),
        "video_steps": int(args.video_steps),
        "fuse_steps": int(args.fuse_steps),
        "audio_feature_dim": int(audio_arr.shape[-1]),
        "video_feature_dim": int(video_arr.shape[-1]),
        "concat_feature_dim": int(fused_c_arr.shape[-1]),
    }
    with open(feature_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")
    main()
