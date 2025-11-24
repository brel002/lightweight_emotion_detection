#!/usr/bin/env python3
import argparse, itertools, os, subprocess, sys, time


# ----- Phase-0 grid -----
HIDDEN_LIST = [128, 256, 384]
LAYERS_LIST = [1, 2, 3]
PDROP_LIST  = [0.10, 0.20, 0.25, ] 
              # 0.30, 
              #0.35, 0.40
              

# 4 switch presets:
SWITCHES = [
    ["--proj-dim=0", "--use-gate", "--fuse-mode=concat", "--fuse-dim=192"],
    ["--proj-dim=0", "--use-gate", "--fuse-mode=replace", "--fuse-dim=192"],
    ["--proj-dim=192"],                                                         # proj=192 + no gate
    ["--proj-dim=0"],                                                           # no proj + no gate
]




# SWITCHES = [
#     ["--proj-dim=256"],  # projection only
#     ["--use-gate", "--fuse-mode-replace", "--fuse-dim=192"],  # gate replace
# ]

# Fixed flags
FIXED_ARGS = [
    "--mod-drop=0.0",
    "--freq-mask-ratio=0.0",
    "--time-mask-ratio=0.0",
    "--num_t=0",
    "--num_f=0",
    "--lr=7e-4",
    "--weight-decay=1e-3",
    "--label-smoothing=0.015",
    "--seed=42",
    "--pool-type=attn",
    "--top-k=5",  # not used in this phase, as pool_type!="topk"
    "--margin=0",
    "--fold-idx=0",
    "--folds-json=scripts/ravdess_5cv_val.json",     # train/val splits
    "--skip-test",
    "--uar-floor=0.56",
    "--uar-warmup=6",
    "--early-metric=acc",
]

def run_one(py, hidden, layers, pdrop, switches, retries, backoff, backbone):
    cmd = [
        py, "-u", "-m", "scripts.train_gru_CV",
        f"--hidden={hidden}",
        f"--num-layers={layers}",
        f"--p-drop={pdrop}",  
        "--ablation-phase=0",   
        f"--backbone={backbone}",   
        *FIXED_ARGS,
        *switches, 
    ]
    for attempt in range(1, retries + 1):
        tag = f"h={hidden} l={layers} p_drop={pdrop} {'gate' if any(s.startswith('--use-gate') for s in switches) else 'proj'}"
        tag = f"h={hidden} l={layers} p_drop={pdrop} {'gate' if '--use-gate' in switches else [s for s in switches if s.startswith('--proj-dim=')][0].replace('--proj-dim=','proj')}"

        print(f"[{time.strftime('%H:%M:%S')}] {tag} (try {attempt}/{retries})")
        rc = subprocess.call(cmd)
        if rc == 0:
            print(f"[OK] {tag}")
            return True
        print(f"[WARN] rc={rc} for {tag}; retrying in {backoff}s…")
        time.sleep(backoff)

    print(f"[FAIL] {tag} after {retries} attempts.")
    return False

def main():
    ap = argparse.ArgumentParser("Phase-0 sequential ablation")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    ap.add_argument("--retries", type=int, default=2, help="Retries per combo if it exits non-zero")
    ap.add_argument("--backoff", type=int, default=5, help="Seconds to sleep between retries")
    ap.add_argument("--cpus", type=int, default=0, help="If >0, set OMP/MKL/OPENBLAS/NUMEXPR threads to this value")
    ap.add_argument("--backbone", type=str, choices=["mobilenetv2_100",
                                                        "efficientnet_b0", 
                                                        "mobilenetv3_small_100", 
                                                        "mobilenetv3_large_100",
                                                        "mobilenetv4_conv_small_050.e3000_r224_in1k",
                                                        "mobilevit_s.cvnets_in1k",
                                                        "deit_tiny_patch16_224",
                                                        "tf_efficientnetv2_s.in21k_ft_in1k",
                                                        "efficientnet_b1"], default=None,
                                                        help="If omitted, uses BACKBONE_NAME from scripts.config")

    args = ap.parse_args()

    if args.cpus > 0:
        os.environ["OMP_NUM_THREADS"]      = str(args.cpus)
        os.environ["MKL_NUM_THREADS"]      = str(args.cpus)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.cpus)
        os.environ["NUMEXPR_NUM_THREADS"]  = str(args.cpus)
        os.environ["MKL_DYNAMIC"] = "FALSE"
        os.environ["OMP_DYNAMIC"] = "FALSE"
        os.environ["OMP_PROC_BIND"] = "TRUE"
        os.environ["OMP_PLACES"] = "cores"

    # Create grid of configs
    combos = list(itertools.product(HIDDEN_LIST, LAYERS_LIST, PDROP_LIST,SWITCHES))
    total = len(combos)
    print(f"[grid] Phase-0: {total} runs")

    ok = 0
    start = time.time()
    try:
        for i, (h, l, pd, sw) in enumerate(combos, 1):   # ← unpack 4 items
            # label for logs
            if "--use-gate" in sw:
                label = "gate+proj0"
            elif any(s == "--proj-dim=0" for s in sw):
                label = "proj0"         # no projection
            elif any(s.startswith("--proj-dim=") for s in sw):
                label = f"proj{[s.split('=')[1] for s in sw if s.startswith('--proj-dim=')][0]}"
            else:
                label = "default"

            print(f"\n=== [{i}/{total}] hidden={h} layers={l} p_drop={pd} {label} ===")

            # run the experiment
            if run_one(args.python, h, l, pd, sw, args.retries, args.backoff, args.backbone):  # ← pass sw
                ok += 1

    except KeyboardInterrupt:
        print("\n[abort] interrupted by user")

    dur = time.time() - start
    print(f"\n[done] {ok}/{total} succeeded in {dur/60:.1f} min")

if __name__ == "__main__":
    main()
