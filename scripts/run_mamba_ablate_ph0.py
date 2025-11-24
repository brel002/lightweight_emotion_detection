#!/usr/bin/env python3
import argparse, itertools, os, subprocess, sys, time

# ----- Phase-0 grid (tiny & effective) -----
DM_LIST   = [128, 192, 256]
LAY_LIST  = [2, 3]
DROP_LIST = [0.15, 0.25]

# Fixed flags for this phase
FIXED = [
    "--pool=attn",
    "--tin=off",
    "--label-smoothing=0.02",
    "--lr=1e-3",
    "--weight-decay=5e-4",
    "--epochs=40",
    "--patience=10",
    "--batch-size=32",
    "--seed=42",
    "--early-metric=acc",
    "--uar-floor=0.56",
    "--uar-warmup=6",
    "--skip-test",
    "--fold-idx=0",
    "--folds-json=scripts/ravdess_5cv_val.json",  # train/val splits
    "--ablation-phase=0",
]

def run_one(py, backbone, d_model, layers, dropout, retries, backoff):
    cmd = [py, "-u", "-m", "scripts.train_mamba_CV",
           f"--backbone={backbone}",
           f"--d-model={d_model}",
           f"--layers={layers}",
           f"--dropout={dropout}",
           *FIXED]
    tag = f"d={d_model} L={layers} p={dropout}"
    for a in range(1, retries+1):
        print(f"[{time.strftime('%H:%M:%S')}] {tag} (try {a}/{retries})")
        rc = subprocess.call(cmd)
        if rc == 0:
            print(f"[OK] {tag}"); return True
        print(f"[WARN] rc={rc}; sleeping {backoff}s…"); time.sleep(backoff)
    print(f"[FAIL] {tag}"); return False

def main():
    ap = argparse.ArgumentParser("Mamba Phase-0")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff", type=int, default=8)
    ap.add_argument("--cpus", type=int, default=0)
    ap.add_argument("--backbone", required=True)
    args = ap.parse_args()

    if args.cpus>0:
        for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            os.environ[k]=str(args.cpus)
        os.environ["MKL_DYNAMIC"]="FALSE"; os.environ["OMP_DYNAMIC"]="FALSE"
        os.environ["OMP_PROC_BIND"]="TRUE"; os.environ["OMP_PLACES"]="cores"

    # Create grid of configurations
    grid = list(itertools.product(DM_LIST, LAY_LIST, DROP_LIST))
    print(f"[grid] Phase-0: {len(grid)} runs")
    ok = 0; t0=time.time()
    try:
        for i,(dm,ly,dp) in enumerate(grid,1):
            print(f"\n=== [{i}/{len(grid)}] d={dm} L={ly} p={dp} ===")
            # run the experiment
            ok += run_one(args.python, args.backbone, dm, ly, dp, args.retries, args.backoff)
    except KeyboardInterrupt:
        print("\n[abort]")
    mins=(time.time()-t0)/60; print(f"\n[done] {ok}/{len(grid)} ok in {mins:.1f} min")

if __name__=="__main__":
    main()
