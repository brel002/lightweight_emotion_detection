#!/usr/bin/env python3
import argparse, csv, math, os, subprocess, sys, time
from pathlib import Path
from scripts.config import OUTPUTS_DIR

POOLS = ["attn","mean","topk"]  # topk uses model defaults (frac=0.35, min_k=6)

FIXED = [
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
    "--ablation-phase=1",
]

def f(x):
    try: return float(x)
    except: return float("nan")

def load_topk(csv_path: Path, top_k: int, metric_col: str):
    rows=[]
    with csv_path.open(newline="",encoding="utf-8") as fcsv:
        # iterate through previous results and get configs
        rdr=csv.DictReader(fcsv)
        for row_index, r in enumerate(rdr):
            dm=f(r.get("d_model"))
            ly=f(r.get("layers"))
            dp=f(r.get("dropout"))
            m =f(r.get(metric_col or "val_acc"))

            if any(math.isnan(z) for z in (dm,ly,dp,m)): 
                continue
            
            rows.append({"d_model":int(dm),
                         "layers":int(ly),
                         "dropout":float(dp),
                         "metric":float(m),
                         "_ridx": row_index}),
            
    # placehlolder for best configs
    best={}

    # select top-k unique configurations
    for r in rows:
        # get configuration
        k=(r["d_model"], r["layers"], round(r["dropout"],6))

        # hold unique configurations
        if (k not in best) or (                     # we haven’t seen this config
            r["metric"]> best[k]["metric"]) or (    # same config, but this run (row) scored higher
            (r["metric"]== best[k]["metric"] and r["_ridx"] < best[k]["_ridx"])  # same metric, pick up row from earlier run
        ): 
            best[k]=r
    
    # get unique configs sorted by metric
    uniq=list(best.values()); 
    uniq.sort(key=lambda x:x["metric"], reverse=True)
    return uniq[:top_k]

def main():
    ap=argparse.ArgumentParser("Mamba Phase-1")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--top-k", type=int, default=10)        # get top-k configs from phase-0
    ap.add_argument("--metric", default="val_acc")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff", type=int, default=8)
    ap.add_argument("--cpus", type=int, default=0)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--csv", default=None)
    args=ap.parse_args()

    EXPER_DIR = OUTPUTS_DIR / "experiments" / args.backbone
    csv_path  = Path(args.csv or (EXPER_DIR/"mamba_results.csv"))
    if not csv_path.exists():
        print(f"[ERR] CSV not found: {csv_path}"); sys.exit(2)

    cfgs = load_topk(csv_path, args.top_k, args.metric)
    
    if not cfgs: print("[ERR] no configs from Phase-0"); sys.exit(2)
    combos=[]
    for base in cfgs:
        for pool in POOLS:
            combos.append((base, pool))

    print(f"[grid] Phase-1: {len(combos)} runs from top-{len(cfgs)} Phase-0")
    if args.cpus>0:
        for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            os.environ[k]=str(args.cpus)
        os.environ["MKL_DYNAMIC"]="FALSE"; os.environ["OMP_DYNAMIC"]="FALSE"
        os.environ["OMP_PROC_BIND"]="TRUE"; os.environ["OMP_PLACES"]="cores"

    ok=0; 
    t0=time.time()

    for i,(base,pool) in enumerate(combos,1):
        dm = base["d_model"]
        ly = base["layers"]
        dp = base["dropout"]
        tag=f"d_model={dm} layers={ly} dropout={dp} pool={pool} phase=1"
        print(f"\n=== [{i}/{len(combos)}] {tag} ===")
        
        cmd=[args.python,"-u","-m","scripts.train_mamba_CV",
             f"--backbone={args.backbone}",
             f"--d-model={dm}", 
             f"--layers={ly}", 
             f"--dropout={dp}",
             f"--pool={pool}", 
             *FIXED]
        
        for a in range(1,args.retries+1):
            print(f"[{time.strftime('%H:%M:%S')}] {tag} (try {a}/{args.retries})")
            
            # run the experiment
            rc=subprocess.call(cmd)
            if rc==0: 
                ok+=1
                break
            if a<args.retries: 
                print(f"[WARN] rc={rc}; sleep {args.backoff}s"); 
                time.sleep(args.backoff)
    
    mins=(time.time()-t0)/60; 
    print(f"\n[done] {ok}/{len(combos)} succeeded in {mins:.1f} min")

if __name__=="__main__":
    main()
