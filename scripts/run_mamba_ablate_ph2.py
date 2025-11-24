#!/usr/bin/env python3
import argparse, csv, math, os, subprocess, sys, time
from pathlib import Path
from scripts.config import OUTPUTS_DIR


SMOOTH    = [0.00, 0.02, 0.05]
MODDROP   = [False, True]   # train-only; keep off at eval

FIXED_BASE = [
    "--tin=off",
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
    "--ablation-phase=2",
]

# fro each dropout, also try +/-0.05 within [0.10,0.35]
# def refine_drop(p):
#     cand = {round(x, 2) for x in (p, p-0.05, p+0.05) if 0.10 <= x <= 0.35}
#     return sorted(cand)


# check each dropout from top-k previous configs, do edge cases
def refine_drop(p):
    if abs(p-0.15) < 1e-6: return [0.10, 0.15, 0.20]
    if abs(p-0.25) < 1e-6: return [0.20, 0.25, 0.30]
    return [p]



def f(x):
    try: return float(x)
    except: return float("nan")

def load_topk(csv_path: Path, top_k: int, metric_col: str):
    rows=[]
    
    # load previous results from the file
    with csv_path.open(newline="",encoding="utf-8") as fcsv:
        # iterate through previous results and get configs
        rdr=csv.DictReader(fcsv)
        for row_index, r in enumerate(rdr):
            dm=f(r.get("d_model")) 
            ly=f(r.get("layers"))
            dp=f(r.get("dropout"))
            pool=(r.get("pool") or "attn").strip()
            m=f(r.get(metric_col or "val_acc"))
            
            if any(math.isnan(z) for z in (dm,ly,dp,m)): 
                continue
            
            rows.append({"d_model":int(dm),
                         "layers":int(ly),
                         "dropout":float(dp),
                         "pool":pool,
                         "metric":float(m),
                         "_ridx": row_index})
    
    # placehlolder for best configs            
    best={}
    
    # select top-k unique configurations
    for r in rows:
        # get configuration
        k=(r["d_model"], r["layers"], round(r["dropout"],6), r["pool"])

        # hold unique configurations
        if (k not in best) or (                     # we haven’t seen this config
            r["metric"]> best[k]["metric"]) or (    # same config, but this run (row) scored higher
            (r["metric"]== best[k]["metric"] and r["_ridx"] < best[k]["_ridx"])  # same metric, pick up row from earlier run
        ): 
            best[k]=r

    # get unique configs sorted by metric
    uniq=list(best.values())
    uniq.sort(key=lambda x:x["metric"], reverse=True)
    return uniq[:top_k]

def main():
    ap=argparse.ArgumentParser("Mamba Phase-2")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--top-k", type=int, default=10)
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

    bases = load_topk(csv_path, args.top_k, args.metric)
    if not bases: print("[ERR] no configs from previous phase"); sys.exit(2)

    combos=[]
    for base in bases:
        for dp in refine_drop(base["dropout"]):
            for ls in SMOOTH:
                for md in MODDROP:
                    combos.append((base, dp, ls, md))

    if args.cpus>0:
        for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            os.environ[k]=str(args.cpus)
        os.environ["MKL_DYNAMIC"]="FALSE"; os.environ["OMP_DYNAMIC"]="FALSE"
        os.environ["OMP_PROC_BIND"]="TRUE"; os.environ["OMP_PLACES"]="cores"

    print(f"[grid] Phase-2: {len(combos)} runs from top-{len(bases)}")
    ok=0
    t0=time.time()
    
    for i,(base,dp,ls,md) in enumerate(combos,1):
        
        dm = base["d_model"]
        ly = base["layers"]
        pool = base["pool"]

        tag=f"d_model={dm} layers={ly} pool={pool} dropout={dp} smoothing={ls} moddrop={int(md)} phase=2"
        print(f"\n=== [{i}/{len(combos)}] {tag} ===")

        cmd=[args.python,"-u","-m","scripts.train_mamba_CV",
             f"--backbone={args.backbone}",
             f"--d-model={dm}", 
             f"--layers={ly}",
             f"--pool={pool}",
             f"--dropout={dp}",
             f"--label-smoothing={ls}",
             "--use-moddrop" if md else "",
             *FIXED_BASE]
        cmd=[c for c in cmd if c!=""]
        for a in range(1,args.retries+1):
            print(f"[{time.strftime('%H:%M:%S')}] {tag} (try {a}/{args.retries})")
            rc=subprocess.call(cmd)
            if rc==0: ok+=1; break
            if a<args.retries: print(f"[WARN] rc={rc}; sleep {args.backoff}s"); time.sleep(args.backoff)
    mins=(time.time()-t0)/60; print(f"\n[done] {ok}/{len(combos)} ok in {mins:.1f} min")

if __name__=="__main__":
    main()
