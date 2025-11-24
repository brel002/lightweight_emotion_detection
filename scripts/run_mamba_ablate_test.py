#!/usr/bin/env python3
import argparse, csv, math, os, subprocess, sys, time
from pathlib import Path
from scripts.config import OUTPUTS_DIR

FIXED_BASE = [
    "--epochs=40",
    "--patience=10",
    "--batch-size=32",
    "--seed=42",
    "--early-metric=acc",
    "--uar-floor=0.56",
    "--uar-warmup=6",
    "--fold-idx=0",
    "--folds-json=scripts/ravdess_5cv_val.json",  # train/val splits
    "--ablation-phase=4",
]

# round floats to stabilize keys
def round_floats(x):      
        return None if x is None else round(float(x), 6)

def f(x):
    try: return float(x)
    except: return float("nan")

def load_topk(csv_path: Path, top_k: int, metric_col: str):
    rows=[]
    with csv_path.open(newline="",encoding="utf-8") as fcsv:
        rdr=csv.DictReader(fcsv)
        for row_index, r in enumerate(rdr):
            dm=f(r.get("d_model"))
            ly=f(r.get("layers"))
            dp=f(r.get("dropout"))
            pool=(r.get("pool") or "attn").strip()
            ls=f(r.get("label_smoothing") or 0.0)
            md=(r.get("use_moddrop") or "0").strip().lower() in {"1","true","t","yes","y"}
            tin=(r.get("tin") or "off").strip()  # in case you log it
            m=f(r.get(metric_col or "val_acc"))
            lr=f(r.get("lr"))
            wd=f(r.get("weight_decay"))

            
            if any(math.isnan(z) for z in (dm,ly,dp,m)): 
                continue
            
            rows.append({# phase 0
                        "d_model":int(dm),
                         "layers":int(ly),
                         "dropout":float(dp),

                        # phase 1
                         "pool":pool,

                         #pahse 2
                         "label_smoothing":float(ls),
                         "use_moddrop":bool(md),

                        # phase 3
                        "lr": round_floats(lr),
                        "weight_decay": round_floats(wd),


                         "tin":tin,
                         "metric":float(m),
                         "_ridx": row_index,})
    best={}
    for r in rows:
        k=(r["d_model"], 
           r["layers"], 
           round(r["dropout"],6), 

           r["pool"], 
           
           round(r["label_smoothing"],6), 
           r["use_moddrop"], 
           
           r["lr"],
           r["weight_decay"],   
           
           r["tin"])
        
        # hold unique configurations
        if (k not in best) or (                     # we haven’t seen this config
            r["metric"]> best[k]["metric"]) or (    # same config, but this run (row) scored higher
            (r["metric"]== best[k]["metric"] and r["_ridx"] < best[k]["_ridx"])  # same metric, pick up row from earlier run
        ): 
            best[k]=r
    
    uniq=list(best.values())
    uniq.sort(key=lambda x:x["metric"], reverse=True)
    return uniq[:top_k]

def main():
    ap=argparse.ArgumentParser("Mamba test phase (CV + retrain)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--metric", default="val_acc")
    ap.add_argument("--cpus", type=int, default=0)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--folds-json", default="scripts/ravdess_5cv_val_test.json")  # 5 folds: train,va, test splits
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retrain-trainval", action="store_true", default=None)
    args=ap.parse_args()

    if args.cpus>0:
        for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            os.environ[k]=str(args.cpus)
        os.environ["MKL_DYNAMIC"]="FALSE"; os.environ["OMP_DYNAMIC"]="FALSE"
        os.environ["OMP_PROC_BIND"]="TRUE"; os.environ["OMP_PLACES"]="cores"

    EXPER_DIR = OUTPUTS_DIR / "experiments" / args.backbone
    csv_path  = Path(args.csv or (EXPER_DIR/"mamba_results.csv"))
    if not csv_path.exists():
        print(f"[ERR] CSV not found: {csv_path}"); sys.exit(2)

    cfgs = load_topk(csv_path, args.top_k, args.metric)
    if not cfgs: print("[ERR] no configs to test"); sys.exit(2)

    print(f"[test] running CV + retrain on top-{len(cfgs)} configs")
    for i,base in enumerate(cfgs,1):

        dm = base["d_model"]
        ly = base["layers"]
        pool = base["pool"]
        dp = base["dropout"]
        ls = base["label_smoothing"]    
        md = base["use_moddrop"]
        lr = base["lr"]
        wd = base["weight_decay"]   
        tin = base["tin"]

        tag=f"d_model={dm} layers={ly} pool={pool} dropout={dp} label_smoothing={ls} use_moddrop={int(md)} lr={lr} wd={wd} tin={tin} phase=4"

        print(f"\n=== {tag} ===")
        cmd=[args.python,"-u","-m","scripts.train_mamba_CV",
             f"--backbone={args.backbone}",
             f"--d-model={dm}", 
             f"--layers={ly}", 
             f"--dropout={dp}",
             f"--pool={pool}",
             f"--label-smoothing={ls}",
             "--use-moddrop" if md else "",
             f"--tin={tin}",
             f"--lr={lr}", 
             f"--weight-decay={wd}",
             "--epochs=40", 
             "--patience=10", 
             "--batch-size=32",
             f"--folds-json={args.folds_json}",
             f"--seed={args.seed}",
             "--early-metric=acc",
             "--uar-floor=0.56",
             "--uar-warmup=6",
             "--ablation-phase=4",]
        
        if args.retrain_trainval:
            cmd.append("--retrain-trainval")



        cmd=[c for c in cmd if c!=""]
        print(" ".join(cmd))
        rc=subprocess.call(cmd)
        if rc!=0: print(f"[WARN] rc={rc} for {tag}")

if __name__=="__main__":
    main()
