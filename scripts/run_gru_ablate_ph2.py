#!/usr/bin/env python3
import argparse, csv, math, itertools, os, signal, subprocess, sys, time
from pathlib import Path
from scripts.config import OUTPUTS_DIR 


FIXED_ARGS = [
    "--lr=7e-4",
    "--weight-decay=1e-3",
    "--label-smoothing=0.015",
    "--margin=0",
    "--fold-idx=0",
    "--folds-json=scripts/ravdess_5cv_val.json",
    "--skip-test",
    "--uar-floor=0.56",
    "--uar-warmup=6",
    "--early-metric=acc",
]



def parse_float_safe(x):
    try:
        return float(x)
    except Exception:
        return float("nan")
    

def parse_optional_int(x, none_vals=("none", "null", "", "nan")):
    s = "" if x is None else str(x).strip().lower()
    if s in none_vals:
        return None
    try:
        return int(s)
    except Exception:
        return None

def parse_bool_str(x, default=False):
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"1","true","t","yes","y"}:
        return True
    if s in {"0","false","f","no","n",""}:
        return False
    return default  # unexpected token → default

def parse_fuse_mode(x, default="concat"):
    s = ("" if x is None else str(x).strip().lower())
    return s if s in {"concat","replace"} else default

# round floats to stabilize keys
def round_floats(x):      
        return None if x is None else round(float(x), 6)

def configuration_key(x):       
        return (
            int(x["hidden"]),
            int(x["layers"]),
            round_floats(x["pdrop"]),

            x["pool"],
            x["topk"],                 # only matters when pool=='topk'

            x["proj_dim"],
            x["fuse_mode"],
            x["fuse_dim"],
            bool(x["use_gate"]),
        )



def load_topk(csv_path: Path, top_k: int, metric_name=None, filter_gap=None):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_index, r in enumerate(reader):
        

             # Phase-0 params    
            h = parse_float_safe(r.get("hidden"))      
            l = parse_float_safe(r.get("num_layers"))   
            pd = parse_float_safe(r.get("p_drop"))      
            if any(math.isnan(x) for x in (h, l, pd)):
                continue

            proj_dim  = parse_optional_int(r.get("proj_dim") )      # this can be 256 or 384 or empty ( blank) in results file!
            use_gate  = parse_bool_str(r.get("use_gate") )          # TRUE or FALSE in results file
            fuse_mode = parse_fuse_mode(r.get("fuse_mode") )       # concat or replace
            fuse_dim  = parse_optional_int(r.get("fuse_dim") ) or 192   # 193 or 256

            # Phase-1 params
            pool = r.get("pool_type")                   
            topk = r.get("top_k")                       # used with pool="topk"

            # pick metric column to rank the results when selecting top-k (default to validation metric)
            mcol = metric_name or "val_acc"
            metric = parse_float_safe(r.get(mcol))
            if math.isnan(metric):
                continue                        


            # load a result from an ablation run: configuration of hyper-params
            rows.append({
                "hidden": int(h),
                "layers": int(l),
                "pdrop": float(pd),
                "pool":pool,
                "topk":topk,
                "metric": float(metric),
                "proj_dim":proj_dim,            # can be None, 256 or 384
                "fuse_mode":fuse_mode,
                "fuse_dim":fuse_dim,
                "use_gate":use_gate,
                "_ridx": row_index,
               # "raw": r
            })

    # iterate through configurations and de-duplicate
    best = {}           # dictionary collection, k:row pairs where row is a dictionary of hyperparams and their values
    for row in rows:
        #
        k = configuration_key(row)  # get a configuration key: a tuple of hyper-param values from the row
        cur = best.get(k)           # look up if we’ve already stored a best-so-far row for this configuration key.
        
        # make decission to insert/update the run into collection of best ones
        if  (cur is None) or (                          # we haven’t seen this config
            row["metric"] > cur["metric"]) or (         # same config, but this run (row) scored higher
            row["metric"] == cur["metric"] and row["_ridx"] < cur["_ridx"]  # same metric, pick up row from earlier run
        ):
            best[k] = row           # add/replace the configuration ( row dictionary) in the collection

    # get top-k unique configurations
    uniq = list(best.values())
    uniq.sort(key=lambda x: x["metric"], reverse=True)
    return uniq[:top_k]


def set_math_threads(n):
    if n <= 0: return
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)
    os.environ["MKL_DYNAMIC"]="FALSE"; 
    os.environ["OMP_DYNAMIC"]="FALSE"
    os.environ["OMP_PROC_BIND"]="TRUE"; 
    os.environ["OMP_PLACES"]="cores"

def run_and_report(cmd):
    return subprocess.call(cmd) 


# set regularization combos
def reg_grid():

    return [
       #time_m, freq_m, mod_drop
        # Pure baselines (no SpecAug)
        (0.00, 0.00, 0.00),
        (0.00, 0.00, 0.05),
        (0.00, 0.00, 0.10),

        # Very light SpecAug (no mod-drop)
        (0.03, 0.00, 0.00),
        (0.00, 0.03, 0.00),
        (0.05, 0.00, 0.00),
        (0.00, 0.05, 0.00),
        (0.03, 0.03, 0.00),

        # Light SpecAug + light mod-drop
        (0.03, 0.00, 0.05),
        (0.00, 0.03, 0.05),

        # Very light SpecAug + mod-drop=0.10 (kept gentle)
        (0.03, 0.00, 0.10),
        (0.00, 0.03, 0.10),
    ]

def main():
    ap = argparse.ArgumentParser("Phase-2 ablation (SpecAug/ModDrop) on top-K Phase-0,1 results")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--csv", default=None)  # concatenated results from Phase0 and Phase1
    ap.add_argument("--top-k", type=int, default=3)         # consider top-k results from previous phases
    ap.add_argument("--metric", default=None)

    ap.add_argument("--filter-gap", type=float, default=None, help="Drop rows where |val_prec - test_prec| > this")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff", type=int, default=10)
    ap.add_argument("--cpus", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra", nargs="*", default=[])
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

    set_math_threads(args.cpus)

    # get file with results from previous ablation
    EXPER_DIR   = OUTPUTS_DIR  / "experiments" / args.backbone
    RESULTS_GRU_CSV = EXPER_DIR / "gru_results.csv"  
    csv_path = args.csv or RESULTS_GRU_CSV
    csv_path = Path(csv_path)
    

    if not csv_path.exists():
        print(f"[ERR] CSV not found: {csv_path}")
        sys.exit(2)


    



    base_cfgs = load_topk(csv_path, args.top_k, args.metric, args.filter_gap)

    if not base_cfgs:
        print("No base configs found after filtering. Check --csv/--metric/--filter-gap.")
        sys.exit(2)

    #combos = list(itertools.product(TIME_MASKS, FREQ_MASKS, MOD_DROPS))
    # get reduced, explicitly set-up combos
    combos = reg_grid()

    total = len(base_cfgs) * len(combos)
    print(f"[grid] Phase-2: {total} runs (from top-{len(base_cfgs)} Phase-1 configs)")
    ok = 0; t0 = time.time()

    for idx, base in enumerate(base_cfgs, 1):
        # unpack hyperparams from previous phases
        h        = base["hidden"]
        l        = base["layers"]
        pd       = base["pdrop"]
        proj_dim = base["proj_dim"]     # None or int
        use_gate = base["use_gate"]     # bool
        fuse_mode= base["fuse_mode"]    # "concat"|"replace"
        fuse_dim = base["fuse_dim"] 
        pool     = base["pool"]
        topk     = base["topk"]
        
      
        
        # Iterate over grid of variations set in this phase
        for (tm,fm,md) in combos:
            tag = (f"h={h} l={l} p_drop={pd} proj_dim={proj_dim} use_gate={use_gate} fuse_mode={fuse_mode} fuse_dim={fuse_dim}" + 
                   (f":k={topk}" if topk else "") + 
                   f"pool={pool} tm={tm} fm={fm} md={md}"
            )
            print(f"\n=== [{ok+1}/{total}] {tag} ===")      
            
            cmd = [
                args.python, "-u", "-m", "scripts.train_gru_CV",

                # Phase-0 params 
                f"--hidden={h}",                 
                f"--num-layers={l}",            
                f"--p-drop={pd}",               
                
                (f"--proj-dim={proj_dim}" if proj_dim is not None else "--proj-dim=0"),
                f"--fuse-mode={fuse_mode}",
                f"--fuse-dim={fuse_dim}",

                # Phase-1 params
                f"--pool-type={pool}",          

                # variations set in this phase (Phase-2)
                f"--mod-drop={md}",             
                f"--time-mask-ratio={tm}",      
                f"--freq-mask-ratio={fm}",     
                
                f"--seed={args.seed}",
                "--ablation-phase=2",   
                f"--backbone={args.backbone}",        
                *FIXED_ARGS,

            ]
                   
            if pool == "topk" and topk:
                cmd += [f"--top-k={topk}"]      # set in Phase-1

            if use_gate:
                cmd.append("--use-gate")

            cmd += args.extra

            print(" ".join(cmd))

            for attempt in range(1, args.retries+1):
                print(f"[{time.strftime('%H:%M:%S')}] {tag} (try {attempt}/{args.retries})")
                rc = run_and_report(cmd)
                if rc == 0: 
                    ok += 1; 
                    break
                if attempt < args.retries:
                    print(f"[WARN] rc={rc}; sleeping {args.backoff}s…")
                    time.sleep(args.backoff)

    mins = (time.time() - t0)/60
    print(f"\n[done] {ok}/{total} succeeded in {mins:.1f} min")

if __name__ == "__main__":
    main()
