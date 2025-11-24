# scripts/log_results.py
from pathlib import Path
import csv, time, os

class ResultsLogger:
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self):
        rows, cols = [], []
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            with self.csv_path.open("r", newline="") as f:
                r = csv.DictReader(f)
                cols = list(r.fieldnames or [])
                rows = list(r)
        return rows, cols

    def _write_all(self, rows, cols):
        with self.csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow(row)

    def append(self, row: dict):
        # 1) sanitize keys (underscores; avoid dup seed)
        clean = {}
        for k, v in row.items():
            kk = k.replace("-", "_")
            clean[kk] = v
        if "seed" in clean and "SEED" in clean:
            clean.pop("SEED", None)

        # 2) load existing rows/cols
        rows, cols = self._read_all()

        # 3) union of columns (preserve original order; append new ones)
        new_keys = [k for k in clean.keys() if k not in cols]
        cols = cols + new_keys if cols else list(clean.keys())

        # 4) append and rewrite file (header may have grown)
        rows.append(clean)
        self._write_all(rows, cols)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def file_size_mb(path):
    return round(os.path.getsize(path)/(1024*1024), 4) if Path(path).exists() else ""

def benchmark_latency_cpu(forward_fn, warmup=5, iters=30):
    import time
    for _ in range(warmup): forward_fn()
    t0 = time.time()
    for _ in range(iters): forward_fn()
    return round((time.time()-t0)*1000.0/iters, 3)