#!/usr/bin/env python3
"""
Full QRGS grid launcher: drives ../src/rerun_foldcontained.py over the
72-config grid x 5 seeds x 4 training sizes, in parallel and checkpointed.

Fixes baked in via the runner: fold-contained PCA + balancing, epochs=K*(N+1),
saved <Z> scores + per-iteration loss. T=1000 is run first (priority order).

Paths are configurable through environment variables (defaults assume the repo
layout and that the dataset CSV sits at ./data/):
    QRGS_SRC      directory containing rerun_foldcontained.py   (default ../src)
    QRGS_DATA     path to BCCC-CIRA-CIC-DoHBrw-2020.csv          (default data/...)
    QRGS_RAW      cache dir for X.npy / y.npy                    (default results/raw)
    QRGS_RESULTS  output root for per-run JSON                   (default results/results_fc)
    QRGS_PYTHON   python interpreter for worker processes        (default this one)
"""
import argparse, glob, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE    = os.path.dirname(os.path.abspath(__file__))
SRC     = os.environ.get("QRGS_SRC", os.path.normpath(os.path.join(HERE, "..", "src")))
RUNNER  = os.path.join(SRC, "rerun_foldcontained.py")
DATA    = os.environ.get("QRGS_DATA", os.path.join("data", "BCCC-CIRA-CIC-DoHBrw-2020.csv"))
RAW     = os.environ.get("QRGS_RAW", os.path.join("results", "raw"))
PY      = os.environ.get("QRGS_PYTHON", sys.executable)
RESULTS = os.environ.get("QRGS_RESULTS", os.path.join("results", "results_fc"))
SEEDS   = [7, 13, 37, 42, 101]
SIZES   = [1000, 500, 200, 100]          # priority order: T=1000 first
K       = 5.0

SCALERS = ["min0max2pi", "min0max1"]
QUBITS  = [1, 2, 4]
LAYERS  = [2, 4, 6, 8]


def full_grid():
    combos = []
    for scaler in SCALERS:
        for q in QUBITS:
            for l in LAYERS:
                if q == 1:
                    combos.append(dict(feature_scaler=scaler, q_num_qubit=1,
                                       q_num_layers=l, q_entanglement="false",
                                       q_reup_method="symmetrical"))
                else:
                    for ent in ["true", "false"]:
                        for mt in ["symmetrical", "asymmetrical"]:
                            combos.append(dict(feature_scaler=scaler, q_num_qubit=q,
                                               q_num_layers=l, q_entanglement=ent,
                                               q_reup_method=mt))
    return combos                                          # 8 + 32 + 32 = 72


def N_of(c):
    cc = 12 if c["q_reup_method"] == "symmetrical" else 6
    return cc * c["q_num_qubit"] * c["q_num_layers"]


def epochs_of(c):
    return int(K * (N_of(c) + 1))


def run_name_for(c):
    return (f"gs_sc-{c['feature_scaler']}_qb-{c['q_num_qubit']}_ly-{c['q_num_layers']}"
            f"_en-{c['q_entanglement']}_mt-{c['q_reup_method']}")


def one_run(combo, size, seed):
    rn = run_name_for(combo)
    rroot = Path(RESULTS) / f"T{size}" / f"seed_{seed}"
    (rroot / "logs").mkdir(parents=True, exist_ok=True)
    run_dir = rroot / rn
    if (run_dir / "all_metrics.json").exists():
        return (size, seed, rn, "skip", 0.0)
    ep = epochs_of(combo)
    cmd = [PY, RUNNER, "--dataset_path", DATA, "--raw_dir", RAW,
           "--run_dir", str(run_dir),
           "--feature_scaler", combo["feature_scaler"],
           "--q_num_qubit", str(combo["q_num_qubit"]),
           "--q_num_layers", str(combo["q_num_layers"]),
           "--q_entanglement", combo["q_entanglement"],
           "--q_reup_method", combo["q_reup_method"],
           "--epochs", str(ep), "--seed", str(seed),
           "--train_count", str(size), "--test_count", "4000", "--n_splits", "3"]
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"; env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"; env.pop("PYTHONPATH", None)
    t0 = time.time()
    with open(rroot / "logs" / f"{rn}.log", "w") as lf:
        r = subprocess.run(cmd, cwd=SRC, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return (size, seed, rn, "ok" if r.returncode == 0 else f"rc{r.returncode}",
            round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    grid = full_grid()
    jobs = [(c, size, seed) for size in args.sizes for c in grid for seed in args.seeds]
    if args.limit:
        jobs = jobs[:args.limit]

    print(f"[{time.strftime('%H:%M:%S')}] grid={len(grid)} jobs={len(jobs)} "
          f"workers={args.workers} sizes={args.sizes} seeds={args.seeds}", flush=True)
    if args.dry_run:
        tot = sum(3 * epochs_of(c) for (c, s, sd) in jobs)
        seen = {}
        for c in grid:
            seen[(c["q_num_qubit"], c["q_num_layers"], c["q_reup_method"], N_of(c), epochs_of(c))] = 1
        for k in sorted(seen):
            print(f"  Q{k[0]} L{k[1]} {k[2][:3]} N={k[3]:<4} epochs={k[4]}")
        print(f"TOTAL jobs={len(jobs)} total COBYLA evals (x3 folds) = {tot:,}")
        return

    # ensure raw cache exists
    if not os.path.exists(os.path.join(RAW, "X.npy")):
        print("building raw cache...", flush=True)
        subprocess.run([PY, RUNNER, "--prep", "--dataset_path", DATA, "--raw_dir", RAW],
                       cwd=SRC, env={**os.environ, "PYTHONNOUSERSITE": "1"})

    done = fail = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one_run, c, size, seed) for (c, size, seed) in jobs]
        for f in as_completed(futs):
            size, seed, rn, status, secs = f.result()
            done += 1
            if status.startswith("rc"):
                fail += 1
            print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(jobs)} T{size} "
                  f"seed{seed} {status} {secs}s {rn}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] COMPLETE done={done} failed={fail} "
          f"wall_min={(time.time()-t0)/60:.1f}", flush=True)


if __name__ == "__main__":
    main()
