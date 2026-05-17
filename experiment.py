"""
experiment.py
-------------
Entry point CLI per tutti gli sweep QRC.

Utilizzo:
    python experiment.py <sweep> [opzioni]

Sweep disponibili:
    dt          → alpha x dt
    c           → c x alpha_fixed  (a alpha piccolo/medio/grande)
    filling     → alpha x nu=n/N
    eps_gamma   → alpha x (epsilon, gamma)
    J           → alpha x J
    all         → tutti in sequenza

Opzioni comuni:
    --n-jobs N      worker paralleli (-1 = tutti i core, default)
    --n-samples N   campioni per punto (default 20)
    --outdir DIR    cartella output (default: results/)
    --save-traces   salva le tracce temporali nel JSON (più pesante)
    --no-single     non calcola S_single_entropy
    --no-fermionic  non calcola S_fermionic_entropy

Esempi:
    python experiment.py dt --n-jobs 8
    python experiment.py filling --n-samples 30 --n-jobs 8
    python experiment.py eps_gamma --n-jobs 4
    python experiment.py all --n-jobs 8 --outdir results/run1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import argparse
import numpy as np

from qrc.sweeps import sweep_dt, sweep_c, sweep_filling, sweep_eps_gamma, sweep_J


# ---------------------------------------------------------------------------
# Serializzazione (numpy → tipi base Python)
# ---------------------------------------------------------------------------

def _serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_serialize(data), f, indent=2)
    print(f"  → Salvato: {path}")


# ---------------------------------------------------------------------------
# Runner per ogni sweep
# ---------------------------------------------------------------------------

def run_sweep_dt(args):
    print("\n" + "="*60)
    print("SWEEP: alpha x dt")
    print("="*60)
    results = sweep_dt(
        n_samples=args.n_samples, n_jobs=args.n_jobs,
        compute_single_entropy=not args.no_single,
        compute_fermionic_entropy=not args.no_fermionic,
        save_traces=args.save_traces,
    )
    save_json(results, os.path.join(args.outdir, "sweep_dt.json"))


def run_sweep_c(args):
    print("\n" + "="*60)
    print("SWEEP: c x alpha_fixed")
    print("="*60)
    results = sweep_c(
        n_samples=args.n_samples, n_jobs=args.n_jobs,
        compute_single_entropy=not args.no_single,
        compute_fermionic_entropy=not args.no_fermionic,
        save_traces=args.save_traces,
    )
    save_json(results, os.path.join(args.outdir, "sweep_c.json"))


def run_sweep_filling(args):
    print("\n" + "="*60)
    print("SWEEP: alpha x nu (filling fermionico)")
    print("="*60)
    results = sweep_filling(
        n_samples=args.n_samples, n_jobs=args.n_jobs,
        compute_single_entropy=not args.no_single,
        compute_fermionic_entropy=not args.no_fermionic,
        save_traces=args.save_traces,
    )
    save_json(results, os.path.join(args.outdir, "sweep_filling.json"))


def run_sweep_eps_gamma(args):
    print("\n" + "="*60)
    print("SWEEP: alpha x (epsilon, gamma)")
    print("="*60)
    results = sweep_eps_gamma(
        n_samples=args.n_samples, n_jobs=args.n_jobs,
        compute_single_entropy=not args.no_single,
        compute_fermionic_entropy=not args.no_fermionic,
        save_traces=args.save_traces,
    )
    save_json(results, os.path.join(args.outdir, "sweep_eps_gamma.json"))


def run_sweep_J(args):
    print("\n" + "="*60)
    print("SWEEP: alpha x J")
    print("="*60)
    results = sweep_J(
        n_samples=args.n_samples, n_jobs=args.n_jobs,
        compute_single_entropy=not args.no_single,
        compute_fermionic_entropy=not args.no_fermionic,
        save_traces=args.save_traces,
    )
    save_json(results, os.path.join(args.outdir, "sweep_J.json"))


SWEEP_RUNNERS = {
    "dt":        run_sweep_dt,
    "c":         run_sweep_c,
    "filling":   run_sweep_filling,
    "eps_gamma": run_sweep_eps_gamma,
    "J":         run_sweep_J,
}


# ---------------------------------------------------------------------------
# Stima tempi prima di lanciare
# ---------------------------------------------------------------------------

def _estimate_time(sweep_name, n_samples, n_jobs):
    """Stima molto approssimativa basata su ~2.5s per campione seriale."""
    T_PER_SAMPLE = 2.5   # secondi, da benchmark precedente

    sizes = {
        "dt":        23 * 4,    # 23 alpha x 4 dt
        "c":         10 * 3,    # 10 c x 3 alpha_fixed
        "filling":   23 * 5,    # 23 alpha x 5 nu
        "eps_gamma": 23 * 16,   # 23 alpha x 16 combo (eps x gamma)
        "J":         23 * 5,    # 23 alpha x 5 J
    }
    if sweep_name == "all":
        n_points = sum(sizes.values())
    else:
        n_points = sizes.get(sweep_name, 23)

    total_serial = n_points * n_samples * T_PER_SAMPLE
    effective_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
    total_parallel = total_serial / max(effective_jobs, 1)

    print(f"\n  Stima tempo ({sweep_name}):")
    print(f"    Punti parametrici: {n_points}")
    print(f"    Campioni/punto:    {n_samples}")
    print(f"    Worker:            {effective_jobs}")
    print(f"    Seriale:           {total_serial/60:.0f} min")
    print(f"    Parallelo (~x{effective_jobs}): {total_parallel/60:.0f} min")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="QRC sweep experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "sweep",
        choices=list(SWEEP_RUNNERS.keys()) + ["all"],
        help="Sweep da eseguire",
    )
    parser.add_argument("--n-jobs",     type=int,  default=-1,  help="Worker paralleli (-1=tutti)")
    parser.add_argument("--n-samples",  type=int,  default=20,  help="Campioni per punto")
    parser.add_argument("--outdir",     type=str,  default="results", help="Cartella output")
    parser.add_argument("--save-traces",action="store_true",    help="Salva tracce temporali")
    parser.add_argument("--no-single",  action="store_true",    help="Salta S_single_entropy")
    parser.add_argument("--no-fermionic",action="store_true",   help="Salta S_fermionic_entropy")
    parser.add_argument("--dry-run",    action="store_true",    help="Stima tempi senza eseguire")

    args = parser.parse_args()

    _estimate_time(args.sweep, args.n_samples, args.n_jobs)

    if args.dry_run:
        print("\n  --dry-run: nessun calcolo avviato.")
        return

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n  Output in: {args.outdir}/\n")

    if args.sweep == "all":
        for name, runner in SWEEP_RUNNERS.items():
            runner(args)
    else:
        SWEEP_RUNNERS[args.sweep](args)

    print(f"\nFatto. Risultati in: {args.outdir}/")


if __name__ == "__main__":
    main()
