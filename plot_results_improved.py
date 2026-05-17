r"""
plot_results_improved.py
------------------------
Plot più informativi per sweep QRC/NARMA10.

Obiettivo:
- vedere correlazioni tra entropie S, NARMA10/NMSE e Memory Capacity;
- capire se minimi di S corrispondono a massimi di performance:
  * performance NARMA10 migliore = NMSE minima
  * performance MC migliore = MC massima
- confrontare alpha e c, anche per JSON annidati tipo sweep_eps_gamma, sweep_J,
  sweep_dt, sweep_filling, sweep_c.

Uso:
    python plot_results_improved.py sweep_eps_gamma\(1\).json sweep_J\(1\).json
oppure:
    python plot_results_improved.py *.json

Output:
    plots_informative/
      summary_correlations.csv
      best_points.csv
      *_overview_normalized.png
      *_pareto_MC_vs_NMSE_colored_by_S.png
      *_alignment_minS_bestPerformance.png
      *_heatmap_NMSE.png, *_heatmap_MC.png, *_heatmap_S_static.png
      *_scatter_grid_correlations.png
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


S_METRICS = [
    ("S_inf_static_mean", r"$S_{\infty}^{static}$"),
    ("S_single_dynamic_mean", r"$S_{single}^{dyn}$"),
    ("S_fermionic_mean", r"$S_{ferm}$"),
    ("dS_single_peak_mean", r"$\max |dS_{single}/dt|$"),
    ("dS_fermionic_peak_mean", r"$\max |dS_{ferm}/dt|$"),
]

PERF_METRICS = [
    ("nmse_mean", "NMSE NARMA10", "min"),
    ("mc_total_mean", "Memory Capacity", "max"),
]


# ============================================================
# IO / flatten robusto
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def scalar_context(d):
    return {k: v for k, v in d.items() if is_number(v) or isinstance(v, str) or v is None}


def flatten_results(data, source_name="results"):
    """
    Ritorna lista di righe con un record per punto di sweep.
    Supporta:
    1) file flat con chiavi tipo alphas, nmse_mean, ...
    2) file annidati:
       {
         "eps_...": {"alphas": [...], "sweep": [{...}, ...]},
         "J_...":   {"alphas": [...], "sweep": [{...}, ...]},
         "alpha_0.30": {"c_values": [...], "sweep": [{...}, ...]}
       }
    """
    rows = []

    # Caso flat: arrays paralleli al top-level.
    if isinstance(data, dict) and "alphas" in data and "nmse_mean" in data:
        alphas = data.get("alphas", [])
        n = len(alphas)
        for i in range(n):
            row = {
                "source": source_name,
                "group": Path(source_name).stem,
                "sweep_param": "alpha",
                "x": float(alphas[i]),
                "alpha": float(alphas[i]),
            }
            for k, v in data.items():
                if isinstance(v, list) and len(v) == n and all(is_number(z) or z is None for z in v):
                    row[k] = np.nan if v[i] is None else float(v[i])
                elif is_number(v) or isinstance(v, str) or v is None:
                    row[k] = v
            rows.append(row)
        return rows

    if not isinstance(data, dict):
        return rows

    # Caso annidato.
    for group, block in data.items():
        if not isinstance(block, dict) or "sweep" not in block:
            continue

        sweep = block.get("sweep", [])
        if "alphas" in block:
            xs = block["alphas"]
            sweep_param = "alpha"
        elif "c_values" in block:
            xs = block["c_values"]
            sweep_param = "c"
        else:
            xs = list(range(len(sweep)))
            sweep_param = "index"

        context = scalar_context(block)

        for i, point in enumerate(sweep):
            if not isinstance(point, dict):
                continue

            x = xs[i] if i < len(xs) else np.nan
            row = {
                "source": source_name,
                "group": group,
                "sweep_param": sweep_param,
                "x": float(x) if is_number(x) else np.nan,
            }

            # Parametri a livello di gruppo, es: epsilon/gamma/J/dt/alpha_fixed/nu.
            row.update(context)

            # Parametri del singolo punto.
            for k, v in point.items():
                if k == "params" and isinstance(v, dict):
                    for pk, pv in v.items():
                        row[pk] = pv
                elif is_number(v) or isinstance(v, str) or v is None:
                    row[k] = np.nan if v is None else v

            # Alias comodi.
            if "alpha" not in row and sweep_param == "alpha":
                row["alpha"] = row["x"]
            if "c" not in row and sweep_param == "c":
                row["c"] = row["x"]

            rows.append(row)

    return rows


def load_many(files):
    rows = []
    for file in files:
        data = load_json(file)
        rows.extend(flatten_results(data, source_name=os.path.basename(file)))
    return rows


# ============================================================
# Statistiche
# ============================================================

def to_float_array(rows, key):
    vals = []
    for r in rows:
        v = r.get(key, np.nan)
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(np.nan)
    return np.array(vals, dtype=float)


def valid_mask(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = valid_mask(x, y)
    if m.sum() < 3 or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def spearman(x, y):
    # Implementazione senza scipy: rank medio semplice abbastanza per sweep continui.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = valid_mask(x, y)
    if m.sum() < 3:
        return np.nan
    rx = rankdata(x[m])
    ry = rankdata(y[m])
    return pearson(rx, ry)


def rankdata(a):
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # Gestione tie: media dei rank.
    vals = a[order]
    start = 0
    while start < len(a):
        end = start + 1
        while end < len(a) and vals[end] == vals[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def zscore_for_plot(y):
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    out = np.full_like(y, np.nan, dtype=float)
    if m.sum() == 0:
        return out
    lo, hi = np.nanmin(y[m]), np.nanmax(y[m])
    if hi == lo:
        out[m] = 0.5
    else:
        out[m] = (y[m] - lo) / (hi - lo)
    return out


def performance_score(rows):
    """Score intuitivo: alto = meglio. Usa -NMSE e MC normalizzati."""
    nmse = to_float_array(rows, "nmse_mean")
    mc = to_float_array(rows, "mc_total_mean")
    score = np.full(len(rows), np.nan)
    terms = []
    if np.isfinite(nmse).sum() > 0:
        terms.append(1.0 - zscore_for_plot(nmse))
    if np.isfinite(mc).sum() > 0:
        terms.append(zscore_for_plot(mc))
    if terms:
        stack = np.vstack(terms)
        good = np.isfinite(stack)
        denom = good.sum(axis=0)
        summed = np.where(good, stack, 0.0).sum(axis=0)
        score = np.full(len(rows), np.nan)
        score[denom > 0] = summed[denom > 0] / denom[denom > 0]
    return score


def group_rows(rows):
    groups = {}
    for r in rows:
        key = (r.get("source", "source"), r.get("group", "group"))
        groups.setdefault(key, []).append(r)
    return groups


def short_name(source):
    return Path(source).stem.replace("(", "").replace(")", "").replace(" ", "_")


# ============================================================
# Plot helpers
# ============================================================

def savefig(fig, outdir, filename):
    ensure_dir(outdir)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Salvato: {path}")
    plt.close(fig)


def annotate_best(ax, x, y, label, color="black"):
    if not np.isfinite(x) or not np.isfinite(y):
        return
    ax.scatter([x], [y], s=95, marker="*", zorder=5, color=color)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


# ============================================================
# Plot 1: overview normalizzato per singolo gruppo
# ============================================================

def plot_overview_group(rows, outdir, source, group):
    rows = sorted(rows, key=lambda r: r.get("x", np.nan))
    x = to_float_array(rows, "x")
    nmse = to_float_array(rows, "nmse_mean")
    mc = to_float_array(rows, "mc_total_mean")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title(f"{group}: performance e S normalizzate")
    ax.set_xlabel(rows[0].get("sweep_param", "x"))
    ax.set_ylabel("scala normalizzata [0, 1]")

    if np.isfinite(nmse).sum():
        ax.plot(x, 1 - zscore_for_plot(nmse), marker="o", linewidth=2.2, label="NARMA10 performance = 1 - norm(NMSE)")
        i = np.nanargmin(nmse)
        annotate_best(ax, x[i], 1 - zscore_for_plot(nmse)[i], f"min NMSE\n{x[i]:.3g}")

    if np.isfinite(mc).sum():
        ax.plot(x, zscore_for_plot(mc), marker="s", linewidth=2.0, label="MC normalizzata")
        i = np.nanargmax(mc)
        annotate_best(ax, x[i], zscore_for_plot(mc)[i], f"max MC\n{x[i]:.3g}")

    for key, label in S_METRICS[:3]:
        y = to_float_array(rows, key)
        if np.isfinite(y).sum() >= 3:
            ax.plot(x, 1 - zscore_for_plot(y), marker=".", linewidth=1.5, alpha=0.8, label=f"1 - norm({label})")
            i = np.nanargmin(y)
            annotate_best(ax, x[i], 1 - zscore_for_plot(y)[i], f"min {key.replace('_mean','')}\n{x[i]:.3g}")

    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fname = f"{short_name(source)}__{safe_name(group)}__overview_normalized.png"
    savefig(fig, outdir, fname)


# ============================================================
# Plot 2: scatter grid correlazioni S vs NMSE/MC
# ============================================================

def plot_scatter_grid(rows, outdir, source):
    n = len(S_METRICS)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.0 * n))
    if n == 1:
        axes = np.array([axes])

    nmse = to_float_array(rows, "nmse_mean")
    mc = to_float_array(rows, "mc_total_mean")
    alpha = to_float_array(rows, "alpha")
    color = alpha if np.isfinite(alpha).sum() else to_float_array(rows, "x")

    for i, (skey, slabel) in enumerate(S_METRICS):
        s = to_float_array(rows, skey)

        for j, (perf, perf_label, mode) in enumerate(PERF_METRICS):
            ax = axes[i, j]
            y = nmse if perf == "nmse_mean" else mc
            m = valid_mask(s, y)

            if m.sum() < 3:
                ax.set_visible(False)
                continue

            sc = ax.scatter(s[m], y[m], c=color[m], s=42, alpha=0.85)
            ax.set_xlabel(slabel)
            ax.set_ylabel(perf_label)
            ax.grid(alpha=0.25)

            rp = pearson(s[m], y[m])
            rs = spearman(s[m], y[m])
            ax.text(
                0.03,
                0.97,
                f"Pearson r={rp:.3f}\nSpearman ρ={rs:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Evidenzia punti migliori.
            if perf == "nmse_mean":
                idx_local = np.argmin(y[m])
            else:
                idx_local = np.argmax(y[m])
            xm = s[m][idx_local]
            ym = y[m][idx_local]
            ax.scatter([xm], [ym], marker="*", s=140, color="black", zorder=5)

    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.9, label=r"$\alpha$ o x")
    fig.suptitle(f"{Path(source).name}: correlazioni S ↔ NARMA10/MC", fontsize=14)
    fname = f"{short_name(source)}__scatter_grid_correlations.png"
    savefig(fig, outdir, fname)


# ============================================================
# Plot 3: Pareto MC vs NMSE colorato da S
# ============================================================

def plot_pareto(rows, outdir, source):
    nmse = to_float_array(rows, "nmse_mean")
    mc = to_float_array(rows, "mc_total_mean")
    alpha = to_float_array(rows, "alpha")
    s = to_float_array(rows, "S_inf_static_mean")
    m = valid_mask(nmse, mc, s)

    if m.sum() < 3:
        return

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(nmse[m], mc[m], c=s[m], s=60, alpha=0.9)
    ax.set_xlabel("NMSE NARMA10, più basso = meglio")
    ax.set_ylabel("Memory Capacity, più alto = meglio")
    ax.set_title(f"{Path(source).name}: fronte performance colorato da $S_static$")
    ax.grid(alpha=0.25)

    # Punti top: best combined score.
    score = performance_score(rows)
    mm = np.where(m)[0]
    top = mm[np.argsort(score[mm])[-5:]][::-1]
    for k, idx in enumerate(top, start=1):
        ax.scatter([nmse[idx]], [mc[idx]], marker="*", s=150, color="black", zorder=5)
        lab = f"#{k}"
        if np.isfinite(alpha[idx]):
            lab += f" α={alpha[idx]:.3g}"
        if "c" in rows[idx] and is_number(rows[idx]["c"]):
            lab += f" c={float(rows[idx]['c']):.3g}"
        ax.annotate(lab, (nmse[idx], mc[idx]), xytext=(7, 7), textcoords="offset points", fontsize=8)

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$S_{\infty}^{static}$")
    fname = f"{short_name(source)}__pareto_MC_vs_NMSE_colored_by_S.png"
    savefig(fig, outdir, fname)


# ============================================================
# Plot 4: heatmap gruppo x alpha/c
# ============================================================

def plot_heatmap(rows, outdir, source, metric, label):
    groups = sorted(set(r.get("group", "group") for r in rows))
    xs = sorted(set(float(r.get("x", np.nan)) for r in rows if np.isfinite(float(r.get("x", np.nan)))))
    if len(groups) < 2 or len(xs) < 2:
        return

    mat = np.full((len(groups), len(xs)), np.nan)
    for r in rows:
        g = r.get("group", "group")
        try:
            x = float(r.get("x"))
            y = float(r.get(metric))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        mat[groups.index(g), xs.index(x)] = y

    if np.isfinite(mat).sum() < 4:
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(xs)), max(5, 0.22 * len(groups))))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_title(f"{Path(source).name}: {label}")
    ax.set_xlabel(rows[0].get("sweep_param", "x"))
    ax.set_ylabel("gruppo sweep")
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([f"{v:.3g}" for v in xs], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups, fontsize=7)

    # Evidenzia miglior valore globale.
    if metric == "nmse_mean":
        idx = np.nanargmin(mat)
    elif metric == "mc_total_mean":
        idx = np.nanargmax(mat)
    else:
        idx = np.nanargmin(mat)
    iy, ix = np.unravel_index(idx, mat.shape)
    ax.scatter([ix], [iy], marker="*", s=180, color="white", edgecolor="black", linewidth=0.8)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(label)
    fname = f"{short_name(source)}__heatmap_{safe_name(metric)}.png"
    savefig(fig, outdir, fname)


# ============================================================
# Plot 5: alignment min S vs best performance
# ============================================================

def plot_alignment(rows, outdir, source):
    groups = group_rows(rows)
    records = []

    for (src, group), gr in groups.items():
        gr = sorted(gr, key=lambda r: r.get("x", np.nan))
        x = to_float_array(gr, "x")
        nmse = to_float_array(gr, "nmse_mean")
        mc = to_float_array(gr, "mc_total_mean")

        if np.isfinite(nmse).sum() >= 2:
            x_best_nmse = x[np.nanargmin(nmse)]
        else:
            x_best_nmse = np.nan

        if np.isfinite(mc).sum() >= 2:
            x_best_mc = x[np.nanargmax(mc)]
        else:
            x_best_mc = np.nan

        for skey, slabel in S_METRICS[:3]:
            s = to_float_array(gr, skey)
            if np.isfinite(s).sum() < 2:
                continue
            x_min_s = x[np.nanargmin(s)]
            records.append((group, skey, x_min_s, x_best_nmse, x_best_mc))

    if not records:
        return

    labels = [f"{g}\n{s.replace('_mean','')}" for g, s, *_ in records]
    dist_nmse = [abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan for _, _, a, b, _ in records]
    dist_mc = [abs(a - c) if np.isfinite(a) and np.isfinite(c) else np.nan for _, _, a, _, c in records]

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(records)), 5.5))
    xpos = np.arange(len(records))
    ax.bar(xpos - 0.18, dist_nmse, width=0.36, label=r"$|x_{\min S} - x_{\min NMSE}|$")
    ax.bar(xpos + 0.18, dist_mc, width=0.36, label=r"$|x_{\min S} - x_{\max MC}|$")
    ax.set_title(f"{Path(source).name}: quanto coincidono minimi di S e best performance")
    ax.set_ylabel("distanza nel parametro sweep")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fname = f"{short_name(source)}__alignment_minS_bestPerformance.png"
    savefig(fig, outdir, fname)


# ============================================================
# CSV summary
# ============================================================

def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(format_csv(row.get(h, "")) for h in header) + "\n")
    print(f"Salvato: {path}")


def format_csv(x):
    if x is None:
        return ""
    if isinstance(x, float):
        if np.isnan(x):
            return ""
        return f"{x:.10g}"
    s = str(x)
    if "," in s or "\n" in s:
        s = '"' + s.replace('"', '""') + '"'
    return s


def make_summaries(rows, outdir):
    corr_rows = []
    best_rows = []

    for source in sorted(set(r["source"] for r in rows)):
        sr = [r for r in rows if r["source"] == source]

        for skey, slabel in S_METRICS:
            s = to_float_array(sr, skey)
            for pkey, plabel, mode in PERF_METRICS:
                p = to_float_array(sr, pkey)
                corr_rows.append({
                    "source": source,
                    "S_metric": skey,
                    "performance_metric": pkey,
                    "pearson": pearson(s, p),
                    "spearman": spearman(s, p),
                    "interpretation": "negativo = S minore associata a performance migliore" if pkey == "nmse_mean" else "positivo = S maggiore associata a MC maggiore; negativo = S minore associata a MC maggiore",
                })

        score = performance_score(sr)
        order = np.argsort(score)[::-1]
        for rank, idx in enumerate(order[:15], start=1):
            r = sr[idx]
            best_rows.append({
                "source": source,
                "rank": rank,
                "group": r.get("group", ""),
                "sweep_param": r.get("sweep_param", ""),
                "x": r.get("x", np.nan),
                "alpha": r.get("alpha", np.nan),
                "c": r.get("c", np.nan),
                "nmse_mean": r.get("nmse_mean", np.nan),
                "mc_total_mean": r.get("mc_total_mean", np.nan),
                "S_inf_static_mean": r.get("S_inf_static_mean", np.nan),
                "S_single_dynamic_mean": r.get("S_single_dynamic_mean", np.nan),
                "S_fermionic_mean": r.get("S_fermionic_mean", np.nan),
                "score": score[idx],
            })

    write_csv(
        os.path.join(outdir, "summary_correlations.csv"),
        ["source", "S_metric", "performance_metric", "pearson", "spearman", "interpretation"],
        corr_rows,
    )
    write_csv(
        os.path.join(outdir, "best_points.csv"),
        ["source", "rank", "group", "sweep_param", "x", "alpha", "c", "nmse_mean", "mc_total_mean",
         "S_inf_static_mean", "S_single_dynamic_mean", "S_fermionic_mean", "score"],
        best_rows,
    )


# ============================================================
# Main
# ============================================================

def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def plot_all(files, outdir="plots_informative", max_group_overviews=12):
    ensure_dir(outdir)
    rows = load_many(files)

    if not rows:
        print("Nessun risultato leggibile.")
        return

    make_summaries(rows, outdir)

    for source in sorted(set(r["source"] for r in rows)):
        sr = [r for r in rows if r["source"] == source]
        print(f"\n=== {source}: {len(sr)} punti ===")

        plot_scatter_grid(sr, outdir, source)
        plot_pareto(sr, outdir, source)
        plot_alignment(sr, outdir, source)

        for metric, label in [
            ("nmse_mean", "NMSE NARMA10"),
            ("mc_total_mean", "Memory Capacity"),
            ("S_inf_static_mean", r"$S_{\infty}^{static}$"),
            ("S_single_dynamic_mean", r"$S_{single}^{dyn}$"),
            ("S_fermionic_mean", r"$S_{ferm}$"),
        ]:
            plot_heatmap(sr, outdir, source, metric, label)

        # Overview solo per i gruppi più interessanti:
        # top per score oppure gruppi con best NMSE.
        groups = group_rows(sr)
        group_scores = []
        for (src, group), gr in groups.items():
            score = performance_score(gr)
            best = np.nanmax(score) if np.isfinite(score).sum() else -np.inf
            group_scores.append((best, group, gr))
        group_scores.sort(reverse=True, key=lambda x: x[0])

        for _, group, gr in group_scores[:max_group_overviews]:
            plot_overview_group(gr, outdir, source, group)

    print("\nFatto. Guarda soprattutto:")
    print("  - summary_correlations.csv")
    print("  - best_points.csv")
    print("  - *_pareto_MC_vs_NMSE_colored_by_S.png")
    print("  - *_alignment_minS_bestPerformance.png")
    print("  - *_overview_normalized.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="JSON result files")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--max-group-overviews", type=int, default=12)
    args = parser.parse_args()

    plot_all(args.files, outdir=args.outdir, max_group_overviews=args.max_group_overviews)


if __name__ == "__main__":
    main()
