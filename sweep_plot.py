"""Summarize and plot the mixing-parameter sweep.

    python sweep_plot.py --results results_sweep
    python sweep_plot.py --results results_sweep,results_sweep_pyagc --set reference
Writes sweep_table.csv (mean and std over the 3 graphs per mu, for every method and metric)
and sweep_ARI.png / sweep_NMI.png / sweep_ACC.png (one figure per metric) into the results folder.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# fixed order, never cycled: colour follows the method, not its rank
SHOWN = [("MixFlow-CD", "#2a78d6", "o"), ("RegSpectral", "#eb6834", "s"), ("KMeans-feat", "#1baf7a", "^"),
         ("GAE", "#eda100", "D"), ("DAEGC", "#e87ba4", "v"), ("Louvain", "#008300", "P")]
# with the PyAGC reference implementations: same colours for the three anchors, deep models from PyAGC
SHOWN_REF = [("MixFlow-CD", "#2a78d6", "o"), ("RegSpectral", "#eb6834", "s"), ("KMeans-feat", "#1baf7a", "^"),
             ("GAE[PyAGC]", "#eda100", "D"), ("NS4GC[PyAGC]", "#e87ba4", "v"), ("MS2CAG[PyAGC]", "#008300", "P")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results_sweep", help="folder, or comma list of folders (outputs go to the first)")
    p.add_argument("--set", default="default", choices=["default", "reference"], help="which methods to draw")
    a = p.parse_args()
    folders = a.results.split(",")
    files = [f for d in folders for f in glob.glob(os.path.join(d, "mu*.csv"))]
    if not files:
        raise SystemExit(f"no mu*.csv files in {a.results}")
    a.results = folders[0]
    shown = SHOWN_REF if a.set == "reference" else SHOWN
    suffix = "_ref" if a.set == "reference" else ""
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["method"] = df["method"].replace({"MixFlow-CD_v2": "MixFlow-CD"})
    df["mu"] = df["dataset"].map(lambda s: float(re.match(r"mu([\d.]+)g", s).group(1)))
    df["graph"] = df["dataset"].map(lambda s: int(re.search(r"g(\d+)$", s).group(1)))
    # average seeds within a graph first, then mean/std across graphs
    per_graph = df.groupby(["method", "mu", "graph"])[["ACC", "NMI", "ARI", "F1"]].mean().reset_index()
    agg = per_graph.groupby(["method", "mu"]).agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.drop(columns=[c for c in agg.columns if c.startswith("graph")]).reset_index()
    out = os.path.join(a.results, f"sweep_table{suffix}.csv")
    agg.round(4).to_csv(out, index=False)
    print(f"saved {out}")
    wide = agg.pivot(index="mu", columns="method", values="NMI_mean").round(3)
    print("\nNMI (mean over graphs) by mu:\n" + wide.to_string())

    # NMI of a random labelling with the same n and K: NMI is not chance-corrected and is inflated when K is large
    import numpy as np
    from sklearn.metrics import normalized_mutual_info_score
    rng = np.random.default_rng(0)
    K = int(df["K_true"].median())
    rand_nmi = float(np.mean([normalized_mutual_info_score(rng.integers(0, K, 5000), rng.integers(0, K, 5000))
                              for _ in range(5)]))
    print(f"\nNMI of random labels (n = 5000, K = {K}): {rand_nmi:.3f}  -> use ARI as the chance-corrected metric")
    wide = agg.pivot(index="mu", columns="method", values="ARI_mean").round(3)
    print("\nARI (mean over graphs) by mu:\n" + wide.to_string())

    for metric in ["ARI", "NMI", "ACC"]:
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
        for name, color, marker in shown:
            s = agg[agg.method == name].sort_values("mu")
            if s.empty:
                continue
            m, sd = s[f"{metric}_mean"], s[f"{metric}_std"].fillna(0)
            ax.fill_between(s["mu"], m - sd, m + sd, color=color, alpha=0.12, lw=0)
            ax.plot(s["mu"], m, color=color, lw=2, marker=marker, ms=6, mec="white", mew=1.2, label=name.replace("[PyAGC]", ""))
        if metric == "NMI":
            ax.axhline(rand_nmi, color=MUTED, lw=1, ls="--")
            ax.text(0.1, rand_nmi + 0.015, f"random labels ({rand_nmi:.2f})", color=MUTED, fontsize=7.5)
        ax.set_xlabel("Mixing parameter μ (fraction of edges leaving a node's community)", color=INK, fontsize=9)
        ax.set_ylabel(metric, color=INK, fontsize=9)
        ax.set_ylim(-0.02 if metric == "ARI" else 0, 1.02)
        ax.set_xticks(sorted(agg["mu"].unique()))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, labelcolor=INK)
        ax.set_title(f"{metric} on LFR graphs (n = 5,000) as communities get noisier",
                     loc="left", fontsize=10, color=INK)
        fig.tight_layout()
        path = os.path.join(a.results, f"sweep_{metric}{suffix}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
