"""Summary tables + Friedman / Iman-Davenport / Nemenyi tests over all results/*.csv.

    python stats.py --results results --methods "MixFlow-CD,Louvain,Leiden,Spectral,KMeans-feat"
Writes results/summary_table.csv (mean±std) and prints average ranks and the critical difference.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range, f as fdist

METRICS = ["ACC", "NMI", "ARI", "F1"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--methods", default="", help="comma list to include (default: all found)")
    p.add_argument("--alpha", type=float, default=0.05)
    a = p.parse_args()
    folders = [d.strip() for d in a.results.split(",")]  # several result folders can be combined
    files = [f for d in folders for f in glob.glob(os.path.join(d, "*.csv"))
             if not os.path.basename(f).startswith(("summary", "sweep"))]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if a.methods:
        df = df[df.method.isin(a.methods.split(","))]
    agg = df.groupby(["dataset", "method"])[METRICS].agg(["mean", "std", "count"])
    table = pd.DataFrame(index=agg.index)
    for m in METRICS:
        table[m] = [f"{mu:.4f}±{sd if sd == sd else 0:.4f}" for mu, sd in zip(agg[(m, "mean")], agg[(m, "std")])]
    table["runs"] = agg[("ACC", "count")]
    out = os.path.join(folders[0], "summary_table.csv")
    table.to_csv(out)
    print(table.to_string())
    print(f"\nsaved {out}")

    means = df.groupby(["dataset", "method"])[METRICS].mean().reset_index()
    wide = means.pivot(index="dataset", columns="method", values=METRICS)  # tasks = dataset x metric
    tasks = []
    for m in METRICS:
        tasks.append(wide[m])
    T = pd.concat(tasks).dropna()
    methods = list(T.columns)
    N, k = T.shape
    if k < 3 or N < 2:
        print("need >=3 methods and >=2 complete tasks for the Friedman test")
        return
    ranks = T.rank(axis=1, ascending=False).mean()
    chi2, pval = friedmanchisquare(*[T[c].values for c in methods])
    Ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
    pF = 1 - fdist.cdf(Ff, k - 1, (k - 1) * (N - 1))
    q = studentized_range.ppf(1 - a.alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    print(f"\nFriedman over N={N} tasks (dataset x metric), k={k} methods")
    print(f"  chi2 = {chi2:.3f} (p = {pval:.2e});  Iman-Davenport F = {Ff:.3f}, df=({k - 1},{(k - 1) * (N - 1)}), p = {pF:.2e}")
    print(f"  Nemenyi critical difference (alpha={a.alpha}): {cd:.3f}")
    print("  average ranks:")
    for mth, r in ranks.sort_values().items():
        print(f"    {mth:28s} {r:.3f}")


if __name__ == "__main__":
    main()
