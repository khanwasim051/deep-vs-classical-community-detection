"""Show which method x dataset combinations have all their runs, and which are missing.

    python check_coverage.py --results results_pyagc
    python check_coverage.py --results results_pyagc --runs 5
"""
import argparse
import glob
import os

import pandas as pd

from data import ALL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results_pyagc")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--methods", default="", help="comma list of methods expected (default: all found)")
    a = p.parse_args()
    files = [f for f in glob.glob(os.path.join(a.results, "*.csv")) if not os.path.basename(f).startswith(("summary", "sweep"))]
    if not files:
        raise SystemExit(f"no result files in {a.results}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    methods = a.methods.split(",") if a.methods else sorted(df.method.unique())
    datasets = [d for d in ALL if d in set(df.dataset)] + sorted(set(df.dataset) - set(ALL))
    counts = df.groupby(["dataset", "method"]).seed.nunique().unstack(fill_value=0)
    counts = counts.reindex(index=datasets, columns=methods, fill_value=0)
    short = {m: m.replace("[PyAGC]", "") for m in methods}
    print("Runs saved per dataset (rows) and method (columns); '.' = complete, a number = runs saved so far\n")
    print(counts.rename(columns=short).map(lambda v: "." if v >= a.runs else str(v)).to_string())
    missing = [(d, m, int(counts.loc[d, m])) for d in datasets for m in methods if counts.loc[d, m] < a.runs]
    print(f"\n{len(missing)} of {len(datasets) * len(methods)} combinations incomplete")
    for d, m, c in missing:
        print(f"  {d:18s} {m:18s} {c}/{a.runs} runs")


if __name__ == "__main__":
    main()
