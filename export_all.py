"""Collect every per-run result into one file for the label-free selection analysis.

    python export_all.py            -> all_runs.csv (one row per dataset x method x seed, with its source folder)
"""
import glob
import os

import pandas as pd

rows = []
for folder in sorted(glob.glob("results*")):
    if not os.path.isdir(folder):
        continue
    for f in glob.glob(os.path.join(folder, "*.csv")):
        if os.path.basename(f).startswith(("summary", "sweep")):
            continue
        try:
            d = pd.read_csv(f)
        except Exception as ex:
            print(f"skipped {f}: {ex}")
            continue
        d["folder"] = folder
        rows.append(d)
df = pd.concat(rows, ignore_index=True)
df.to_csv("all_runs.csv", index=False)
print(f"saved all_runs.csv: {len(df)} runs, {df.method.nunique()} method labels, {df.dataset.nunique()} datasets")
