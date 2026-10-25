"""Runtime and peak GPU memory per method and dataset, from the time_s / peak_mem_mb columns already saved.

    python runtime_table.py > runtime.txt
Also writes runtime_table.csv (mean seconds per run and max peak GPU MB, per dataset x method).
"""
import glob
import os

import pandas as pd

FOLDERS = ["results", "results_pyagc"]
METHODS = ["MixFlow-CD_v2", "RegSpectral", "Louvain", "Leiden", "LabelProp", "KMeans-feat",
           "GAE[PyAGC]", "DGI[PyAGC]", "DAEGC[PyAGC]", "DMoN[PyAGC]", "DinkNet[PyAGC]",
           "SSGC[PyAGC]", "NS4GC[PyAGC]", "CCASSG[PyAGC]", "MS2CAG[PyAGC]", "Infomap"]
DATASETS = ["email-eu-core", "football", "dolphins", "ukfaculty", "kki", "cora", "citeseer", "pubmed",
            "amazon-photo", "lfr1", "lfr2", "lfr3", "lfr4", "lfr6", "lfr7"]

files = [f for d in FOLDERS for f in glob.glob(os.path.join(d, "*.csv"))
         if not os.path.basename(f).startswith(("summary", "sweep"))]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df[df.method.isin(METHODS) & df.dataset.isin(DATASETS)]
g = df.groupby(["dataset", "method"]).agg(time_s=("time_s", "mean"), peak_mem_mb=("peak_mem_mb", "max"),
                                         runs=("seed", "nunique")).reset_index()
g.round(1).to_csv("runtime_table.csv", index=False)
short = {m: m.replace("[PyAGC]", "").replace("_v2", "") for m in METHODS}
pd.set_option("display.width", 250)
for col, title in [("time_s", "Mean seconds per run"), ("peak_mem_mb", "Peak GPU memory (MB; 0 = CPU only)")]:
    t = g.pivot(index="dataset", columns="method", values=col).reindex(index=DATASETS, columns=METHODS)
    print(f"\n{title}\n" + t.rename(columns=short).round(1).to_string())
print("\nsaved runtime_table.csv")
