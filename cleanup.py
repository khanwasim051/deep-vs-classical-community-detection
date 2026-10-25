"""Remove result rows that must be re-run with the corrected code (run once)."""
import glob
import pandas as pd

ATTRIBUTED = ["amazon-computers", "amazon-photo", "citeseer", "cora", "pubmed"]
for f in glob.glob("results/*.csv"):
    if "summary" in f:
        continue
    df = pd.read_csv(f)
    drop = ((df.method == "MixFlow-CD_v2") & df.dataset.isin(ATTRIBUTED)) | \
           ((df.method == "DAEGC") & (df.dataset == "amazon-computers"))
    if drop.any():
        df[~drop].to_csv(f, index=False)
        print(f"{f}: removed {int(drop.sum())} rows")
print("done")
