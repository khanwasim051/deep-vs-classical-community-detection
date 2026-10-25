"""Tables 4 (reproduction check), 16 (plain vs regularized spectral clustering), 17 (random-label NMI) and
19 (change in ARI, Amazon-Photo minus Cora configuration)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi
from common import OUT, PY, PUBLISHED_ACC, runs

d = runs()
M = d[~d.folder.str.contains("sweep|syn|tune|repro")].groupby(["dataset", "method"])[["ACC", "NMI", "ARI"]].agg(["mean", "std"])
MR = d[d.folder.str.contains("repro")].groupby(["dataset", "method"])[["ACC", "NMI", "ARI"]].agg(["mean", "std", "count"])
def gr(ds, m, met="ACC", s="mean"):
    return float(MR.loc[(ds, m), (met, s)]) if (ds, m) in MR.index else None
NAT = None
_np = os.path.join(os.path.dirname(OUT), "..", "results_native.csv")
if os.path.exists(_np):
    NAT = pd.read_csv(_np)
def g(ds, m, met="ACC", s="mean"):
    return float(M.loc[(ds, m), (met, s)]) if (ds, m) in M.index else None
# Table 4: our input (unit-length rows) and PyAGC's own input (rows sum to one, results_repro), published configuration
T4 = []
for ds in ["cora", "amazon-photo"]:
    for m in PY:
        ours = g(ds, f"{m}[PyAGC]")
        nat = None
        if NAT is not None:
            x = NAT[(NAT.dataset.str.lower() == ("cora" if ds == "cora" else "photo")) & (NAT.method == m)]
            if len(x):
                nat = dict(mean=float(x.ACC.mean()), lo=float(x.ACC.min()), hi=float(x.ACC.max()), sd_within=float(x.ACC_sd.mean()), n=len(x))
        T4.append(dict(d=ds, m=m, ours=ours, ours_sd=g(ds, f"{m}[PyAGC]", s="std"),
                       ours10=gr(ds, f"{m}[PyAGC]"), ours10_sd=gr(ds, f"{m}[PyAGC]", s="std"),
                       l1=gr(ds, f"{m}[PyAGC-l1]"), l1_sd=gr(ds, f"{m}[PyAGC-l1]", s="std"),
                       l1_n=None if gr(ds, f"{m}[PyAGC-l1]") is None else int(MR.loc[(ds, f"{m}[PyAGC-l1]"), ("ACC", "count")]),
                       native=nat, published=PUBLISHED_ACC[ds][m]))
# Table 16
T16 = [dict(d=ds, nmi_plain=g(ds, "Spectral", "NMI"), nmi_reg=g(ds, "RegSpectral", "NMI"), ari_plain=g(ds, "Spectral", "ARI"),
            ari_reg=g(ds, "RegSpectral", "ARI")) for ds in ["cora", "citeseer", "pubmed", "amazon-photo", "kki"]]
# Table 17: NMI between two independent uniform random labellings with K groups on n nodes (mean of five draws)
r = np.random.default_rng(0)
T17 = [dict(graph=nm_, n=n, K=K, nmi=float(np.mean([nmi(r.integers(0, K, n), r.integers(0, K, n)) for _ in range(5)])))
       for nm_, n, K in [("Email-Eu-core", 1005, 42), ("KKI", 2238, 83), ("LFR-1 / LFR-4", 2048, 23), ("LFR-2, -3, -6, -7", 5000, 129),
                         ("Sweep graphs", 5000, 133), ("Cora", 2708, 7), ("Football", 115, 12)]]
# Table 19
T19 = {ds: {m: (None if g(ds, f"{m}[PyAGC-Photo]", "ARI") is None or g(ds, f"{m}[PyAGC]", "ARI") is None
                else round(g(ds, f"{m}[PyAGC-Photo]", "ARI") - g(ds, f"{m}[PyAGC]", "ARI"), 3)) for m in PY}
       for ds in ["cora", "citeseer", "email-eu-core", "football", "kki", "lfr1", "lfr2", "lfr6"]}
T19raw = {ds: {m: [g(ds, f"{m}[PyAGC]", "ARI"), g(ds, f"{m}[PyAGC-Photo]", "ARI")] for m in PY} for ds in T19}
json.dump(dict(T4=T4, T16=T16, T17=T17, T19=T19, T19raw=T19raw), open(os.path.join(OUT, "misc_tables.json"), "w"), indent=1)
print("misc_tables.json written")
