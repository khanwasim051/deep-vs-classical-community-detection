"""Section 5.6 controls with one fixed configuration per method (Cora; Amazon-Photo on Amazon-Photo):
Table 11 (attributed graphs grouped by input), Table 12 (unknown K) and the unknown-K paired comparisons."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, studentized_range
from common import ROOT, OUT, ATT_ORIG, ORIG15, runs

d = runs()
d = d[~d.folder.str.contains("sweep|syn|tune|repro")]
M = d.groupby(["dataset", "method"])[["ACC", "NMI", "ARI", "F1"]].mean()
v = lambda ds, m, met="ARI": float(M.loc[(ds, m), met]) if (ds, m) in M.index else None
khat = json.load(open(os.path.join(ROOT, "khat.json")))
K = d.groupby("dataset").K_true.first()

# Table 11
GROUPS = [("Topology only", [("RegSpectral", "RegSpectral"), ("Leiden (K given)", "LeidenK"), ("Leiden", "Leiden"), ("Louvain", "Louvain"),
                             ("Infomap", "Infomap")] + [(f"{n}, structural input", f"{m}[PyAGC-struct]") for m, n in
                             [("MS2CAG", "MS2CAG"), ("SSGC", "SSGC"), ("NS4GC", "NS4GC"), ("CCASSG", "CCA-SSG"), ("DGI", "DGI"), ("GAE", "GAE")]]),
          ("Attributes only", [("k-means on attributes", "KMeans-feat")]),
          ("Topology and attributes, no neural training", [("SpecAttr", "SpecAttr"), ("SSGC", "SSGC[PyAGC]"), ("MS2CAG", "MS2CAG[PyAGC]")]),
          ("Topology and attributes, trained", [(n, f"{m}[PyAGC]") for m, n in [("NS4GC", "NS4GC"), ("DGI", "DGI"), ("CCASSG", "CCA-SSG")]]
           + [("MixFlow-CD", "MixFlow-CD_v2")] + [(m, f"{m}[PyAGC]") for m in ["GAE", "DinkNet", "DMoN", "DAEGC"]])]
T1 = []
for g, rows in GROUPS:
    T1.append({"group": g})
    for lab, m in rows:
        vals = [v(ds, m) for ds in ATT_ORIG]
        ok = [x for x in vals if x is not None]
        T1.append(dict(label=lab, vals=[None if x is None else round(x, 4) for x in vals], mean=float(np.mean(ok)), n=len(ok)))

# Table 12 and the unknown-K rank test (17 methods, four metrics, 15 graphs)
UM = ["Louvain", "Leiden", "Infomap", "RegSpectral@Khat", "MS2CAG[PyAGC]@Khat", "CCASSG[PyAGC]@Khat", "NS4GC[PyAGC]@Khat", "MixFlow-CD_khat"]
T3 = [dict(d=ds, K=int(K[ds]), Kh=int(khat[ds]), vals=[round(v(ds, m), 4) for m in UM], orig=round(v(ds, "RegSpectral"), 4)) for ds in ORIG15]
U17 = ["Louvain", "Leiden", "Infomap", "LabelProp", "RegSpectral@Khat", "KMeans-feat@Khat", "SpecAttr@Khat", "MixFlow-CD_khat"] + \
      [f"{m}[PyAGC]@Khat" for m in ["MS2CAG", "SSGC", "DGI", "CCASSG", "NS4GC", "DinkNet", "GAE", "DAEGC", "DMoN"]]
bl = pd.DataFrame([[v(ds, m, met) for m in U17] for met in ["ACC", "NMI", "ARI", "F1"] for ds in ORIG15], columns=U17)
N, k = bl.shape
chi2, p = friedmanchisquare(*[bl[c] for c in U17])
cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
unknown = dict(N=int(N), k=int(k), chi2=float(chi2), p=float(p), cd=float(cd),
               ranks={a: round(float(b), 3) for a, b in bl.rank(axis=1, ascending=False).mean().sort_values().items()})
deg = {}
for known, unk in [("RegSpectral", "RegSpectral@Khat"), ("MS2CAG[PyAGC]", "MS2CAG[PyAGC]@Khat"), ("CCASSG[PyAGC]", "CCASSG[PyAGC]@Khat"),
                   ("NS4GC[PyAGC]", "NS4GC[PyAGC]@Khat"), ("DGI[PyAGC]", "DGI[PyAGC]@Khat"), ("MixFlow-CD_v2", "MixFlow-CD_khat")]:
    gs = [ds for ds in ORIG15 if khat[ds] != K[ds] and v(ds, known) is not None]
    deg[known] = dict(known=float(np.mean([v(ds, known) for ds in gs])), khat=float(np.mean([v(ds, unk) for ds in gs])))
rng = np.random.default_rng(0); pairs = {}
for m in ["CCASSG", "NS4GC", "MS2CAG", "SSGC", "DGI", "MixFlow-CD"]:
    lab = "MixFlow-CD_khat" if m == "MixFlow-CD" else f"{m}[PyAGC]@Khat"
    for ref in ["Leiden", "Louvain"]:
        x = np.array([v(ds, lab) - v(ds, ref) for ds in ORIG15])
        bm = x[rng.integers(0, len(x), (20000, len(x)))].mean(1)
        pairs[f"{m}|{ref}"] = dict(mean=float(x.mean()), ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], up95=float(np.percentile(bm, 95)))
json.dump(dict(T1=T1, T3=T3, UM=UM, deg=deg, unknown=unknown, unknown_pairs=pairs, khat={ds: [int(K[ds]), int(khat[ds])] for ds in ORIG15}),
          open(os.path.join(OUT, "controls.json"), "w"), indent=1)
print("controls.json written")
