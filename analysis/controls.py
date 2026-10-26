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

# Table 12 and the unknown-K rank test (17 methods, four metrics, 15 graphs), for each label-free estimator of K
EST = {"eigengap": ("@Khat", "MixFlow-CD_khat", "khat.json"), "bethe": ("@KhatBH", "MixFlow-CD_khatBH", "khat_bethe.json")}
BASE17 = ["RegSpectral", "KMeans-feat", "SpecAttr"] + [f"{m}[PyAGC]" for m in ["MS2CAG", "SSGC", "DGI", "CCASSG", "NS4GC", "DinkNet", "GAE", "DAEGC", "DMoN"]]
lab = lambda m, e: EST[e][1] if m == "MixFlow-CD" else m + EST[e][0]

def block(e):
    kh = json.load(open(os.path.join(ROOT, EST[e][2])))
    um = ["Louvain", "Leiden", "Infomap"] + [lab(m, e) for m in ["RegSpectral", "MS2CAG[PyAGC]", "CCASSG[PyAGC]", "NS4GC[PyAGC]", "MixFlow-CD"]]
    t3 = [dict(d=ds, K=int(K[ds]), Kh=int(kh[ds]), vals=[round(v(ds, m), 4) for m in um], orig=round(v(ds, "RegSpectral"), 4)) for ds in ORIG15]
    u17 = ["Louvain", "Leiden", "Infomap", "LabelProp"] + [lab(m, e) for m in BASE17 + ["MixFlow-CD"]]
    bl = pd.DataFrame([[v(ds, m, met) for m in u17] for met in ["ACC", "NMI", "ARI", "F1"] for ds in ORIG15], columns=u17)
    N, k = bl.shape
    chi2, p = friedmanchisquare(*[bl[c] for c in u17])
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    unk = dict(N=int(N), k=int(k), chi2=float(chi2), p=float(p), cd=float(cd),
               ranks={a: round(float(b), 3) for a, b in bl.rank(axis=1, ascending=False).mean().sort_values().items()})
    dg = {}
    for known in ["RegSpectral", "MS2CAG[PyAGC]", "CCASSG[PyAGC]", "NS4GC[PyAGC]", "DGI[PyAGC]", "MixFlow-CD"]:
        kn = "MixFlow-CD_v2" if known == "MixFlow-CD" else known
        gs = [ds for ds in ORIG15 if kh[ds] != K[ds] and v(ds, kn) is not None and v(ds, lab(known, e)) is not None]
        dg[kn] = dict(known=float(np.mean([v(ds, kn) for ds in gs])), khat=float(np.mean([v(ds, lab(known, e)) for ds in gs])), n=len(gs))
    rng = np.random.default_rng(0); pr = {}
    for m in ["CCASSG", "NS4GC", "MS2CAG", "SSGC", "DGI", "MixFlow-CD", "RegSpectral"]:
        lb = lab(m if m in ("MixFlow-CD", "RegSpectral") else f"{m}[PyAGC]", e)
        for ref in ["Leiden", "Louvain"]:
            x = np.array([v(ds, lb) - v(ds, ref) for ds in ORIG15])
            bm = x[rng.integers(0, len(x), (20000, len(x)))].mean(1)
            pr[f"{m}|{ref}"] = dict(mean=float(x.mean()), ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                                    up95=float(np.percentile(bm, 95)), wins=int((x > 0).sum()))
    mean15 = {m: float(np.mean([v(ds, lab(m, e)) for ds in ORIG15])) for m in BASE17 + ["MixFlow-CD"]}
    err = dict(exact=int(sum(kh[ds] == K[ds] for ds in ORIG15)), under=int(sum(kh[ds] < K[ds] for ds in ORIG15)),
               over=int(sum(kh[ds] > K[ds] for ds in ORIG15)),
               mape=float(np.mean([abs(kh[ds] - K[ds]) / K[ds] for ds in ORIG15])))
    return dict(T3=t3, UM=um, deg=dg, unknown=unk, unknown_pairs=pr, mean15=mean15, err=err, khat={ds: [int(K[ds]), int(kh[ds])] for ds in ORIG15})

B = {e: block(e) for e in EST}
khat = json.load(open(os.path.join(ROOT, "khat.json")))
T3, UM, deg, unknown, pairs = B["eigengap"]["T3"], B["eigengap"]["UM"], B["eigengap"]["deg"], B["eigengap"]["unknown"], B["eigengap"]["unknown_pairs"]
mk = {}
for m in BASE17 + ["MixFlow-CD"]:
    xs = [v(ds, "MixFlow-CD_v2" if m == "MixFlow-CD" else m) for ds in ORIG15]
    mk[m] = dict(mean=float(np.mean([x for x in xs if x is not None])), missing=[ds for ds, x in zip(ORIG15, xs) if x is None])
cls = {m: float(np.mean([v(ds, m) for ds in ORIG15])) for m in ["Louvain", "Leiden", "Infomap", "LabelProp", "LeidenK"] if all(v(ds, m) is not None for ds in ORIG15)}
json.dump(dict(T1=T1, T3=T3, UM=UM, deg=deg, unknown=unknown, unknown_pairs=pairs, khat={ds: [int(K[ds]), int(khat[ds])] for ds in ORIG15},
               est=B, known15=mk, classic15=cls),
          open(os.path.join(OUT, "controls.json"), "w"), indent=1)
print("controls.json written")
