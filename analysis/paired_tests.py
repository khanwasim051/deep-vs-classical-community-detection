"""Concern 4: rerun the primary analysis (graph-level ARI Friedman + paired/non-inferiority) on label-free selected
configurations, and separately on oracle (label-selected) and default configurations."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, OUT  # noqa: E402
os.chdir(OUT)  # every output of this script goes to analysis/out
import json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, studentized_range
R = pd.read_csv("selection.csv")
ATT = ["cora", "citeseer", "pubmed", "amazon-photo", "wiki", "blogcatalog", "coauthor-cs", "chameleon", "squirrel", "actor"]
DS = ["email-eu-core", "football", "dolphins", "ukfaculty", "kki", "cora", "citeseer", "pubmed", "amazon-photo",
      "wiki", "blogcatalog", "coauthor-cs", "chameleon", "squirrel", "actor",
      "lfr1", "lfr2", "lfr3", "lfr4", "lfr6", "lfr7"]
STR = [x for x in DS if x not in ATT]
DEEP = ["MS2CAG", "SSGC", "CCASSG", "DGI", "NS4GC", "DinkNet", "GAE", "DAEGC", "DMoN", "MixFlow-CD"]
M16 = DEEP + ["RegSpectral", "KMeans-feat", "Louvain", "Leiden", "Infomap", "LabelProp"]
DELTAS = [0.03, 0.05, 0.10]

def table(col):
    t = {}
    for r in R.itertuples():
        v = getattr(r, col)
        if not pd.isna(v):
            t.setdefault(r.dataset, {})[r.method] = float(v)
    return t

def analyse(ari, tag):
    rng = np.random.default_rng(0)
    def pair(m, ref, graphs, nboot=20000):
        gs = [g for g in graphs if m in ari[g] and ref in ari[g]]
        d = np.array([ari[g][m] - ari[g][ref] for g in gs])
        bm = d[rng.integers(0, len(d), (nboot, len(d)))].mean(1)
        try:
            p = wilcoxon(d, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        o = dict(n=len(d), mean=float(d.mean()), ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                 ci90=[float(np.percentile(bm, 5)), float(np.percentile(bm, 95))], upper95_one=float(np.percentile(bm, 95)),
                 lower95_one=float(np.percentile(bm, 5)), wins=int((d > 0.005).sum()), losses=int((d < -0.005).sum()), p=float(p))
        for dl in DELTAS:
            o[f"noninf_{dl}"] = o["upper95_one"] < dl
        return o
    def holm(ps):
        o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
        for k, i in enumerate(o):
            run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
        return adj
    res = {}
    for ref, strata in [("RegSpectral", {"all": DS, "structure": STR, "attributed": ATT, "homattr": [g for g in ATT if g not in ("chameleon", "squirrel", "actor")], "hetero": ["chameleon", "squirrel", "actor"]}), ("LeidenK", {"structure": STR})]:
        for sname, graphs in strata.items():
            rows = {m: pair(m, ref, graphs) for m in DEEP}
            for m, a in zip(DEEP, holm(np.array([rows[m]["p"] for m in DEEP]))):
                rows[m]["p_holm"] = float(a)
            res[f"{ref}|{sname}"] = rows
    comp = [x for x in DS if all(m in ari[x] for m in M16)]
    bl = pd.DataFrame([[ari[x][m] for m in M16] for x in comp], columns=M16, index=comp)
    N, k = bl.shape
    chi2, p = friedmanchisquare(*[bl[c] for c in M16])
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    ranks = bl.rank(axis=1, ascending=False).mean().sort_values()
    res["friedman_ari"] = dict(N=int(N), k=int(k), chi2=float(chi2), p=float(p), cd=float(cd), graphs=comp,
                               ranks={a: round(float(b), 3) for a, b in ranks.items()})
    res["mean_ari"] = {m: dict(all=float(np.mean([ari[x][m] for x in DS])), structure=float(np.mean([ari[x][m] for x in STR if m in ari[x]])),
                               attributed=float(np.mean([ari[x][m] for x in ATT if m in ari[x]]))) for m in M16 + ["LeidenK", "SpecAttr"]
                       if all(m in ari[x] for x in DS)}
    print(f"\n===== {tag}: Friedman N={N} k={k} chi2={chi2:.1f} p={p:.1e} CD={cd:.2f}")
    print("  ranks:", {a: round(b, 2) for a, b in ranks.items()})
    for key in ["RegSpectral|structure", "RegSpectral|attributed", "LeidenK|structure"]:
        print(" ", key)
        for m in DEEP:
            r = res[key][m]
            print(f"    {m:10s} n={r['n']:2d} mean={r['mean']:+.3f} CI95=[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] up95={r['upper95_one']:+.3f} lo95={r['lower95_one']:+.3f} W/L={r['wins']}/{r['losses']} pH={r['p_holm']:.3f} NI.03/.05/.10={r['noninf_0.03']},{r['noninf_0.05']},{r['noninf_0.1']}")
    return res

out = {"selected": analyse(table("ARI_sel"), "LABEL-FREE SELECTED"),
       "oracle": analyse(table("ARI_orc"), "ORACLE (label-selected)"),
       "default": analyse(table("ARI_def"), "DEFAULT (main paper)")}
json.dump(out, open("equiv_sel.json", "w"), indent=1)
