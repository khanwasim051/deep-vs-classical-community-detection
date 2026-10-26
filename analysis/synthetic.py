"""Concern 5: synthetic graphs. 48-graph mu sweep (6 per mu) and 18 heterogeneous LFR graphs; every method with its
label-free configuration chosen on the six benchmark LFR graphs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, OUT  # noqa: E402
os.chdir(OUT)  # every output of this script goes to analysis/out
import ast, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
d = pd.read_csv(os.path.join(ROOT, "all_runs.csv"))
gs = {}
for f in ["graph_stats.txt", "graph_stats_base.txt"]:
    for ln in open(os.path.join(ROOT, f)):
        if ln.startswith("{"):
            r = ast.literal_eval(ln); gs[r["dataset"]] = r
LAB = {"MS2CAG": "MS2CAG[PyAGC-Photo]", "SSGC": "SSGC[PyAGC-HM]", "CCASSG": "CCASSG[PyAGC-Photo]", "DGI": "DGI[PyAGC]",
       "NS4GC": "NS4GC[PyAGC-HM]", "DinkNet": "DinkNet[PyAGC-Photo]", "GAE": "GAE[PyAGC-Photo]", "DAEGC": "DAEGC[PyAGC-Photo]",
       "DMoN": "DMoN[PyAGC-Photo]", "MixFlow-CD": "MixFlow-CD_cfgC", "RegSpectral": "RegSpectral_t0.25", "Louvain": "Louvain",
       "Leiden": "Leiden", "Infomap": "Infomap", "LabelProp": "LabelProp", "KMeans-feat": "KMeans-feat", "LeidenK": "LeidenK",
       "KMeans-featM": "KMeans-featM"}
DEEP = ["MS2CAG", "SSGC", "CCASSG", "DGI", "NS4GC", "DinkNet", "GAE", "DAEGC", "DMoN", "MixFlow-CD"]
MUS = [round(0.1 * i, 1) for i in range(1, 9)]
SW = [f"mu{m}g{g}" for m in MUS for g in range(1, 7)]
HET = [f"het_t{t}_c{c}_g{g}" for t in ("2", "2.5", "3") for c in ("1.1", "2") for g in (1, 2, 3)]
syn = d[d.dataset.isin(SW + HET)]
syn = syn[syn.folder.str.contains("sweep|syn")]
A = syn.groupby(["dataset", "method"]).ARI.mean()
T = syn.groupby(["dataset", "method"]).time_s.mean()
def ari(g, m): return float(A.get((g, LAB[m]), np.nan))
miss = [(g, m) for g in SW + HET for m in LAB if np.isnan(ari(g, m))]
print("missing", len(miss), miss[:10])
out = {"sweep": {}, "het": {}, "mix": {}}
for m in LAB:
    out["sweep"][m] = {str(mu): [float(np.mean([ari(f"mu{mu}g{g}", m) for g in range(1, 7)])),
                                  float(np.std([ari(f"mu{mu}g{g}", m) for g in range(1, 7)], ddof=1))] for mu in MUS}
    out["het"][m] = {f"{t}|{c}": float(np.mean([ari(f"het_t{t}_c{c}_g{g}", m) for g in (1, 2, 3)])) for t in ("2", "2.5", "3") for c in ("1.1", "2")}
    out["het"][m]["all"] = float(np.mean([ari(g, m) for g in HET]))
out["mix"] = {str(mu): float(np.mean([gs[f"mu{mu}g{g}"]["realized_mixing"] for g in range(1, 7)])) for mu in MUS}
out["het_mix"] = {f"{t}|{c}": float(np.mean([gs[f"het_t{t}_c{c}_g{g}"]["realized_mixing"] for g in (1, 2, 3)])) for t in ("2", "2.5", "3") for c in ("1.1", "2")}
out["het_K"] = {f"{t}|{c}": float(np.mean([gs[f"het_t{t}_c{c}_g{g}"]["K"] for g in (1, 2, 3)])) for t in ("2", "2.5", "3") for c in ("1.1", "2")}
rng = np.random.default_rng(0)
def pair(m, ref, graphs):
    x = np.array([ari(g, m) - ari(g, ref) for g in graphs])
    bm = x[rng.integers(0, len(x), (20000, len(x)))].mean(1)
    p = wilcoxon(x, zero_method="zsplit").pvalue if np.any(np.abs(x) > 1e-12) else 1.0
    return dict(n=len(x), mean=float(x.mean()), ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                up95=float(np.percentile(bm, 95)), wins=int((x > 0.005).sum()), losses=int((x < -0.005).sum()), p=float(p))
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj
SWI = [g for g in SW if not g.startswith(("mu0.1", "mu0.8"))]  # informative range (some method between 0.05 and 0.99)
for ref in ["LeidenK", "RegSpectral", "KMeans-featM"]:
    for name, graphs in [("sweep", SW), ("het", HET)]:
        rows = {m: pair(m, ref, graphs) for m in DEEP}
        for m, a in zip(DEEP, holm(np.array([rows[m]["p"] for m in DEEP]))):
            rows[m]["p_holm"] = float(a)
        out[f"{ref}|{name}"] = rows
# Issue 5: grouped design. Strata are the eight mu levels (six graphs each) and the six (tau1, tau2) cells (three graphs each).
from scipy.stats import rankdata, ttest_1samp
STRATA = {"sweep": [[f"mu{mu}g{g}" for g in range(1, 7)] for mu in MUS],
          "het": [[f"het_t{t}_c{c}_g{g}" for g in (1, 2, 3)] for t in ("2", "2.5", "3") for c in ("1.1", "2")]}
NPERM = 100000
def strat_stats(m, ref, strata):
    X0 = [np.array([ari(g, m) - ari(g, ref) for g in st]) for st in strata]
    X = X0
    allx = np.concatenate(X0)
    # stratified bootstrap: resample graphs within each stratum
    r2 = np.random.default_rng(1)
    # within-stratum resampling with the small-sample variance correction (Davison and Hinkley, 1997, Sec. 3.7)
    X = [x.mean() + (x - x.mean()) * np.sqrt(len(x) / (len(x) - 1)) for x in X]
    idx = [r2.integers(0, len(x), (20000, len(x))) for x in X]
    tot = sum(len(x) for x in X)
    boots = sum(x[i].sum(1) for x, i in zip(X, idx)) / tot
    # stratified signed-rank test: ranks of |d| within each stratum, Monte Carlo sign flips of every graph
    X = X0
    R = [rankdata(np.abs(x)) * (np.abs(x) > 1e-12) for x in X]
    S = [np.sign(x) for x in X]
    obs = sum((r * s).sum() for r, s in zip(R, S))
    r3 = np.random.default_rng(2)
    null = sum((r[None, :] * r3.choice([-1.0, 1.0], (NPERM, len(r)))).sum(1) for r in R)
    p_strat = float((np.sum(np.abs(null) >= abs(obs) - 1e-9) + 1) / (NPERM + 1))
    # level-aggregated sensitivity: one mean difference per stratum, exact sign-flip test over strata
    lv = np.array([x.mean() for x in X]); k = len(lv)
    p_lev = float(ttest_1samp(lv, 0.0).pvalue) if np.ptp(lv) > 0 else 1.0
    nz = allx[np.abs(allx) > 1e-12]; rk = rankdata(np.abs(nz))
    rbc = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()) if len(nz) else 0.0
    return dict(ci95_strat=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], p_strat=p_strat,
                levels=[float(v) for v in lv], lev_neg=int((lv < -0.005).sum()), lev_pos=int((lv > 0.005).sum()), p_lev=p_lev,
                rbc=rbc, dz=float(allx.mean() / allx.std(ddof=1)) if allx.std(ddof=1) > 0 else 0.0,
                ties=int(len(allx) - (allx > 0.005).sum() - (allx < -0.005).sum()))
for ref in ["LeidenK", "RegSpectral", "KMeans-featM"]:
    for name in ["sweep", "het"]:
        rows = out[f"{ref}|{name}"]
        for m in DEEP:
            rows[m].update(strat_stats(m, ref, STRATA[name]))
        for key in ["p_strat", "p_lev"]:
            for m, a in zip(DEEP, holm(np.array([rows[m][key] for m in DEEP]))):
                rows[m][key + "_holm"] = float(a)
out["per_graph"] = {g: {m: ari(g, m) for m in LAB} for g in SW + HET}
out["gstats"] = {g: dict(edges=gs[g]["edges"], K=gs[g]["K"], mix=gs[g]["realized_mixing"]) for g in SW + HET}
# Table 14: the first three graphs per mu, on which both published configurations and our earlier re-implementation ran
T14L = {"KMeans-feat": "KMeans-feat", "KMeans-featM": "KMeans-featM", "GAE": "GAE[PyAGC]", "GAE (Photo)": "GAE[PyAGC-Photo]",
        "DAEGC": "DAEGC[PyAGC]", "DAEGC (Photo)": "DAEGC[PyAGC-Photo]", "DAEGC (own)": "DAEGC"}
out["T14"] = {k: {str(mu): float(np.mean([A.get((f"mu{mu}g{g}", lab), np.nan) for g in (1, 2, 3)])) for mu in MUS} for k, lab in T14L.items()}
json.dump(out, open("syn.json", "w"), indent=1)
show = ["LeidenK", "RegSpectral", "Infomap", "Louvain", "KMeans-featM"] + DEEP
print("mu  mix  " + " ".join(f"{m[:7]:>7s}" for m in show))
for mu in MUS:
    print(f"{mu}  {out['mix'][str(mu)]:.2f} " + " ".join(f"{out['sweep'][m][str(mu)][0]:7.3f}" for m in show))
print("HET")
for k in out["het_mix"]:
    print(k, round(out["het_mix"][k], 3), round(out["het_K"][k]), " ".join(f"{out['het'][m][k]:7.3f}" for m in show))
print("all", " ".join(f"{out['het'][m]['all']:7.3f}" for m in show))
for key in ["LeidenK|sweep", "LeidenK|het", "RegSpectral|sweep", "RegSpectral|het", "KMeans-featM|sweep", "KMeans-featM|het"]:
    print(key, {m: (round(r["mean"], 3), [round(c, 3) for c in r["ci95"]], round(r["p_holm"], 4), f"{r['wins']}/{r['losses']}") for m, r in out[key].items()})
for key in ["LeidenK|sweep", "LeidenK|het", "KMeans-featM|sweep", "KMeans-featM|het"]:
    print("ISSUE5", key)
    for m, r in out[key].items():
        print(f"  {m:10s} {r['mean']:+.3f} pooledCI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] stratCI[{r['ci95_strat'][0]:+.3f},{r['ci95_strat'][1]:+.3f}] "
              f"W/L/T {r['wins']}/{r['losses']}/{r['ties']} pH {r['p_holm']:.1e} pStratH {r['p_strat_holm']:.1e} lev -{r['lev_neg']}/+{r['lev_pos']} pLevH {r['p_lev_holm']:.3f} rbc {r['rbc']:+.2f} dz {r['dz']:+.2f}")
