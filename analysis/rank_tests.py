"""sel.json for the manuscript (21 graphs, label-free selected configurations)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, OUT  # noqa: E402
os.chdir(OUT)  # every output of this script goes to analysis/out
import json
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, studentized_range
R = pd.read_csv("selection.csv"); E = json.load(open("equiv_sel.json")); runs = pd.read_csv(os.path.join(ROOT, "all_runs.csv"))
DS = ["email-eu-core", "football", "dolphins", "ukfaculty", "kki", "cora", "citeseer", "pubmed", "amazon-photo",
      "wiki", "blogcatalog", "coauthor-cs", "chameleon", "squirrel", "actor", "lfr1", "lfr2", "lfr3", "lfr4", "lfr6", "lfr7"]
ATT = ["cora", "citeseer", "pubmed", "amazon-photo", "wiki", "blogcatalog", "coauthor-cs", "chameleon", "squirrel", "actor"]
HETERO = ["chameleon", "squirrel", "actor"]; STR = [d for d in DS if d not in ATT]; SAT = ["kki", "lfr3", "lfr4"]
ORIG = [d for d in DS if d not in ["wiki", "blogcatalog", "coauthor-cs", "chameleon", "squirrel", "actor"]]
DEEP = ["MS2CAG", "SSGC", "CCASSG", "DGI", "NS4GC", "DinkNet", "GAE", "DAEGC", "DMoN", "MixFlow-CD"]
TRAINED = ["DGI", "CCASSG", "NS4GC", "GAE", "DAEGC", "DinkNet", "DMoN"]
M16 = DEEP + ["RegSpectral", "KMeans-feat", "Louvain", "Leiden", "Infomap", "LabelProp"]
TUNED = DEEP + ["RegSpectral", "Louvain", "Leiden"]
Rx = R.set_index(["dataset", "method"])
ok = lambda d, m: (d, m) in Rx.index and not pd.isna(Rx.loc[(d, m), "ARI_sel"])
out = {"tables": {}}
for met, c, s in [("ARI", "ARI_sel", "sd_sel"), ("ACC", "ACC_sel", "ACCsd_sel"), ("NMI", "NMI_sel", "NMIsd_sel"), ("F1", "F1_sel", "F1sd_sel")]:
    out["tables"][met] = {d: {m: [round(float(Rx.loc[(d, m), c]), 4), round(float(Rx.loc[(d, m), s]), 4)] for m in M16 + ["LeidenK", "SpecAttr"] if ok(d, m)} for d in DS}
COMP = [d for d in DS if all(m in out["tables"]["ARI"][d] for m in M16)]
def fried(blocks):
    N, k = blocks.shape
    chi2, p = friedmanchisquare(*[blocks[c].values for c in blocks.columns])
    F = (N - 1) * chi2 / (N * (k - 1) - chi2)
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    r = blocks.rank(axis=1, ascending=False).mean().sort_values()
    return dict(N=int(N), k=int(k), chi2=float(chi2), p=float(p), F=float(F), cd=float(cd), ranks={a: round(float(v), 3) for a, v in r.items()})
def blocks(graphs, metrics=("ARI",), methods=M16, src=None, fill=False):
    src = src or out["tables"]
    rows = []
    for met in metrics:
        for d in graphs:
            if all(m in src[met][d] for m in methods) or fill:
                rows.append([src[met][d][m][0] if m in src[met][d] else -1.0 for m in methods])
    return pd.DataFrame(rows, columns=methods)
out["ranks_by_metric"] = {met: fried(blocks(COMP, (met,)))["ranks"] for met in ["ARI", "ACC", "NMI", "F1"]}
out["FA"] = fried(blocks(COMP)); out["COMP"] = COMP
out["ari_struct"] = fried(blocks(STR)); out["ari_attr"] = fried(blocks(ATT))
def tbl(col):
    return {"ARI": {d: {m: [float(Rx.loc[(d, m), col])] for m in M16 if (d, m) in Rx.index and not pd.isna(Rx.loc[(d, m), col])} for d in DS}}
dflt, orc = tbl("ARI_def"), tbl("ARI_orc")
nb = lambda graphs, src=None: len(blocks(graphs, src=src))
out["rob"] = [
    [f"Label-free selection, {len(COMP)} complete graphs (primary)", out["FA"]],
    [f"All four metrics as blocks, {len(COMP)} graphs", fried(blocks(COMP, ("ARI", "ACC", "NMI", "F1")))],
    ["All 21 graphs, missing result ranked last", fried(blocks(DS, fill=True))],
    [f"Without heterophilous graphs ({nb([d for d in DS if d not in HETERO])})", fried(blocks([d for d in DS if d not in HETERO]))],
    [f"Without saturated graphs ({nb([d for d in DS if d not in SAT])})", fried(blocks([d for d in DS if d not in SAT]))],
    [f"Original 15 graphs", fried(blocks(ORIG))],
    [f"Structure-only graphs ({nb(STR)})", out["ari_struct"]],
    [f"Attributed graphs ({nb(ATT)})", out["ari_attr"]],
    [f"Homophilous attributed graphs ({nb([d for d in ATT if d not in HETERO])})", fried(blocks([d for d in ATT if d not in HETERO]))],
    [f"Fixed published configuration ({nb(DS, dflt)} graphs)", fried(blocks(DS, src=dflt))],
    [f"Oracle: label-selected configuration ({nb(DS, orc)} graphs)", fried(blocks(DS, src=orc))],
    [f"Known K for all: Leiden-K and SpecAttr added ({len(blocks(COMP, methods=M16 + ['LeidenK', 'SpecAttr']))} graphs)", fried(blocks(COMP, methods=M16 + ["LeidenK", "SpecAttr"]))],
]
out["EQ"] = E["selected"]; out["EQ_oracle"] = E["oracle"]
sds = [out["tables"]["ARI"][d][m][1] for d in COMP for m in M16]
out["seed_sd_median"] = float(np.nanmedian(sds)); out["seed_sd_p90"] = float(np.nanpercentile(sds, 90))
# cost and budget
cost = {}
for m in M16:
    x = R[(R.method == m) & R.dataset.isin(COMP)]
    xa = R[(R.method == m) & R.dataset.isin(DS)]
    fails = int(sum(len(str(v).split(",")) for v in xa.missing if isinstance(v, str) and v))
    fails += int(4 * ((xa.n_cand == 0).sum()))
    over6 = int((xa.mem_sel > 6144).sum())
    ncfg = int(xa.n_cand.max())
    cost[m] = dict(total=float(x.t_sel.sum()), median=float(x.t_sel.median()), mem=float(xa.mem_sel.max()), search=float(x.t_search.sum()),
                   ncand=ncfg, fails=fails, graphs_ok=int((xa.n_cand > 0).sum()), mean_ari=float(x.ARI_sel.mean()), rank=out["FA"]["ranks"][m], over6=over6)
out["cost"] = cost
summ = {}
for m in TUNED:
    x = R[R.method == m].dropna(subset=["ARI_sel"]); both = x.dropna(subset=["ARI_def"])
    summ[m] = dict(n=len(x), sel_def=int((x.selected == x.default).sum()), sel_orc=int((x.selected == x.oracle).sum()),
                   ari_def=float(both.ARI_def.mean()), ari_sel_c=float(both.ARI_sel.mean()), ari_orc_c=float(both.ARI_orc.mean()),
                   n_def=len(both), regret=float((x.ARI_orc - x.ARI_sel).mean()), max_regret=float((x.ARI_orc - x.ARI_sel).max()),
                   gain=float((both.ARI_sel - both.ARI_def).mean()), rho=float(x.rho.median()))
out["selection"] = summ
km = runs[(runs.method == "KMeans-featM") & ~runs.folder.str.contains("sweep|syn")].groupby("dataset").ARI.mean()
T2 = []
for d in STR:
    a = out["tables"]["ARI"][d]; bt = max(TRAINED, key=lambda m: a[m][0])
    T2.append(dict(d=d, vals=[a["Leiden"][0], a["LeidenK"][0], a["RegSpectral"][0], a["KMeans-feat"][0], float(km[d]), a["MS2CAG"][0], a[bt][0]], bt="CCA-SSG" if bt == "CCASSG" else bt))
out["T2"] = T2
out["per_graph"] = {d: {m: dict(selected=Rx.loc[(d, m), "selected"], oracle=Rx.loc[(d, m), "oracle"], ari_orc=float(Rx.loc[(d, m), "ARI_orc"]),
                              t=float(Rx.loc[(d, m), "t_sel"]), mem=float(Rx.loc[(d, m), "mem_sel"]), missing=Rx.loc[(d, m), "missing"] if isinstance(Rx.loc[(d, m), "missing"], str) else "")
                      for m in TUNED + ["KMeans-feat", "Infomap", "LabelProp"] if ok(d, m)} for d in DS}
json.dump(out, open("sel.json", "w"), indent=1, default=float)
for lab, fr in out["rob"]:
    print(f"{lab:62s} N={fr['N']:2d} CD={fr['cd']:.2f} top3={list(fr['ranks'].items())[:3]} RegSpec={fr['ranks']['RegSpectral']} ({list(fr['ranks']).index('RegSpectral')+1})")
for m, v in summ.items(): print(m, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
for m, c in cost.items(): print(m, {k: round(v, 3) for k, v in c.items()})
print(out["seed_sd_median"], out["seed_sd_p90"], COMP)
