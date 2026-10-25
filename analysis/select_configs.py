"""Concern 4: label-free configuration selection (max mean modularity over 5 seeds) vs oracle (max mean ARI) vs default."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, OUT  # noqa: E402
os.chdir(OUT)  # every output of this script goes to analysis/out
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
d = pd.read_csv(os.path.join(ROOT, "all_runs.csv"))
DS = ["email-eu-core", "football", "dolphins", "ukfaculty", "kki", "cora", "citeseer", "pubmed", "amazon-photo",
      "wiki", "blogcatalog", "coauthor-cs", "chameleon", "squirrel", "actor",
      "lfr1", "lfr2", "lfr3", "lfr4", "lfr6", "lfr7"]
d = d[d.dataset.isin(DS) & ~d.folder.str.contains("sweep|khat|tune|syn|repro")]
g = d.groupby(["dataset", "method"]).agg(n=("ARI", "size"), ARI=("ARI", "mean"), ARIsd=("ARI", "std"), Q=("Q", "mean"),
                                          ACC=("ACC", "mean"), ACCsd=("ACC", "std"), NMI=("NMI", "mean"), NMIsd=("NMI", "std"),
                                          F1=("F1", "mean"), F1sd=("F1", "std"), t=("time_s", "mean"), mem=("peak_mem_mb", "max")).reset_index()
G = {(r.dataset, r.method): r for r in g.itertuples()}
PY = ["MS2CAG", "SSGC", "CCASSG", "DGI", "NS4GC", "DinkNet", "GAE", "DAEGC", "DMoN"]

def cands(m, ds):
    if m in PY:
        cora = f"{m}[PyAGC-Cora]" if ds == "amazon-photo" or (ds, f"{m}[PyAGC]") not in G else f"{m}[PyAGC]"
        photo = f"{m}[PyAGC]" if ds == "amazon-photo" else f"{m}[PyAGC-Photo]"
        c = {"Cora": cora, "Photo": photo, "Physics": f"{m}[PyAGC-Physics]", "HM": f"{m}[PyAGC-HM]"}
        default = "Photo" if ds == "amazon-photo" else "Cora"
    elif m == "MixFlow-CD":
        c = {"default": "MixFlow-CD_v2", "mask0.3": "MixFlow-CD_cfgB", "lamQ2": "MixFlow-CD_cfgC", "ep100": "MixFlow-CD_cfgD"}; default = "default"
    elif m == "RegSpectral":
        c = {"tau1": "RegSpectral", "tau0.25": "RegSpectral_t0.25", "tau0.5": "RegSpectral_t0.5", "tau2": "RegSpectral_t2"}; default = "tau1"
    elif m in ("Louvain", "Leiden"):
        c = {"r1": m, "r0.5": f"{m}_r0.5", "r2": f"{m}_r2", "r4": f"{m}_r4"}; default = "r1"
    else:
        c = {"fixed": m}; default = "fixed"
    return c, default

METHODS = ["MixFlow-CD"] + PY + ["RegSpectral", "KMeans-feat", "Louvain", "Leiden", "Infomap", "LabelProp", "LeidenK", "SpecAttr"]
TUNED = ["MixFlow-CD"] + PY + ["RegSpectral", "Louvain", "Leiden"]
rows = []
for m in METHODS:
    for ds in DS:
        c, default = cands(m, ds)
        ok = {k: G[(ds, v)] for k, v in c.items() if (ds, v) in G and G[(ds, v)].n >= 5}
        missing = [k for k, v in c.items() if k not in ok]
        if not ok:
            rows.append(dict(method=m, dataset=ds, n_cand=0)); continue
        sel = max(ok, key=lambda k: (ok[k].Q, k == default))
        orc = max(ok, key=lambda k: ok[k].ARI)
        rho = np.nan if len(set(ok[k].Q for k in ok)) < 2 else spearmanr([ok[k].Q for k in ok], [ok[k].ARI for k in ok])[0] if len(ok) >= 3 else np.nan
        dfl = ok.get(default)
        rows.append(dict(method=m, dataset=ds, n_cand=len(ok), default=default, missing=",".join(missing), selected=sel, oracle=orc,
                         ARI_sel=ok[sel].ARI, sd_sel=ok[sel].ARIsd, ACC_sel=ok[sel].ACC, NMI_sel=ok[sel].NMI, Q_sel=ok[sel].Q,
                         ACCsd_sel=ok[sel].ACCsd, NMIsd_sel=ok[sel].NMIsd, F1_sel=ok[sel].F1, F1sd_sel=ok[sel].F1sd, mem_sel=ok[sel].mem,
                         t_search=sum(ok[k].t for k in ok), ACC_orc=ok[orc].ACC, ACC_def=dfl.ACC if dfl is not None else np.nan,
                         ARI_orc=ok[orc].ARI, ARI_def=dfl.ARI if dfl is not None else np.nan, t_sel=ok[sel].t,
                         rho=rho, cand={k: [round(ok[k].Q, 4), round(ok[k].ARI, 4)] for k in ok}))
R = pd.DataFrame(rows)
R.to_csv("selection.csv", index=False)
print(R[(R.n_cand < 4) & R.method.isin(TUNED)][["method", "dataset", "n_cand", "missing"]].to_string())
T = R[R.method.isin(TUNED)]
summ = T.groupby("method").apply(lambda x: pd.Series(dict(
    sel_is_def=int((x.selected == x.default).sum()),
    sel_is_orc=int((x.selected == x.oracle).sum()),
    mean_sel=x.ARI_sel.mean(), mean_def=x.ARI_def.mean(), mean_orc=x.ARI_orc.mean(),
    regret=(x.ARI_orc - x.ARI_sel).mean(), max_regret=(x.ARI_orc - x.ARI_sel).max(),
    gain_vs_def=(x.ARI_sel - x.ARI_def).mean(), rho=x.rho.median()))).round(3)
print(summ.to_string())
print("\nselected configurations:")
print(R[R.method.isin(TUNED)].pivot(index="dataset", columns="method", values="selected").loc[DS].to_string())
