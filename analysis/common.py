"""Shared constants and loaders for the paper's analysis. All inputs live in the repository root."""
import ast, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
os.makedirs(OUT, exist_ok=True)

STRUCT = ["email-eu-core", "football", "dolphins", "ukfaculty", "kki"]
ATT_ORIG = ["cora", "citeseer", "pubmed", "amazon-photo"]
ATT_NEW = ["wiki", "blogcatalog", "coauthor-cs", "chameleon", "squirrel", "actor"]
HETERO = ["chameleon", "squirrel", "actor"]
LFRB = ["lfr1", "lfr2", "lfr3", "lfr4", "lfr6", "lfr7"]
DS = STRUCT + ATT_ORIG + ATT_NEW + LFRB                       # the 21 graphs
ORIG15 = STRUCT + ATT_ORIG + LFRB
ATT = ATT_ORIG + ATT_NEW
STR = STRUCT + LFRB
SAT = ["kki", "lfr3", "lfr4"]
PY = ["MS2CAG", "SSGC", "CCASSG", "DGI", "NS4GC", "DinkNet", "GAE", "DAEGC", "DMoN"]
DEEP = PY + ["MixFlow-CD"]
M16 = DEEP + ["RegSpectral", "KMeans-feat", "Louvain", "Leiden", "Infomap", "LabelProp"]
TUNED = DEEP + ["RegSpectral", "Louvain", "Leiden"]
MUS = [round(0.1 * i, 1) for i in range(1, 9)]
SWEEP = [f"mu{m}g{g}" for m in MUS for g in range(1, 7)]
HET = [f"het_t{t}_c{c}_g{g}" for t in ("2", "2.5", "3") for c in ("1.1", "2") for g in (1, 2, 3)]

# Clustering accuracy published with the PyAGC reference code (Liu et al., arXiv:2602.08519), for Table 4
PUBLISHED_ACC = {
    "cora": {"SSGC": 0.6538, "MS2CAG": 0.7029, "GAE": 0.6753, "DGI": 0.7222, "CCASSG": 0.7335, "NS4GC": 0.7492,
             "DAEGC": 0.6406, "DinkNet": 0.7249, "DMoN": 0.5299},
    "amazon-photo": {"SSGC": 0.7634, "MS2CAG": 0.7913, "GAE": 0.7154, "DGI": 0.7757, "CCASSG": 0.709, "NS4GC": 0.7921,
                     "DAEGC": 0.7778, "DinkNet": 0.7403, "DMoN": 0.7268},
}


def runs():
    return pd.read_csv(os.path.join(ROOT, "all_runs.csv"))


def graph_stats():
    gs = {}
    for f in ["graph_stats.txt", "graph_stats_base.txt"]:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            for ln in open(p):
                if ln.startswith("{"):
                    r = ast.literal_eval(ln); gs[r["dataset"]] = r
    return gs


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj


def save_tif(fig, name, dpi=600):
    """Save a print-resolution copy of a figure: RGB TIFF with LZW compression at 600 dpi (Elsevier artwork guidelines)."""
    from PIL import Image
    fig.savefig(name, dpi=dpi, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    Image.open(name).convert("RGB").save(name, compression="tiff_lzw", dpi=(dpi, dpi))
