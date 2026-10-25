import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, OUT  # noqa: E402
os.chdir(OUT)  # every output of this script goes to analysis/out
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = json.load(open("sel.json")); C = S["cost"]
nm = lambda m: "CCA-SSG" if m == "CCASSG" else m
FAM = {"Trained deep": ["DGI", "CCASSG", "NS4GC", "DinkNet", "GAE", "DAEGC", "DMoN", "MixFlow-CD"], "Graph filter": ["MS2CAG", "SSGC"],
       "Classical": ["RegSpectral", "KMeans-feat", "Louvain", "Leiden", "Infomap", "LabelProp"]}
COL = {"Trained deep": "#d9622b", "Graph filter": "#2a9d8f", "Classical": "#3a6ea5"}
fig, ax = plt.subplots(figsize=(7.2, 4.4))
pts = []
for fam, ms in FAM.items():
    for m in ms:
        c = C[m]; x, y = c["total"], c["mean_ari"]; pts.append((x, y, m))
        size = 40 + c["mem"] / 40 if c["mem"] else 40
        mk = "o" if c["mem"] else "s"
        edge = "black" if c["over6"] or c["fails"] else "none"
        ax.scatter(x, y, s=size, c=COL[fam], marker=mk, edgecolors=edge, linewidths=1.2, alpha=0.9, zorder=3, label=fam if m == ms[0] else None)
        off = {"SSGC": (-34, -16), "KMeans-feat": (-20, -13), "GAE": (8, -2), "CCASSG": (-20, 8), "NS4GC": (8, 5)}.get(m, (6, 4))
        ax.annotate(nm(m), (x, y), textcoords="offset points", xytext=off, fontsize=7.5)
pts.sort()
front, best = [], -1
for x, y, m in pts:
    if y > best:
        front.append((x, y)); best = y
ax.step([p[0] for p in front], [p[1] for p in front], where="post", color="#555", lw=1, ls="--", zorder=2, label="Pareto front")
ax.set_xscale("log"); ax.set_xlabel("Total time for one run on each of the 20 graphs (s, log scale)"); ax.set_ylabel("Mean ARI over the 20 graphs")
ax.grid(alpha=0.25, which="both"); from matplotlib.lines import Line2D
hs = [Line2D([], [], marker="o", ls="", color=COL["Trained deep"], ms=7, label="Trained deep (GPU)"), Line2D([], [], marker="o", ls="", color=COL["Graph filter"], ms=7, label="Graph filter (GPU)"),
      Line2D([], [], marker="s", ls="", color=COL["Classical"], ms=7, label="Classical (CPU)"), Line2D([], [], marker="o", ls="", mfc="none", mec="black", ms=9, label="Exceeded 6 GB or a candidate failed"),
      Line2D([], [], color="#555", ls="--", label="Pareto front")]
ax.legend(handles=hs, fontsize=7.5, loc="lower right", frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("pareto.png", dpi=180, bbox_inches="tight", facecolor="white")
print(front)
