"""Fig. 3: mean ARI against total run time (Pareto analysis). (a) all methods; (b) enlarged view of the crowded
high-runtime region, with every label joined to its marker by a leader line. Legend below the plotting areas."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, OUT, save_tif  # noqa: E402
os.chdir(OUT)  # every output of this script goes to analysis/out
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
S = json.load(open("sel.json")); C = S["cost"]
nm = lambda m: "CCA-SSG" if m == "CCASSG" else m
FAM = {"Neural, trained": ["DGI", "CCASSG", "NS4GC", "DinkNet", "GAE", "DAEGC", "DMoN", "MixFlow-CD"], "Graph filter": ["MS2CAG", "SSGC"],
       "Classical": ["RegSpectral", "KMeans-feat", "Louvain", "Leiden", "Infomap", "LabelProp"]}
COL = {"Neural, trained": "#d9622b", "Graph filter": "#2a9d8f", "Classical": "#3a6ea5"}
ZX, ZY = (140, 1700), (0.482, 0.556)                      # region enlarged in panel (b)
pts = [(C[m]["total"], C[m]["mean_ari"], m, fam) for fam, ms in FAM.items() for m in ms]
size = lambda m: 40 + C[m]["mem"] / 40 if C[m]["mem"] else 40
front, best = [], -1
for x, y, m, _ in sorted(pts):
    if y > best:
        front.append((x, y)); best = y

fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 8.2), gridspec_kw=dict(height_ratios=[1, 1], hspace=0.32))
for a in (ax, bx):
    for x, y, m, fam in pts:
        a.scatter(x, y, s=size(m), c=COL[fam], marker="o" if C[m]["mem"] else "s",
                  edgecolors="black" if C[m]["over6"] or C[m]["fails"] else "none", linewidths=1.2, alpha=0.85, zorder=3)
    a.step([p[0] for p in front], [p[1] for p in front], where="post", color="#555", lw=1, ls="--", zorder=2)
    a.set_xscale("log"); a.grid(alpha=0.25, which="both")
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.set_xlabel("Total time for one run on each of the 20 graphs (s, log scale)", fontsize=9); a.set_ylabel("Mean ARI over the 20 graphs", fontsize=9)
    a.tick_params(labelsize=8)

# (a) every method outside the enlarged region is labelled here; the region itself is labelled in (b)
OFF_A = {"KMeans-feat": (-22, -14), "Louvain": (-8, -14), "DMoN": (6, 5)}
for x, y, m, _ in pts:
    if not (ZX[0] <= x <= ZX[1] and ZY[0] <= y <= ZY[1]):
        ax.annotate(nm(m), (x, y), textcoords="offset points", xytext=OFF_A.get(m, (6, 4)), fontsize=7.5)
ax.add_patch(Rectangle((ZX[0], ZY[0]), ZX[1] - ZX[0], ZY[1] - ZY[0], fill=False, ls=":", lw=1.0, ec="#333", zorder=1))
ax.text(ZX[1], ZY[1] + 0.004, "enlarged in (b)", fontsize=7.5, va="bottom", ha="right", color="#333")
ax.set_xlim(1.5, 2500); ax.set_title("(a) All methods", loc="left", fontsize=10, fontweight="bold")

# (b) enlarged region; label positions chosen so that no label covers a marker
OFF_B = {"DGI": (-34, 16), "NS4GC": (26, 18), "CCASSG": (-10, -24), "SSGC": (-58, -16), "GAE": (-34, -26),
         "DAEGC": (18, -2), "DinkNet": (28, 12), "MixFlow-CD": (30, -14)}
for x, y, m, _ in pts:
    if ZX[0] <= x <= ZX[1] and ZY[0] <= y <= ZY[1]:
        bx.annotate(nm(m), (x, y), textcoords="offset points", xytext=OFF_B.get(m, (12, 8)), fontsize=8, va="center",
                    arrowprops=dict(arrowstyle="-", color="#777", lw=0.6, shrinkA=1, shrinkB=5), zorder=5)
bx.set_xlim(*ZX); bx.set_ylim(*ZY)
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
bx.xaxis.set_major_locator(FixedLocator([200, 300, 500, 700, 1000, 1500])); bx.xaxis.set_minor_locator(NullLocator())
bx.set_xticklabels(["200", "300", "500", "700", "1,000", "1,500"]); bx.set_title("(b) Enlarged high-runtime region", loc="left", fontsize=10, fontweight="bold")

hs = [Line2D([], [], marker="o", ls="", color=COL["Neural, trained"], ms=7, label="Neural, trained (GPU)"),
      Line2D([], [], marker="o", ls="", color=COL["Graph filter"], ms=7, label="Graph filtering (GPU)"),
      Line2D([], [], marker="s", ls="", color=COL["Classical"], ms=7, label="Classical (CPU)"),
      Line2D([], [], marker="o", ls="", mfc="none", mec="black", ms=9, label="Exceeded 6 GB or a candidate failed"),
      Line2D([], [], color="#555", ls="--", label="Pareto front")]
hs += [Line2D([], [], marker="o", ls="", mfc="#bbb", mec="none", ms=(40 + g * 1024 / 40) ** 0.5, label=f"Peak GPU memory {g} GB") for g in (1, 3, 6)]
hs = [hs[0], hs[3], hs[1], hs[5], hs[2], hs[6], hs[4], hs[7]]
fig.legend(handles=hs, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=4, frameon=False, labelspacing=1.0, columnspacing=1.5)
fig.savefig("pareto.png", dpi=200, bbox_inches="tight", facecolor="white")
save_tif(fig, "Fig3.tif")  # print-resolution file for the journal
print(front)
