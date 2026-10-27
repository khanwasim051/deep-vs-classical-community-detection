"""Fig. 2: critical-difference diagrams of the ARI ranks, by graph type (structure-only and attributed) and pooled."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, OUT, save_tif  # noqa: E402
os.chdir(OUT)
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
S = json.load(open("sel.json"))
nm = lambda m: "CCA-SSG" if m == "CCASSG" else m
FAM = {**{m: "neural" for m in ["DGI", "CCASSG", "NS4GC", "DinkNet", "GAE", "DAEGC", "DMoN", "MixFlow-CD"]},
       **{m: "filter" for m in ["SSGC", "MS2CAG"]},
       **{m: "classical" for m in ["RegSpectral", "KMeans-feat", "KMeans-featM", "Louvain", "Leiden", "LeidenK", "Infomap", "LabelProp", "SpecAttr"]}}
COL = {"neural": "#c4501b", "filter": "#1f8a7e", "classical": "#2f5f99"}
INK, CLQ = "#222", "#555"


def draw(ax, fr, title):
    cd = fr["cd"]; items = list(fr["ranks"].items()); k = len(items); N = fr["N"]
    ax.set_xlim(-2.6, k + 2.6); ax.set_ylim(-0.2, 1.62); ax.axis("off")
    y0 = 1.05
    ax.text(-2.5, 1.6, title, ha="left", va="top", fontsize=10.5, fontweight="bold", color=INK)
    ax.plot([1, k], [y0, y0], color=INK, lw=1)
    for t in range(1, k + 1):
        ax.plot([t, t], [y0, y0 + 0.03], color=INK, lw=1); ax.text(t, y0 + 0.05, str(t), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.text((1 + k) / 2, y0 + 0.17, "Average ARI rank (1 = best)", ha="center", va="bottom", fontsize=8.5, color=INK)
    # critical difference, drawn to scale from rank 1
    yc = y0 + 0.40
    ax.plot([1, 1 + cd], [yc, yc], color="#b22222", lw=2.2)
    for xx in (1, 1 + cd): ax.plot([xx, xx], [yc - 0.03, yc + 0.03], color="#b22222", lw=1.4)
    ax.text(1 + cd + 0.2, yc, f"critical difference CD = {cd:.2f}  (Nemenyi, α = 0.05; {k} methods, {N} graphs)",
            ha="left", va="center", color="#b22222", fontsize=8.3)
    half = (k + 1) // 2
    for side, part in [(0, items[:half]), (1, items[half:][::-1])]:
        for i, (m, r) in enumerate(part):
            y = 0.78 - i * 0.1
            x_end = 0.2 if side == 0 else k + 0.8
            c = COL[FAM.get(m, "classical")]
            ax.plot([r, r, x_end], [y0, y, y], color=c, lw=0.9)
            ax.text(0.1 if side == 0 else k + 0.9, y, f"{nm(m)} ({r:.2f})", ha="right" if side == 0 else "left", va="center", fontsize=8.5, color=c)
    rs = [r for _, r in items]; cl = []
    for i in range(k):
        j = max(j for j in range(k) if rs[j] - rs[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cl):
            cl.append((i, j))
    for n, (i, j) in enumerate(cl):
        yy = y0 - 0.06 - n * 0.04
        ax.plot([rs[i] - 0.05, rs[j] + 0.05], [yy, yy], color=CLQ, lw=3.2, solid_capstyle="butt")


def legend(fig):
    hs = [Line2D([], [], color=COL["neural"], lw=2, label="Neural, trained"), Line2D([], [], color=COL["filter"], lw=2, label="Graph filtering, no training"),
          Line2D([], [], color=COL["classical"], lw=2, label="Classical"),
          Line2D([], [], color=CLQ, lw=4, label="Methods under one bar: not significantly different"),
          Line2D([], [], color="#b22222", lw=2, label="Critical difference (to scale)")]
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, fontsize=8.3, bbox_to_anchor=(0.5, -0.01))


fig, axes = plt.subplots(2, 1, figsize=(10, 8.0))
draw(axes[0], S["ari_struct"], f"(a) Structure-only and LFR graphs ({S['ari_struct']['N']})")
draw(axes[1], S["ari_attr"], f"(b) Attributed graphs ({S['ari_attr']['N']})")
fig.tight_layout(pad=0.3, rect=(0, 0.05, 1, 1)); legend(fig)
fig.savefig("cd_ari_groups.png", dpi=170, bbox_inches="tight", facecolor="white")
save_tif(fig, "Fig2.tif")  # print-resolution file for the journal
fig, ax = plt.subplots(figsize=(10, 4.2)); draw(ax, S["FA"], f"All graphs ({S['FA']['N']})")
fig.tight_layout(pad=0.1, rect=(0, 0.1, 1, 1)); legend(fig)
fig.savefig("cd_ari_sel.png", dpi=170, bbox_inches="tight", facecolor="white")
print("CD figures written")
