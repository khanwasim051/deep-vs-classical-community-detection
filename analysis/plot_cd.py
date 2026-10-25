"""Fig. 2: critical-difference diagrams of the ARI ranks, by graph type (structure-only and attributed) and pooled."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, OUT  # noqa: E402
os.chdir(OUT)
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = json.load(open("sel.json"))
nm = lambda m: "CCA-SSG" if m == "CCASSG" else m


def draw(ax, fr, title):
    cd = fr["cd"]; items = list(fr["ranks"].items()); k = len(items)
    ax.set_xlim(-2.6, k + 2.6); ax.set_ylim(-0.2, 1.55); ax.axis("off")
    y0 = 1.05
    ax.text(-2.5, 1.5, title, ha="left", va="top", fontsize=10.5, fontweight="bold")
    ax.plot([1, k], [y0, y0], color="#333", lw=1)
    for t in range(1, k + 1):
        ax.plot([t, t], [y0, y0 + 0.03], color="#333", lw=1); ax.text(t, y0 + 0.06, str(t), ha="center", va="bottom", fontsize=8.5)
    ax.plot([1, 1 + cd], [y0 + 0.3, y0 + 0.3], color="#b22222", lw=3); ax.text(1 + cd / 2, y0 + 0.33, f"CD = {cd:.2f}", ha="center", color="#b22222", fontsize=9)
    half = (k + 1) // 2
    for side, part in [(0, items[:half]), (1, items[half:][::-1])]:
        for i, (m, r) in enumerate(part):
            y = 0.8 - i * 0.1
            x_end = 0.2 if side == 0 else k + 0.8
            ax.plot([r, r, x_end], [y0, y, y], color="#333", lw=0.9)
            ax.text(0.1 if side == 0 else k + 0.9, y, f"{nm(m)} ({r:.2f})", ha="right" if side == 0 else "left", va="center", fontsize=8.5,
                    fontweight="bold" if m == "MixFlow-CD" else None)
    rs = [r for _, r in items]; cl = []
    for i in range(k):
        j = max(j for j in range(k) if rs[j] - rs[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cl):
            cl.append((i, j))
    for n, (i, j) in enumerate(cl):
        yy = y0 - 0.05 - n * 0.035
        ax.plot([rs[i] - 0.05, rs[j] + 0.05], [yy, yy], color="#2563a8", lw=3.0, solid_capstyle="butt")


fig, axes = plt.subplots(2, 1, figsize=(10, 7.4))
draw(axes[0], S["ari_struct"], f"(a) Structure-only and LFR graphs ({S['ari_struct']['N']})")
draw(axes[1], S["ari_attr"], f"(b) Attributed graphs ({S['ari_attr']['N']})")
fig.tight_layout(pad=0.3); fig.savefig("cd_ari_groups.png", dpi=170, bbox_inches="tight", facecolor="white")
fig, ax = plt.subplots(figsize=(10, 3.7)); draw(ax, S["FA"], f"All graphs ({S['FA']['N']})")
fig.tight_layout(pad=0.1); fig.savefig("cd_ari_sel.png", dpi=170, bbox_inches="tight", facecolor="white")
print("CD figures written")
