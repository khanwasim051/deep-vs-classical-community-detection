import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, OUT  # noqa: E402
os.chdir(OUT)  # every output of this script goes to analysis/out
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
S = json.load(open("sel.json")); fr = S["FA"]; cd = fr["cd"]
nm = lambda m: "CCA-SSG" if m == "CCASSG" else m
items = list(fr["ranks"].items()); k = len(items)
fig, ax = plt.subplots(figsize=(10, 3.7)); ax.set_xlim(-2.6, 18.6); ax.set_ylim(-0.2, 1.45); ax.axis("off")
X = lambda r: r  # rank axis
y0 = 1.05
ax.plot([1, k], [y0, y0], color="#333", lw=1)
for t in range(1, k + 1):
    ax.plot([t, t], [y0, y0 + 0.03], color="#333", lw=1); ax.text(t, y0 + 0.06, str(t), ha="center", va="bottom", fontsize=9)
ax.plot([1, 1 + cd], [y0 + 0.35, y0 + 0.35], color="#b22222", lw=3); ax.text(1 + cd / 2, y0 + 0.39, f"CD = {cd:.2f}", ha="center", color="#b22222", fontsize=9.5)
half = (k + 1) // 2
left, right = items[:half], items[half:][::-1]
for i, (m, r) in enumerate(left):
    y = 0.8 - i * 0.1
    ax.plot([r, r, 0.2], [y0, y, y], color="#333", lw=0.9)
    ax.text(0.1, y, f"{nm(m)} ({r:.2f})", ha="right", va="center", fontsize=9, fontweight="bold" if m == "MixFlow-CD" else None)
for i, (m, r) in enumerate(right):
    y = 0.8 - i * 0.1
    ax.plot([r, r, k + 0.8], [y0, y, y], color="#333", lw=0.9)
    ax.text(k + 0.9, y, f"{nm(m)} ({r:.2f})", ha="left", va="center", fontsize=9, fontweight="bold" if m == "MixFlow-CD" else None)
rs = [r for _, r in items]
cl = []
for i in range(k):
    j = max(j for j in range(k) if rs[j] - rs[i] < cd)
    if j > i and not any(a <= i and j <= b for a, b in cl):
        cl.append((i, j))
for n, (i, j) in enumerate(cl):
    yy = y0 - 0.05 - n * 0.035
    ax.plot([rs[i] - 0.05, rs[j] + 0.05], [yy, yy], color="#2563a8", lw=3.2, solid_capstyle="butt")
fig.tight_layout(pad=0.1); fig.savefig("cd_ari_sel.png", dpi=170, bbox_inches="tight", facecolor="white")
print(cl)
