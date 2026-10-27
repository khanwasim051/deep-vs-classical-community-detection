import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, OUT, save_tif  # noqa: E402
os.chdir(OUT)  # every output of this script goes to analysis/out
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = json.load(open("syn.json")); SW = S["sweep"]; MUS = list(S["mix"])
SHOWN = [("Infomap", "Infomap", "#2a78d6", "o"), ("RegSpectral", "RegSpectral", "#eb6834", "s"),
         ("KMeans-featM", "k-means on input (matched)", "#1baf7a", "^"), ("DGI", "DGI", "#7b4fc9", "X"),
         ("DAEGC", "DAEGC", "#e87ba4", "v"), ("NS4GC", "NS4GC", "#eda100", "D"), ("LeidenK", "Leiden (K given)", "#008300", "P")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.4, 6.2), dpi=200, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.45))
x = np.array([float(m) for m in MUS])
for n, (key, lab, c, mk) in enumerate(SHOWN):
    mu = np.array([SW[key][m][0] for m in MUS]); sd = np.array([SW[key][m][1] for m in MUS])
    dx = 0.0
    ax.plot(x, mu, color=c, lw=1.6, marker=mk, ms=4.8, mec="white", mew=0.8, label=lab, zorder=3)
    ax.errorbar(x + dx, mu, yerr=sd, color=c, ls="none", elinewidth=1.0, capsize=2.2, capthick=1.0, zorder=4)
    bx.plot(x, sd, color=c, lw=1.4, marker=mk, ms=4, mec="white", mew=0.6)
ax.set_xlabel("Nominal mixing parameter μ", color=INK, fontsize=9)
ax.set_ylabel("ARI (mean ± 1 s.d.)", color=INK, fontsize=9); ax.set_ylim(-0.02, 1.02); ax.set_xticks(x)
top = ax.secondary_xaxis("top"); top.set_xticks(x); top.set_xticklabels([f"{S['mix'][m]:.2f}" for m in MUS])
top.set_xlabel("Realized mixing (1 − edge homophily)", color=INK, fontsize=9); top.tick_params(colors=MUTED, labelsize=8)
ax.text(-0.09, 1.13, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK)
bx.text(-0.09, 1.08, "(b)", transform=bx.transAxes, fontsize=10, fontweight="bold", color=INK)
bx.set_xticks(x); bx.set_xlabel("Nominal mixing parameter μ", color=INK, fontsize=9)
bx.set_ylabel("s.d. of ARI\nacross six graphs", color=INK, fontsize=9); bx.set_ylim(0, None)
for a in (ax, bx):
    a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True)
    for sp in ["top", "right"]: a.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: a.spines[sp].set_color(MUTED)
    a.tick_params(colors=MUTED, labelsize=8)
h, l = ax.get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=4, labelcolor=INK)
fig.savefig("sweep48.png", bbox_inches="tight", facecolor="white")
save_tif(fig, "Fig4.tif")  # print-resolution file for the journal
