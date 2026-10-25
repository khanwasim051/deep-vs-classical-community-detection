import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, OUT  # noqa: E402
os.chdir(OUT)  # every output of this script goes to analysis/out
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = json.load(open("syn.json")); SW = S["sweep"]; MUS = list(S["mix"])
SHOWN = [("Infomap", "Infomap", "#2a78d6", "o"), ("RegSpectral", "RegSpectral", "#eb6834", "s"),
         ("KMeans-featM", "k-means on input (matched)", "#1baf7a", "^"), ("DGI", "DGI", "#7b4fc9", "X"),
         ("DAEGC", "DAEGC", "#e87ba4", "v"), ("NS4GC", "NS4GC", "#eda100", "D"), ("LeidenK", "Leiden (K given)", "#008300", "P")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
x = np.array([float(m) for m in MUS])
for key, lab, c, mk in SHOWN:
    mu = np.array([SW[key][m][0] for m in MUS]); sd = np.array([SW[key][m][1] for m in MUS])
    ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.12, lw=0)
    ax.plot(x, mu, color=c, lw=1.8, marker=mk, ms=5.5, mec="white", mew=1.0, label=lab)
ax.set_xlabel("Nominal mixing parameter μ", color=INK, fontsize=9)
ax.set_ylabel("ARI", color=INK, fontsize=9); ax.set_ylim(-0.02, 1.02); ax.set_xticks(x)
top = ax.secondary_xaxis("top"); top.set_xticks(x); top.set_xticklabels([f"{S['mix'][m]:.2f}" for m in MUS])
top.set_xlabel("Realized mixing (1 − edge homophily)", color=INK, fontsize=9); top.tick_params(colors=MUTED, labelsize=8)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax.spines[sp].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, labelcolor=INK)
fig.tight_layout(); fig.savefig("sweep48.png", bbox_inches="tight", facecolor="white")
