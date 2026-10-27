"""Fig. 1: study design overview (graphs -> methods -> protocol -> evaluation)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, save_tif  # noqa: E402
os.chdir(OUT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, LINE = "#1f1f1e", "#55544f", "#9a9890"
FILL = {"data": "#e8f0fb", "method": "#fdeee6", "proto": "#e6f6ef", "eval": "#f3f3f1"}
EDGE = {"data": "#2a78d6", "method": "#eb6834", "proto": "#1baf7a", "eval": "#6b6a64"}

fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")


def box(x, y, w, h, kind, title, lines, fs=6.6, step=3.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=FILL[kind], ec=EDGE[kind], lw=0.9))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=7.2, fontweight="bold", color=INK)
    for i, ln in enumerate(lines):
        ax.text(x + 1.4, y + h - 5.6 - i * step, ln, ha="left", va="top", fontsize=fs, color=INK)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.0, mutation_scale=8))


ax.text(12.5, 58.5, "Graphs", ha="center", fontsize=8, color=MUTED, fontweight="bold")
ax.text(41, 58.5, "Methods (16)", ha="center", fontsize=8, color=MUTED, fontweight="bold")
ax.text(66.5, 58.5, "Protocol", ha="center", fontsize=8, color=MUTED, fontweight="bold")
ax.text(89, 58.5, "Evaluation", ha="center", fontsize=8, color=MUTED, fontweight="bold")

box(1, 41, 23, 15, "data", "Structure-only (5)", ["Email-Eu-core, Football,", "Dolphins, UKfaculty, KKI"])
box(1, 21.5, 23, 17, "data", "Attributed (10)", ["Cora, Citeseer, PubMed, Wiki,", "Photo, BlogCatalog, Coauthor-CS;", "heterophilous: Chameleon,", "Squirrel, Actor"], fs=5.9)
box(1, 2, 23, 18, "data", "Synthetic LFR", ["6 graphs, μ = 0.2–0.7", "Sweep: 48 graphs, μ = 0.1–0.8", "Heterogeneity: 18 graphs", "(n = 5,000; realized mixing)"], fs=6.0)

box(28, 41, 26, 15, "method", "Neural, trained (8)", ["DGI, CCA-SSG, NS4GC, DinkNet,", "GAE, DAEGC, DMoN (PyAGC code);", "MixFlow-CD (flow mixture)"], fs=6.2)
box(28, 23, 26, 15, "method", "Graph filtering, no training (2)", ["SSGC, MS2CAG", "(PyAGC reference code)"], fs=6.2)
box(28, 2, 26, 18, "method", "Classical (6)", ["Need K: RegSpectral, KMeans-feat", "Choose K: Louvain, Leiden,", "  Infomap, LabelProp", "Controls: Leiden-K, SpecAttr,", "  KMeans-featM"], fs=6.0, step=2.55)

box(58, 2, 17, 54, "proto", "Controlled inputs", ["Common graph data for all", "Feature-based methods:", "  identical available attributes,", "  or constructed structural", "  features if there are none", "Topology-only methods:", "  the edges only", "Permitted inputs per method", "  explicitly controlled (§5.6)", "K: true K where needed;", "  unknown K: 2 estimators", "5 seeds (1 per synthetic graph)", "No labels for selection:", "  4 configs, max modularity", "Budget: 6 GB GPU, 16 GB RAM,", "  60 min per job"], fs=5.4, step=2.85)

box(79, 31, 20, 25, "eval", "Metrics", ["ACC (Hungarian)", "ARI (headline)", "NMI, macro-F1", "Modularity, time,", "  peak memory"], fs=6.3)
box(79, 2, 20, 26, "eval", "Statistics (per graph)", ["Friedman + Nemenyi on ARI,", "  by kind of graph", "Paired differences,", "  bootstrap 95% CI", "Holm-adjusted signed-rank", "  (stratified for synthetic)", "Non-inferiority (δ = 0.05)"], fs=6.0)

for y in (48.5, 30.5, 11):
    arrow(24.6, y, 27.4, y)
for y in (48.5, 30.5, 11):
    arrow(54.6, y, 57.4, y)
arrow(75.6, 43.5, 78.4, 43.5)
arrow(75.6, 15, 78.4, 15)
arrow(89, 30.6, 89, 28.6)

fig.tight_layout(pad=0.2)
fig.savefig("fig1_overview.png", bbox_inches="tight", facecolor="white")
save_tif(fig, "Fig1.tif")
print("Fig. 1 written")
