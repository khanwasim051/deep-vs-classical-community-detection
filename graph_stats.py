"""Descriptive statistics of every graph (for the datasets table): nodes, edges, communities, attributes,
edge homophily (= 1 - realized mixing), connected components.

    python graph_stats.py --dataset all,lfr6,lfr7,attr_extra,het,sweep,sweep_more > graph_stats.txt
"""
import argparse

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from data import expand_names, homophily, load_dataset
from features import adjacency

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="all")
p.add_argument("--root", default="data")
a = p.parse_args()
rows = []
for name in expand_names(a.dataset):
    try:
        g = load_dataset(name, a.root)
    except Exception as ex:
        print(f"{name}: could not load ({type(ex).__name__}: {str(ex)[:150]})", flush=True)
        continue
    h = homophily(g)
    rows.append(dict(dataset=name, nodes=g["n"], edges=g["edge_index"].shape[1] // 2, K=len(np.unique(g["y"])),
                     attr_dim=0 if g["x"] is None else g["x"].shape[1], homophily=round(h, 3),
                     realized_mixing=round(1 - h, 3),
                     components=int(connected_components(adjacency(g["edge_index"], g["n"]), directed=False)[0])))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv("graph_stats.csv", index=False)
print(df.to_string())
