"""Classical / shallow baselines under the same protocol (same data, same metrics, same CSVs).

    python baselines.py --dataset all --runs 5
Methods: Louvain, Leiden and Infomap (need `leidenalg` + `igraph`), label propagation,
spectral clustering on the adjacency (K given), k-means on the input features (K given).
"""
import argparse
import csv
import os
import time

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

from data import ALL, expand_names, describe, load_dataset
from features import adjacency, spectral_diffusion_features
from metrics import evaluate, repair_and_done


def to_nx(g):
    G = nx.Graph()
    G.add_nodes_from(range(g["n"]))
    ei = g["edge_index"]
    G.add_edges_from(zip(ei[0][ei[0] < ei[1]], ei[1][ei[0] < ei[1]]))
    return G


def partition_to_labels(parts, n):
    y = np.zeros(n, dtype=np.int64)
    for i, c in enumerate(parts):
        y[list(c)] = i
    return y


def louvain(g, K, seed):
    return partition_to_labels(nx.community.louvain_communities(to_nx(g), seed=seed), g["n"])


def leiden(g, K, seed):
    import igraph as ig
    import leidenalg as la
    ei = g["edge_index"]
    m = ei[0] < ei[1]
    G = ig.Graph(n=g["n"], edges=list(zip(ei[0][m].tolist(), ei[1][m].tolist())))
    return np.array(la.find_partition(G, la.ModularityVertexPartition, seed=seed).membership)


def infomap(g, K, seed):
    """Infomap (Rosvall & Bergstrom, 2008) via igraph; chooses its own number of communities."""
    import random
    import igraph as ig
    random.seed(seed)
    ig.set_random_number_generator(random)
    ei = g["edge_index"]
    m = ei[0] < ei[1]
    G = ig.Graph(n=g["n"], edges=list(zip(ei[0][m].tolist(), ei[1][m].tolist())))
    return np.array(G.community_infomap(trials=10).membership)


def _tuned(fn, **kw):
    return lambda g, K, seed: fn(g, K, seed, **kw)


def reg_spectral_t(g, K, seed, tau_mult=1.0):
    X = _reg_spectral_embedding(g, K, seed, tau_mult)
    return KMeans(K, n_init=10 if K <= 50 else 3, random_state=seed).fit_predict(X)


def louvain_r(g, K, seed, res=1.0):
    return partition_to_labels(nx.community.louvain_communities(to_nx(g), seed=seed, resolution=res), g["n"])


def leiden_r(g, K, seed, res=1.0):
    import igraph as ig
    import leidenalg as la
    ei = g["edge_index"]
    m = ei[0] < ei[1]
    G = ig.Graph(n=g["n"], edges=list(zip(ei[0][m].tolist(), ei[1][m].tolist())))
    return np.array(la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter=res, seed=seed).membership)


def leiden_k(g, K, seed):
    """Leiden (modularity with resolution) with the resolution chosen by bisection so that the number of
    communities is as close as possible to K: puts the modularity method in the known-K setting."""
    import igraph as ig
    import leidenalg as la
    ei = g["edge_index"]
    m = ei[0] < ei[1]
    G = ig.Graph(n=g["n"], edges=list(zip(ei[0][m].tolist(), ei[1][m].tolist())))
    lo, hi, best = -4.0, 4.0, None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        memb = np.array(la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter=10 ** mid,
                                          seed=seed).membership)
        c = len(np.unique(memb))
        if best is None or abs(c - K) < abs(best[0] - K):
            best = (c, memb)
        if c == K:
            break
        lo, hi = (mid, hi) if c < K else (lo, mid)
    return best[1]


def kmeans_feat_matched(g, K, seed):
    """k-means on the structural features with the same dimension the learned methods receive."""
    if g["x"] is not None:
        return kmeans_feat(g, K, seed)
    x = spectral_diffusion_features(g["edge_index"], g["n"], dim=int(min(max(64, K + 16), 512)))
    return KMeans(K, n_init=10, random_state=seed).fit_predict(x)


def spec_attr(g, K, seed):
    """Non-neural joint baseline: k-means on [regularized spectral embedding | PCA of the attributes],
    each part row-normalized and given equal weight. Falls back to regularized spectral without attributes."""
    from sklearn.decomposition import PCA
    if g["x"] is None:
        return reg_spectral(g, K, seed)
    U = _reg_spectral_embedding(g, K, seed)
    x = g["x"] / (np.linalg.norm(g["x"], axis=1, keepdims=True) + 1e-12)
    Z = PCA(n_components=min(K, x.shape[1]), random_state=seed).fit_transform(x)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    return KMeans(K, n_init=10, random_state=seed).fit_predict(np.concatenate([U, Z], 1))


def label_prop(g, K, seed):
    return partition_to_labels(nx.community.asyn_lpa_communities(to_nx(g), seed=seed), g["n"])


def spectral(g, K, seed):
    A = adjacency(g["edge_index"], g["n"])
    sc = SpectralClustering(K, affinity="precomputed", random_state=seed, assign_labels="kmeans")
    return sc.fit_predict(A)


def _reg_spectral_embedding(g, K, seed, tau_mult=1.0):
    """Regularized spectral clustering (Amini et al., 2013; Qin & Rohe, 2013).

    A_tau = A + (tau/n) 11^T with tau = mean degree, then the top-K eigenvectors of
    D_tau^{-1/2} A_tau D_tau^{-1/2}, row-normalized, then k-means. The regularizer stops
    small disconnected components (isolated nodes in Cora/Citeseer/Amazon/PubMed) from
    taking all the eigenvectors, which is what breaks plain spectral clustering there.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    n = g["n"]
    A = adjacency(g["edge_index"], n)
    A.data[:] = 1.0
    d = np.asarray(A.sum(1)).ravel()
    tau = d.mean() * tau_mult
    dm = 1.0 / np.sqrt(d + tau)
    k = min(K, n - 2)
    if n <= 3000:
        S = (dm[:, None] * A.toarray() * dm[None, :]) + (tau / n) * np.outer(dm, dm)
        vals, vecs = np.linalg.eigh(S)
        vecs = vecs[:, -k:]
    else:
        def mv(x):
            x = np.asarray(x).reshape(n, -1)
            y = dm[:, None] * (A @ (dm[:, None] * x)) + (tau / n) * dm[:, None] * (dm @ x)
            return y if y.shape[1] > 1 else y.ravel()
        op = LinearOperator((n, n), matvec=mv, matmat=mv, dtype=np.float64)
        v0 = np.random.default_rng(seed).standard_normal(n)
        vals, vecs = eigsh(op, k=k, which="LA", v0=v0, tol=1e-6)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)


def reg_spectral(g, K, seed):
    X = _reg_spectral_embedding(g, K, seed)
    return KMeans(K, n_init=10 if K <= 50 else 3, random_state=seed).fit_predict(X)


def kmeans_feat(g, K, seed):
    x = g["x"] if g["x"] is not None else spectral_diffusion_features(g["edge_index"], g["n"])
    return KMeans(K, n_init=10, random_state=seed).fit_predict(x)


METHODS = {"Louvain": louvain, "Leiden": leiden, "Infomap": infomap, "LabelProp": label_prop, "Spectral": spectral,
           "RegSpectral": reg_spectral, "KMeans-feat": kmeans_feat,
           "LeidenK": leiden_k, "KMeans-featM": kmeans_feat_matched, "SpecAttr": spec_attr}
# label-free selection candidates (concern 4): four values of each method's main parameter, fixed in advance
for _t in (0.25, 0.5, 2.0):
    METHODS[f"RegSpectral_t{_t:g}"] = _tuned(reg_spectral_t, tau_mult=_t)
for _r in (0.5, 2.0, 4.0):
    METHODS[f"Louvain_r{_r:g}"] = _tuned(louvain_r, res=_r)
    METHODS[f"Leiden_r{_r:g}"] = _tuned(leiden_r, res=_r)
DEFAULT = ["Louvain", "Leiden", "Infomap", "LabelProp", "Spectral", "RegSpectral", "KMeans-feat"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all")
    p.add_argument("--methods", default=",".join(DEFAULT))
    p.add_argument("--k_hat", default="", help="K-unknown setting: json file {dataset: K_hat}; method names get '@Khat'")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--resume", action="store_true", help="skip seeds already saved for this method")
    a = p.parse_args()
    names = expand_names(a.dataset)
    os.makedirs(a.out, exist_ok=True)
    for name in names:
        g = load_dataset(name, a.root)
        print(describe(g))
        K = len(np.unique(g["y"]))
        if a.k_hat:
            import json
            K = int(json.load(open(a.k_hat))[name])
        path = os.path.join(a.out, f"{name}.csv")
        new = not os.path.exists(path)
        K_true = len(np.unique(g["y"]))
        for mname in a.methods.split(","):
            label = mname + ("@Khat" if a.k_hat else "")
            res_all = []
            done = repair_and_done(path, label) if a.resume else set()
            new = not os.path.exists(path)
            for r in range(a.runs):
                if r in done:
                    continue
                t0 = time.time()
                try:
                    pred = METHODS[mname](g, K, r)
                except ImportError:
                    print(f"  {mname}: skipped (pip install leidenalg igraph)")
                    break
                except Exception as ex:
                    print(f"  {mname}: failed ({ex})")
                    break
                res = evaluate(g["y"], pred, g["edge_index"])
                res_all.append(res)
                with open(path, "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["dataset", "method", "seed", "ACC", "NMI", "ARI", "F1", "Q", "K_true", "K_used",
                                    "K_pred", "time_s", "peak_mem_mb"])
                        new = False
                    w.writerow([name, label, r] + [round(res[k], 4) for k in ["ACC", "NMI", "ARI", "F1", "Q"]]
                               + [K_true, K, res["K_pred"], round(time.time() - t0, 1), 0])
            if res_all:
                print(f"  {label:12s} " + "  ".join(
                    f"{k} {np.mean([r[k] for r in res_all]):.4f}" for k in ["ACC", "NMI", "ARI", "F1"])
                      + f"  K_pred {res_all[0]['K_pred']}")


if __name__ == "__main__":
    main()
