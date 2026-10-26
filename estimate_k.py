"""Label-free estimate of the number of communities, used for the K-unknown setting.

    python estimate_k.py --dataset all,lfr6,lfr7 --out khat.json

Two label-free estimators (no labels are used by either):
  eigengap: the eigengap of the regularized normalized adjacency (the operator that regularized spectral
            clustering uses, tau = mean degree). With eigenvalues l1 >= l2 >= ..., K_hat is the k in
            [2, k_max] that maximizes l_k - l_{k+1}; k_max = min(200, n // 5).
  bethe:    the number of negative eigenvalues of the Bethe Hessian H(r) = (r^2 - 1) I - r A + D with
            r = sqrt(sum d_i^2 / sum d_i - 1) (Saade, Krzakala and Zdeborova, NeurIPS 2014), clipped to [2, k_max].

    python estimate_k.py --dataset all,lfr6,lfr7 --method bethe --out khat_bethe.json
"""
import argparse
import json
import os

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from data import expand_names, load_dataset
from features import adjacency


def eigengap_k(edge_index, n, k_max=None):
    A = adjacency(edge_index, n)
    A.data[:] = 1.0
    d = np.asarray(A.sum(1)).ravel()
    tau = d.mean()
    dm = 1.0 / np.sqrt(d + tau)
    k_max = int(k_max or min(200, n // 5))
    m = min(k_max + 1, n - 2)
    if n <= 3000:
        S = (dm[:, None] * A.toarray() * dm[None, :]) + (tau / n) * np.outer(dm, dm)
        vals = np.sort(np.linalg.eigvalsh(S))[::-1][:m]
    else:
        def mv(x):
            x = np.asarray(x).reshape(n, -1)
            y = dm[:, None] * (A @ (dm[:, None] * x)) + (tau / n) * dm[:, None] * (dm @ x)
            return y if y.shape[1] > 1 else y.ravel()
        op = LinearOperator((n, n), matvec=mv, matmat=mv, dtype=np.float64)
        vals = np.sort(eigsh(op, k=m, which="LA", tol=1e-6, v0=np.random.default_rng(0).standard_normal(n))[0])[::-1]
    gaps = vals[1:m - 1] - vals[2:m]          # gaps[i] = l_{i+2} - l_{i+3}, i.e. k = i + 2
    return int(np.argmax(gaps) + 2), vals


def bethe_k(edge_index, n, k_max=None):
    A = adjacency(edge_index, n)
    A.data[:] = 1.0
    d = np.asarray(A.sum(1)).ravel()
    r = float(np.sqrt(max((d ** 2).sum() / d.sum() - 1.0, 1.0)))
    import scipy.sparse as sp
    H = ((r * r - 1.0) * sp.eye(n) - r * A + sp.diags(d)).tocsr()
    k_max = int(k_max or min(200, n // 5))
    m = min(k_max + 1, n - 2)
    if n <= 3000:
        vals = np.linalg.eigvalsh(H.toarray())[:m]
    else:
        vals = eigsh(H, k=m, which="SA", tol=1e-6, v0=np.random.default_rng(0).standard_normal(n), return_eigenvectors=False)
    return int(np.clip((vals < 0).sum(), 2, k_max)), vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all")
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="khat.json")
    p.add_argument("--method", default="eigengap", choices=["eigengap", "bethe"])
    a = p.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for name in expand_names(a.dataset):
        g = load_dataset(name, a.root)
        k, _ = (eigengap_k if a.method == "eigengap" else bethe_k)(g["edge_index"], g["n"])
        res[name] = k
        print(f"{name:18s} K_true {len(np.unique(g['y'])):4d}   K_hat {k:4d}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
