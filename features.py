"""Structural input features and tie classes.

* spectral_diffusion_features: for non-attributed graphs. A fixed personalized-PageRank-style
  filter g(lambda) = sum_t alpha (1-alpha)^t lambda^t is applied to the top eigenpairs of the
  normalized adjacency. Simple, fixed and identical for all methods that use it.
* tie_classes: edge role from embeddedness (common neighbours / Jaccard).
    0 = weak tie (no common neighbour, bridge-like), 1 = medium, 2 = strong (above-median Jaccard).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


def adjacency(edge_index, n):
    r, c = edge_index
    return sp.csr_matrix((np.ones(len(r), dtype=np.float64), (r, c)), shape=(n, n))


def spectral_diffusion_features(edge_index, n, dim=64, alpha=0.15, order=10):
    A = adjacency(edge_index, n) + sp.eye(n)
    d = np.asarray(A.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(d))
    S = (dinv @ A @ dinv).tocsr()
    ncomp = connected_components(A, directed=False)[0]
    # a disconnected graph has one eigenvalue 1 per component, so keep enough eigenvectors
    k = int(min(n - 2, max(dim, ncomp + 32)))
    if n <= 3000:
        vals, vecs = np.linalg.eigh(S.toarray())
        vals, vecs = vals[-k:], vecs[:, -k:]
    else:
        vals, vecs = eigsh(S, k=k, which="LA", tol=1e-5)
    g = sum(alpha * (1 - alpha) ** t * vals ** t for t in range(order + 1))
    X = vecs * np.sqrt(np.clip(g, 0, None))
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X.astype(np.float32)


def tie_classes(edge_index, n, chunk=4096):
    """Return (edge_type per directed edge, jaccard per directed edge)."""
    A = adjacency(edge_index, n)
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    r, c = edge_index
    lin = r.astype(np.int64) * n + c
    order = np.argsort(lin)
    lin_sorted = lin[order]
    cn = np.zeros(len(r))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        M = (A[s:e] @ A).multiply(A[s:e]).tocoo()
        if M.nnz == 0:
            continue
        key = (M.row.astype(np.int64) + s) * n + M.col
        pos = np.searchsorted(lin_sorted, key)
        cn[order[pos]] = M.data
    jac = cn / np.maximum(deg[r] + deg[c] - cn, 1.0)
    etype = np.zeros(len(r), dtype=np.int64)
    pos = cn > 0
    if pos.any():
        med = np.median(jac[pos])
        etype[pos] = np.where(jac[pos] <= med, 1, 2)
    return etype, jac.astype(np.float32)
