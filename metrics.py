import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


def hungarian_map(y, pred):
    """Map predicted cluster ids to labels (works when #clusters != #classes; unmatched clusters -> -1)."""
    ty, tp = np.unique(y), np.unique(pred)
    C = np.zeros((len(tp), len(ty)), dtype=np.int64)
    pi = {p: i for i, p in enumerate(tp)}
    yi = {c: j for j, c in enumerate(ty)}
    for a, b in zip(pred, y):
        C[pi[a], yi[b]] += 1
    r, c = linear_sum_assignment(-C)
    mapping = {tp[i]: ty[j] for i, j in zip(r, c)}
    return np.array([mapping.get(p, -1) for p in pred])


def modularity(edge_index, pred):
    src, dst = edge_index
    two_m = len(src)
    deg = np.bincount(src, minlength=len(pred)).astype(float)
    intra = (pred[src] == pred[dst]).sum()
    K = pred.max() + 1
    dk = np.bincount(pred, weights=deg, minlength=K)
    return float(intra / two_m - ((dk / two_m) ** 2).sum())


def evaluate(y, pred, edge_index=None):
    y, pred = np.asarray(y), np.asarray(pred)
    mapped = hungarian_map(y, pred)
    out = dict(
        ACC=float((mapped == y).mean()),
        NMI=float(normalized_mutual_info_score(y, pred)),
        ARI=float(adjusted_rand_score(y, pred)),
        F1=float(f1_score(y, mapped, labels=np.unique(y), average="macro", zero_division=0)),
        K_pred=int(len(np.unique(pred))),
    )
    if edge_index is not None:
        out["Q"] = modularity(edge_index, np.unique(pred, return_inverse=True)[1])
    return out


def repair_and_done(path, method):
    """Drop broken lines from a results CSV (e.g. cut off by a power loss) and return the seeds of
    `method` that are already complete, so a re-run can skip them."""
    import csv
    import os
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return set()
    header, good = rows[0], [rows[0]]
    for r in rows[1:]:
        if len(r) != len(header):
            continue
        try:
            [float(v) for v in r[3:]]
        except ValueError:
            continue
        good.append(r)
    with open(path, "w", newline="") as f:  # always rewrite: also restores a missing final newline
        csv.writer(f).writerows(good)
    if len(good) != len(rows):
        print(f"  repaired {path}: removed {len(rows) - len(good)} broken line(s)")
    return {int(r[2]) for r in good[1:] if r[1] == method}
