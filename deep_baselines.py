"""Deep graph-clustering baselines under the same protocol (same inputs, same metrics, same CSVs).

    python deep_baselines.py --dataset all --runs 5
    python deep_baselines.py --dataset cora --methods DMoN,GAE

Methods (re-implemented in plain PyTorch, full-batch, sized for a 6 GB GPU):
  GAE    - Kipf & Welling (2016): 2-layer GCN autoencoder (link reconstruction) + k-means on z.
  DMoN   - Tsitsulin et al. (JMLR 2023): GCN + softmax assignment, spectral-modularity loss
           + collapse regularizer.
  DAEGC  - Wang et al. (IJCAI 2019): graph-attention autoencoder pre-training, then joint
           reconstruction + DEC self-training (Student-t soft assignment, sharpened target).
These are faithful-in-spirit re-implementations, not the authors' code; say so in the paper.
Input features are the same as MixFlow-CD (attributes, or the structural spectral features).
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from data import ALL, expand_names, describe, load_dataset
from features import spectral_diffusion_features
from metrics import evaluate, repair_and_done
from model import segment_softmax, soft_modularity, collapse_reg


def gcn_norm(ei, n, device):
    src = torch.as_tensor(ei[0], device=device)
    dst = torch.as_tensor(ei[1], device=device)
    loop = torch.arange(n, device=device)
    src, dst = torch.cat([src, loop]), torch.cat([dst, loop])
    deg = torch.zeros(n, device=device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    w = deg[src].rsqrt() * deg[dst].rsqrt()
    return src, dst, w


def prop(h, src, dst, w):
    return torch.zeros_like(h).index_add_(0, dst, h[src] * w.unsqueeze(1))


class GCN(nn.Module):
    def __init__(self, din, hid, dout):
        super().__init__()
        self.l1, self.l2 = nn.Linear(din, hid), nn.Linear(hid, dout)

    def forward(self, x, A):
        h = F.relu(prop(self.l1(x), *A))
        return prop(self.l2(h), *A)


class GATLayer(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.W = nn.Linear(din, dout, bias=False)
        self.a_s = nn.Parameter(torch.randn(dout) * 0.1)
        self.a_d = nn.Parameter(torch.randn(dout) * 0.1)

    def forward(self, x, src, dst):
        h = self.W(x)
        e = F.leaky_relu((h[src] * self.a_s).sum(-1) + (h[dst] * self.a_d).sum(-1), 0.2).unsqueeze(1)
        a = segment_softmax(e, dst, x.shape[0])
        return torch.zeros_like(h).index_add_(0, dst, h[src] * a)


def recon_loss(z, src, dst, n):
    idx = torch.randint(len(src), (min(len(src), 200000),), device=z.device)
    s, d = src[idx], dst[idx]
    nd = torch.randint(n, (len(idx),), device=z.device)
    pos, neg = (z[s] * z[d]).sum(-1), (z[s] * z[nd]).sum(-1)
    return F.binary_cross_entropy_with_logits(torch.cat([pos, neg]),
                                              torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))


def km(z, K, seed):
    return KMeans(K, n_init=10 if K <= 50 else 3, random_state=seed).fit_predict(z.detach().cpu().numpy())


def run_gae(x, P, K, seed, a):
    m = GCN(x.shape[1], a.hid, a.zdim).to(x.device)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    for _ in range(a.epochs):
        z = m(x, P["A"])
        loss = recon_loss(z, P["src"], P["dst"], P["n"])
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return km(m(x, P["A"]), K, seed)


def run_dmon(x, P, K, seed, a):
    m = GCN(x.shape[1], a.hid, K).to(x.device)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    for _ in range(a.epochs):
        C = torch.softmax(F.dropout(m(x, P["A"]), 0.5, True), 1)
        loss = -soft_modularity(C, P["src"], P["dst"], P["deg"], P["two_m"]) + 1.0 * (collapse_reg(C) + 1) / math.sqrt(K)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(x, P["A"]).argmax(1).cpu().numpy()


def run_daegc(x, P, K, seed, a):
    src, dst = P["src_l"], P["dst_l"]
    l1, l2 = GATLayer(x.shape[1], a.hid).to(x.device), GATLayer(a.hid, a.zdim).to(x.device)
    enc = lambda inp: F.normalize(l2(F.elu(l1(inp, src, dst)), src, dst), dim=1)
    params = list(l1.parameters()) + list(l2.parameters())
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=5e-3)
    for _ in range(a.epochs):  # pre-training: reconstruction only
        loss = recon_loss(5 * enc(x), P["src"], P["dst"], P["n"])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = enc(x)
        lab = km(z, K, seed)
        mu = nn.Parameter(torch.stack([z[torch.as_tensor(lab == k, device=x.device)].mean(0)
                                       if (lab == k).any() else z[0] for k in range(K)]))
    opt = torch.optim.Adam(params + [mu], lr=a.lr, weight_decay=5e-3)
    for ep in range(a.epochs):
        z = enc(x)
        q = 1.0 / (1.0 + torch.cdist(z, mu) ** 2)
        q = q / q.sum(1, keepdim=True)
        if ep % 5 == 0:
            with torch.no_grad():
                p = q ** 2 / q.sum(0)
                p = p / p.sum(1, keepdim=True)
        loss = recon_loss(5 * z, P["src"], P["dst"], P["n"]) + 10 * F.kl_div(q.log(), p, reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = enc(x)
        return (1.0 / (1.0 + torch.cdist(z, mu) ** 2)).argmax(1).cpu().numpy()


METHODS = {"GAE": run_gae, "DMoN": run_dmon, "DAEGC": run_daegc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hid", type=int, default=256)
    p.add_argument("--zdim", type=int, default=64)
    p.add_argument("--resume", action="store_true", help="skip seeds already saved for this method")
    a = p.parse_args()
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                       ("cpu" if a.device == "auto" else a.device))
    names = expand_names(a.dataset)
    os.makedirs(a.out, exist_ok=True)
    for name in names:
        g = load_dataset(name, a.root)
        print(describe(g))
        n, ei, K = g["n"], g["edge_index"], len(np.unique(g["y"]))
        if g["x"] is None:
            x = spectral_diffusion_features(ei, n, dim=int(min(max(64, K + 16), 512)))
        else:
            x = g["x"] / (np.linalg.norm(g["x"], axis=1, keepdims=True) + 1e-12)
        x = torch.as_tensor(x, dtype=torch.float32, device=dev)
        src = torch.as_tensor(ei[0], device=dev)
        dst = torch.as_tensor(ei[1], device=dev)
        loop = torch.arange(n, device=dev)
        P = dict(n=n, src=src, dst=dst, A=gcn_norm(ei, n, dev), deg=torch.bincount(src, minlength=n).float(),
                 two_m=float(len(ei[0])), src_l=torch.cat([src, loop]), dst_l=torch.cat([dst, loop]))
        path = os.path.join(a.out, f"{name}.csv")
        new = not os.path.exists(path)
        for mname in a.methods.split(","):
            res_all = []
            done = repair_and_done(path, mname) if a.resume else set()
            new = not os.path.exists(path)
            for r in range(a.runs):
                if r in done:
                    print(f"  {mname} seed {r}: already saved, skipped")
                    continue
                torch.manual_seed(r); np.random.seed(r)
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                try:
                    pred = METHODS[mname](x, P, K, r, a)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {mname}: out of GPU memory on {name}, skipped")
                    torch.cuda.empty_cache()
                    break
                res = evaluate(g["y"], pred, ei)
                res_all.append(res)
                mem = round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == "cuda" else 0
                with open(path, "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["dataset", "method", "seed", "ACC", "NMI", "ARI", "F1", "Q", "K_true", "K_used",
                                    "K_pred", "time_s", "peak_mem_mb"])
                        new = False
                    w.writerow([name, mname, r] + [round(res[k], 4) for k in ["ACC", "NMI", "ARI", "F1", "Q"]]
                               + [K, K, res["K_pred"], round(time.time() - t0, 1), mem])
            if res_all:
                print(f"  {mname:7s} " + "  ".join(
                    f"{k} {np.mean([q[k] for q in res_all]):.4f}" for k in ["ACC", "NMI", "ARI", "F1"]))


if __name__ == "__main__":
    main()
