"""Train and evaluate MixFlow-CD.

Examples
    python train.py --dataset cora --runs 5
    python train.py --dataset all --runs 5
    python train.py --dataset football,dolphins --estimate_k 2-20      # K unknown setting
    python train.py --dataset cora --no_ties --tag abl_noties           # ablation
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from data import ALL, expand_names, describe, load_dataset
from features import spectral_diffusion_features, tie_classes
from metrics import evaluate, repair_and_done
from model import (Encoder, MixFlowHead, PIController, collapse_reg, sce_loss,
                   soft_modularity)


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cora", help="name, comma list, or 'all'")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="", help="suffix for the method name (ablations)")
    # architecture
    p.add_argument("--hid", type=int, default=0, help="0 = auto (256 small graphs, 128 large)")
    p.add_argument("--zdim", type=int, default=0, help="0 = auto: max(64, structural feature dim) for non-attributed graphs, else 64")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--n_tokens", type=int, default=0, help="0 = max(K, 16)")
    p.add_argument("--flow_layers", type=int, default=4, help="0 = plain Gaussian mixture on z (ablation)")
    p.add_argument("--struct_dim", type=int, default=0, help="0 = auto: min(max(64, K+16), 512)")
    p.add_argument("--drop", type=float, default=0.1)
    # pre-training
    p.add_argument("--epochs_pre", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--mask", type=float, default=0.5)
    p.add_argument("--lam_link", type=float, default=-1.0,
                   help="link-reconstruction weight in pre-training; -1 = auto: 0.1 for attributed graphs "
                        "(attributes carry no structure), 0 for non-attributed graphs (inputs are already structural)")
    p.add_argument("--max_pos", type=int, default=200000)
    # fine-tuning
    p.add_argument("--epochs_ft", type=int, default=200)
    p.add_argument("--lr_enc_ft", type=float, default=1e-4)
    p.add_argument("--lr_head", type=float, default=5e-3)
    p.add_argument("--lam_rec", type=float, default=0.5)
    p.add_argument("--lam_coll", type=float, default=1.0)
    p.add_argument("--lam_q", type=float, default=1.0, help="base modularity weight")
    p.add_argument("--tau", type=float, default=0, help="responsibility temperature; 0 = zdim")
    p.add_argument("--H_start", type=float, default=0.5)
    p.add_argument("--H_end", type=float, default=0.05)
    p.add_argument("--kp", type=float, default=2.0)
    p.add_argument("--ki", type=float, default=0.05)
    p.add_argument("--static", action="store_true", help="no PI control (fixed temperature and lambda_Q)")
    p.add_argument("--control", default="tau", choices=["tau", "lam"],
                   help="PI controller drives the responsibility temperature (tau) or the modularity weight (lam)")
    p.add_argument("--kp_tau", type=float, default=2.0)
    p.add_argument("--ki_tau", type=float, default=0.05)
    p.add_argument("--encoder", default="attn", choices=["attn", "none"],
                   help="none = mixture-flow head directly on the input features (no GNN encoder)")
    p.add_argument("--no_skip", action="store_true", help="ablation: no linear input skip in the encoder")
    p.add_argument("--select", default="last", choices=["last", "modularity"],
                   help="which fine-tuning epoch to report; both are label-free")
    # ablations / settings
    p.add_argument("--no_ties", action="store_true")
    p.add_argument("--no_proto", action="store_true")
    p.add_argument("--add_struct", action="store_true", help="concat structural features to attributes")
    p.add_argument("--estimate_k", default="", help="K unknown: candidate range like '2-20' or '5,10,20'")
    p.add_argument("--k_criterion", default="modularity", choices=["modularity", "bic"],
                   help="label-free criterion for --estimate_k")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--k_hat", default="", help="K-unknown setting: json file {dataset: K_hat} used instead of the true K")
    p.add_argument("--resume", action="store_true", help="skip seeds already saved for this method")
    p.add_argument("--diag", action="store_true", help="print label-based metrics during training (diagnostics only)")
    return p.parse_args(argv)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def parse_k_range(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def prepare(g, args, device):
    n, ei = g["n"], g["edge_index"]
    if args.struct_dim:
        sd = args.struct_dim
    else:  # enough spectral dimensions to separate K communities (K_max of the range in the K-unknown setting)
        K = max(parse_k_range(args.estimate_k)) if args.estimate_k else (getattr(args, "_K", None) or len(np.unique(g["y"])))
        sd = int(min(max(64, K + 16), 512))
    args._sd = sd
    if g["x"] is None:
        x = spectral_diffusion_features(ei, n, dim=sd)
    else:
        x = g["x"].copy()
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        if args.add_struct:
            x = np.concatenate([x, spectral_diffusion_features(ei, n, dim=sd)], 1)
    etype, _ = tie_classes(ei, n)
    t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt, device=device)
    src, dst = t(ei[0]), t(ei[1])
    deg = torch.bincount(src, minlength=n).float()
    return dict(x=t(x, torch.float32), src=src, dst=dst, etype=t(etype), deg=deg, two_m=float(len(ei[0])),
                n=n, tie_frac=np.bincount(etype, minlength=3) / max(len(etype), 1))


def link_loss(z, src, dst, n, max_pos):
    E = src.shape[0]
    if E == 0:
        return z.sum() * 0
    idx = torch.randint(E, (min(E, max_pos),), device=z.device)
    s, d = src[idx], dst[idx]
    nd = torch.randint(n, (len(idx),), device=z.device)
    zn = F.normalize(z, dim=-1)
    pos = 5.0 * (zn[s] * zn[d]).sum(-1)
    neg = 5.0 * (zn[s] * zn[nd]).sum(-1)
    return F.binary_cross_entropy_with_logits(torch.cat([pos, neg]),
                                              torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))


def kmeans_labels(u, K, seed):
    km = KMeans(K, n_init=10 if K <= 50 else 3, random_state=seed).fit(u.detach().cpu().numpy())
    return torch.as_tensor(km.labels_, device=u.device, dtype=torch.long)


def fit_head_only(z, K, args, seed, edge_index, steps=300):
    """Fit the mixture-flow head on fixed embeddings (used for BIC-based K selection)."""
    head = MixFlowHead(z.shape[1], K, n_layers=args.flow_layers).to(z.device)
    head.init_from_kmeans(z, kmeans_labels(z, K, seed))
    opt = torch.optim.Adam(head.parameters(), lr=args.lr_head)
    for _ in range(steps):
        logp, _, _ = head(z)
        loss = -logp.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logp, gamma, _ = head(z)
    if args.k_criterion == "modularity":  # lower is better, so return -Q of the hard partition
        from metrics import modularity
        pred = gamma.argmax(1).cpu().numpy()
        return -modularity(edge_index, np.unique(pred, return_inverse=True)[1])
    n, d = z.shape
    params = K * 2 * d + (K - 1)  # flow parameters are identical for every K and cancel out
    return -2 * logp.sum().item() + params * np.log(n)


def run_head_only(g, P, args, seed, device, K):
    """Mixture-flow head + soft modularity + PI temperature control, directly on the input features."""
    x = P["x"]
    z = (x - x.mean(0)) / (x.std(0) + 1e-6)
    d = z.shape[1]
    head = MixFlowHead(d, K, n_layers=args.flow_layers).to(device)
    with torch.no_grad():
        head.init_from_kmeans(z, kmeans_labels(z, K, seed))
    opt = torch.optim.Adam(head.parameters(), lr=args.lr_head)
    logK = np.log(K)
    log_tau0 = float(np.log(args.tau)) if args.tau else calibrate_log_tau(head, z, args.H_start, logK)
    log_tau, integ = log_tau0, 0.0
    for ep in range(args.epochs_ft):
        logp, gamma, _ = head(z, float(np.exp(log_tau)))
        Q = soft_modularity(gamma, P["src"], P["dst"], P["deg"], P["two_m"])
        loss = -logp.mean() / d - args.lam_q * Q + args.lam_coll * collapse_reg(gamma)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            H = (-(gamma * torch.log(gamma + 1e-12)).sum(1) / logK).mean().item()
            target = args.H_start + (args.H_end - args.H_start) * ep / max(args.epochs_ft - 1, 1)
            if not args.static:
                integ += H - target
                log_tau = float(np.clip(log_tau0 - args.kp_tau * (H - target) - args.ki_tau * integ,
                                        np.log(0.5), np.log(1e4)))
        if not args.quiet and (ep % 50 == 0 or ep == args.epochs_ft - 1):
            print(f"    head {ep:4d} nll {(-logp.mean() / d).item():.3f} Q {Q.item():.3f} H {H:.3f} "
                  f"(target {target:.3f}) tau {np.exp(log_tau):.2f}")
    with torch.no_grad():
        pred = head(z)[1].argmax(1).cpu().numpy()
    return pred, K


@torch.no_grad()
def calibrate_log_tau(head, z, H_target, logK, lo=np.log(0.5), hi=np.log(1e4), iters=40):
    """Bisection on log(tau) so that the mean normalized assignment entropy equals H_target."""
    u, _ = head.transform(z)
    joint = head.component_loglik(u) + torch.log_softmax(head.logit_pi, 0)

    def H(lt):
        g = torch.softmax(joint / float(np.exp(lt)), 1)
        return (-(g * torch.log(g + 1e-12)).sum(1) / logK).mean().item()

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if H(mid) > H_target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def run_once(g, P, args, seed, device):
    set_seed(seed)
    n, y = P["n"], g["y"]
    K = getattr(args, "_K", None) or len(np.unique(y))
    hid = args.hid or (256 if (n < 5000 and P["two_m"] < 2e5) else 128)
    zdim = args.zdim or (max(64, args._sd) if g["x"] is None else 64)
    if args.encoder == "none":
        return run_head_only(g, P, args, seed, device, K)
    n_tokens = args.n_tokens or max(K, 16)
    enc = Encoder(P["x"].shape[1], hid, zdim, heads=args.heads, layers=args.layers, n_tokens=n_tokens,
                  drop=args.drop, use_ties=not args.no_ties, use_proto=not args.no_proto,
                  use_skip=not args.no_skip).to(device)
    lam_link = args.lam_link if args.lam_link >= 0 else (0.1 if g["x"] is not None else 0.0)
    dec = nn.Sequential(nn.Linear(zdim, hid), nn.PReLU(), nn.Linear(hid, P["x"].shape[1])).to(device)
    mask_token = nn.Parameter(torch.zeros(1, P["x"].shape[1], device=device))
    bn = nn.BatchNorm1d(zdim, affine=False, track_running_stats=False).to(device)  # fixes scale: no collapse
    G = (P["src"], P["dst"], P["etype"])
    x = P["x"]

    # ---------------- stage 1: masked-feature + link pre-training
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + [mask_token], lr=args.lr,
                           weight_decay=args.wd)
    for ep in range(args.epochs_pre):
        enc.train()
        m = torch.rand(n, device=device) < args.mask
        xin = torch.where(m.unsqueeze(1), mask_token.expand_as(x), x)
        z = enc(xin, *G)
        loss = sce_loss(dec(z)[m], x[m]) + lam_link * link_loss(z, P["src"], P["dst"], n, args.max_pos)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if not args.quiet and (ep % 50 == 0 or ep == args.epochs_pre - 1):
            print(f"    pre {ep:4d} loss {loss.item():.4f}")

    # ---------------- optional: K unknown -> BIC on the pre-trained embedding
    if args.estimate_k:
        enc.eval()
        with torch.no_grad():
            z0 = bn(enc(x, *G))
        cands = [k for k in parse_k_range(args.estimate_k) if 2 <= k < n]
        bics = {k: fit_head_only(z0, k, args, seed, g["edge_index"]) for k in cands}
        K = min(bics, key=bics.get)
        if not args.quiet:
            print(f"    {args.k_criterion} score (lower=better):", {k: round(v, 1) for k, v in bics.items()}, "-> K =", K)

    # ---------------- stage 2: mixture-flow fine-tuning with PI-controlled modularity weight
    head = MixFlowHead(zdim, K, n_layers=args.flow_layers).to(device)
    enc.eval()
    with torch.no_grad():
        z0 = bn(enc(x, *G))
        km0 = kmeans_labels(z0, K, seed)
        head.init_from_kmeans(z0, km0)
    if args.diag:
        print("    [diag] k-means on pre-trained z:", {k: round(v, 4) for k, v in evaluate(y, km0.cpu().numpy()).items()})
    opt = torch.optim.Adam([
        {"params": list(enc.parameters()) + list(dec.parameters()), "lr": args.lr_enc_ft, "weight_decay": args.wd},
        {"params": head.parameters(), "lr": args.lr_head},
    ])
    pi = PIController(base=args.lam_q, kp=args.kp, ki=args.ki)
    lam_q, best_q, best_pred = args.lam_q, -1e9, None
    logK = np.log(K)
    if args.tau:
        log_tau0 = float(np.log(args.tau))
    elif args.control == "tau" and not args.static:
        log_tau0 = calibrate_log_tau(head, z0, args.H_start, logK)  # start exactly at the target entropy
    else:
        log_tau0 = float(np.log(zdim))
    log_tau, integ = log_tau0, 0.0
    for ep in range(args.epochs_ft):
        enc.train()
        z = enc(x, *G)
        zb = bn(z)
        logp, gamma, _ = head(zb, float(np.exp(log_tau)))
        Q = soft_modularity(gamma, P["src"], P["dst"], P["deg"], P["two_m"])
        loss = (-logp.mean() / zdim - lam_q * Q + args.lam_coll * collapse_reg(gamma)
                + args.lam_rec * sce_loss(dec(z), x))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 5.0)
        opt.step()
        with torch.no_grad():
            H = (-(gamma * torch.log(gamma + 1e-12)).sum(1) / logK).mean().item()
            target = args.H_start + (args.H_end - args.H_start) * ep / max(args.epochs_ft - 1, 1)
            if not args.static:
                err = H - target
                if args.control == "lam":
                    lam_q = pi.step(err)
                else:  # PI control of log-temperature: too soft (err > 0) -> lower the temperature
                    integ += err
                    log_tau = float(np.clip(log_tau0 - args.kp_tau * err - args.ki_tau * integ,
                                            np.log(0.5), np.log(1e4)))
        if args.select == "modularity" and (ep % 5 == 0 or ep == args.epochs_ft - 1):
            pred, q_hard = predict(enc, bn, head, x, G, g["edge_index"])
            if q_hard > best_q:
                best_q, best_pred = q_hard, pred
        if not args.quiet and (ep % 50 == 0 or ep == args.epochs_ft - 1):
            if args.diag:
                pd_, _ = predict(enc, bn, head, x, G, g["edge_index"])
                print("    [diag]", {k: round(v, 4) for k, v in evaluate(y, pd_).items()})
            print(f"    ft  {ep:4d} nll {(-logp.mean() / zdim).item():.3f} Q {Q.item():.3f} "
                  f"H {H:.3f} (target {target:.3f}) lam_q {lam_q:.2f} tau {np.exp(log_tau):.2f}")
    pred, _ = predict(enc, bn, head, x, G, g["edge_index"])
    if args.select == "modularity" and best_pred is not None:
        pred = best_pred
    return pred, K


@torch.no_grad()
def predict(enc, bn, head, x, G, edge_index):
    from metrics import modularity
    enc.eval()
    _, gamma, _ = head(bn(enc(x, *G)))
    pred = gamma.argmax(1).cpu().numpy()
    return pred, modularity(edge_index, np.unique(pred, return_inverse=True)[1])


def method_name(args):
    name = "MixFlow-CD"
    if args.estimate_k:
        name += "(K-est)"
    return name + (f"_{args.tag}" if args.tag else "")


def main(argv=None):
    args = get_args(argv)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    names = expand_names(args.dataset)
    os.makedirs(args.out, exist_ok=True)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    for name in names:
        g = load_dataset(name, args.root)
        print(describe(g))
        if args.k_hat:
            import json
            args._K = int(json.load(open(args.k_hat))[name])
        P = prepare(g, args, device)
        print(f"  tie classes weak/medium/strong: {np.round(P['tie_frac'], 3)}")
        path = os.path.join(args.out, f"{name}.csv")
        done = repair_and_done(path, method_name(args)) if args.resume else set()
        new = not os.path.exists(path)
        rows = []
        for r in range(args.runs):
            seed = args.seed + r
            if seed in done:
                print(f"  run {r} seed {seed}: already saved, skipped")
                continue
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            pred, K = run_once(g, P, args, seed, device)
            res = evaluate(g["y"], pred, g["edge_index"])
            res.update(time_s=round(time.time() - t0, 1), K_used=K,
                       peak_mem_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20) if device.type == "cuda" else 0)
            rows.append(res)
            print(f"  run {r} seed {seed}: ACC {res['ACC']:.4f} NMI {res['NMI']:.4f} ARI {res['ARI']:.4f} "
                  f"F1 {res['F1']:.4f} Q {res['Q']:.4f} K_pred {res['K_pred']} ({res['time_s']}s)")
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["dataset", "method", "seed", "ACC", "NMI", "ARI", "F1", "Q", "K_true", "K_used",
                                "K_pred", "time_s", "peak_mem_mb"])
                    new = False
                w.writerow([name, method_name(args), seed] + [round(res[k], 4) for k in ["ACC", "NMI", "ARI", "F1", "Q"]]
                           + [len(np.unique(g["y"])), K, res["K_pred"], res["time_s"], res["peak_mem_mb"]])
        if not rows:
            continue
        M = {k: np.array([r[k] for r in rows]) for k in ["ACC", "NMI", "ARI", "F1"]}
        print(f"==> {name} {method_name(args)}: " + "  ".join(f"{k} {v.mean():.4f}±{v.std():.4f}" for k, v in M.items()))
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
