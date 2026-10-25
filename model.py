"""MixFlow-CD: tie-aware dual-attention encoder + mixture-prior normalizing-flow clustering head.

Pure PyTorch (no torch_scatter / torch_sparse needed), full-batch, sized for a 6 GB GPU.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- graph utils
def segment_softmax(score, index, n):
    """softmax of score (E,H) over edges that share the same destination index."""
    H = score.shape[1]
    idx = index.unsqueeze(1).expand(-1, H)
    mx = torch.full((n, H), -1e30, device=score.device, dtype=score.dtype)
    mx = mx.scatter_reduce(0, idx, score, reduce="amax", include_self=True)
    ex = torch.exp(score - mx[index])
    den = torch.zeros((n, H), device=score.device, dtype=score.dtype).index_add_(0, index, ex)
    return ex / (den[index] + 1e-16)


# --------------------------------------------------------------------------- encoder
class TieAttention(nn.Module):
    """Edge-masked multi-head attention with tie-class-specific value projections and score biases."""

    def __init__(self, dim, heads, n_ties, attn_drop=0.1):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh, self.R = heads, dim // heads, n_ties
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim * n_ties)  # one value projection per tie class
        self.self_lin = nn.Linear(dim, dim)
        self.tie_bias = nn.Parameter(torch.zeros(n_ties, heads))
        self.attn_drop = attn_drop

    def forward(self, h, src, dst, etype):
        n = h.shape[0]
        q = self.q(h).view(n, self.h, self.dh)
        k = self.k(h).view(n, self.h, self.dh)
        v = self.v(h).view(n, self.R, self.h, self.dh)
        score = (q[dst] * k[src]).sum(-1) / math.sqrt(self.dh) + self.tie_bias[etype]
        alpha = segment_softmax(score, dst, n)
        alpha = F.dropout(alpha, self.attn_drop, self.training)
        msg = v[src, etype] * alpha.unsqueeze(-1)  # (E, H, dh)
        out = torch.zeros((n, self.h, self.dh), device=h.device, dtype=h.dtype).index_add_(0, dst, msg)
        return out.reshape(n, -1) + self.self_lin(h)


class PrototypeAttention(nn.Module):
    """Global context in O(n*M*d): M learnable tokens pool from all nodes, nodes read back from tokens."""

    def __init__(self, dim, n_tokens):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
        self.pool_q, self.pool_k, self.pool_v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.read_q, self.read_k, self.read_v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, h):
        logits = self.pool_k(h) @ self.pool_q(self.tokens).T * self.scale  # (n, M)
        tok = self.tokens + torch.softmax(logits, dim=0).T @ self.pool_v(h)  # (M, d)
        a = torch.softmax(self.read_q(h) @ self.read_k(tok).T * self.scale, dim=1)  # (n, M)
        return self.norm(h + self.out(a @ self.read_v(tok)))


class Encoder(nn.Module):
    def __init__(self, in_dim, hid, out_dim, heads=4, n_ties=3, layers=2, n_tokens=16, drop=0.1, attn_drop=0.1,
                 use_ties=True, use_proto=True, use_skip=True):
        super().__init__()
        self.inp = nn.Linear(in_dim, hid)
        # linear input skip: the encoder can add structure-aware information but cannot lose
        # what is already in the input features
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if use_skip else None
        self.use_ties, self.use_proto = use_ties, use_proto
        self.layers = nn.ModuleList([TieAttention(hid, heads, n_ties if use_ties else 1, attn_drop) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.proto = PrototypeAttention(hid, n_tokens) if use_proto else None
        self.out = nn.Linear(hid, out_dim)
        self.drop = drop

    def forward(self, x, src, dst, etype):
        if not self.use_ties:
            etype = torch.zeros_like(etype)
        x0 = x
        h = self.inp(F.dropout(x, self.drop, self.training))
        for layer, norm in zip(self.layers, self.norms):
            h = h + F.dropout(F.elu(norm(layer(h, src, dst, etype))), self.drop, self.training)
        if self.proto is not None:
            h = self.proto(h)
        out = self.out(h)
        return out + self.skip(x0) if self.skip is not None else out


def sce_loss(x, y, gamma=2.0):
    """scaled cosine error (GraphMAE)."""
    return (1 - F.cosine_similarity(x, y, dim=-1)).pow(gamma).mean()


# --------------------------------------------------------------------------- flow + mixture head
class AffineCoupling(nn.Module):
    def __init__(self, dim, hid, flip, max_scale=2.0):
        super().__init__()
        self.d1 = dim // 2
        self.flip = flip
        din, dout = (dim - self.d1, self.d1) if flip else (self.d1, dim - self.d1)
        self.net = nn.Sequential(nn.Linear(din, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, 2 * dout))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)  # starts as the identity map
        self.max_scale = max_scale

    def forward(self, x):
        a, b = x[:, : self.d1], x[:, self.d1:]
        if self.flip:
            a, b = b, a
        s, t = self.net(a).chunk(2, dim=-1)
        s = self.max_scale * torch.tanh(s / self.max_scale)  # bounded log-scale -> bounded log|det|
        b = b * torch.exp(s) + t
        out = torch.cat([b, a], -1) if self.flip else torch.cat([a, b], -1)
        return out, s.sum(-1)


class MixFlowHead(nn.Module):
    """u = f(z) with RealNVP couplings; p(u) = sum_k pi_k N(u; mu_k, diag sigma_k^2).

    log p(z) = logsumexp_k[log pi_k + log N(f(z); mu_k, Sigma_k)] + log|det df/dz|   (exact)
    gamma_ik = posterior responsibility = community membership.
    """

    def __init__(self, dim, K, n_layers=4, hid=128, min_var=1e-2):
        super().__init__()
        self.flows = nn.ModuleList([AffineCoupling(dim, hid, flip=(i % 2 == 1)) for i in range(n_layers)])
        self.mu = nn.Parameter(torch.zeros(K, dim))
        self.raw_var = nn.Parameter(torch.zeros(K, dim))
        self.logit_pi = nn.Parameter(torch.zeros(K))
        self.min_var = min_var
        self.dim, self.K = dim, K

    def var(self):
        return F.softplus(self.raw_var) + self.min_var

    def transform(self, z):
        logdet = torch.zeros(z.shape[0], device=z.device)
        u = z
        for f in self.flows:
            u, ld = f(u)
            logdet = logdet + ld
        return u, logdet

    def component_loglik(self, u):
        """(n, K) log N(u_i; mu_k, Sigma_k) without building an (n, K, d) tensor."""
        var = self.var()
        inv = 1.0 / var
        quad = (u * u) @ inv.T - 2 * u @ (self.mu * inv).T + (self.mu * self.mu * inv).sum(1)
        return -0.5 * (quad + torch.log(var).sum(1) + self.dim * math.log(2 * math.pi))

    def forward(self, z, tau=1.0):
        """tau > 1 tempers the responsibilities. In d dimensions the exact posterior is almost always
        one-hot, which kills the modularity gradient; tau = d gives per-dimension (soft) responsibilities.
        argmax, and therefore the predicted partition, does not depend on tau."""
        u, logdet = self.transform(z)
        joint = self.component_loglik(u) + torch.log_softmax(self.logit_pi, 0)
        logp = torch.logsumexp(joint, 1) + logdet
        gamma = torch.softmax(joint / tau, 1)
        return logp, gamma, u

    @torch.no_grad()
    def init_from_kmeans(self, u, labels):
        K = self.K
        for k in range(K):
            m = labels == k
            if m.sum() == 0:
                self.mu[k] = u[torch.randint(len(u), (1,))].squeeze(0)
                continue
            uk = u[m]
            self.mu[k] = uk.mean(0)
            v = uk.var(0, unbiased=False) if m.sum() > 1 else torch.ones_like(uk[0])
            v = torch.clamp(v, min=self.min_var * 2)
            self.raw_var[k] = torch.log(torch.expm1(v - self.min_var))  # inverse softplus
        counts = torch.bincount(labels, minlength=K).float() + 1.0
        self.logit_pi.copy_(torch.log(counts / counts.sum()))


def soft_modularity(gamma, src, dst, deg, two_m):
    """Q = 1/(2m) [ sum_ij A_ij g_i.g_j - (d^T G)(G^T d)/(2m) ] on directed (both-way) edge list."""
    intra = (gamma[src] * gamma[dst]).sum()
    dg = deg @ gamma
    return (intra - (dg * dg).sum() / two_m) / two_m


def collapse_reg(gamma):
    """DMoN collapse regularizer: 0 for equal-size clusters, sqrt(K)-1 when all nodes fall in one cluster."""
    n, K = gamma.shape
    return torch.linalg.norm(gamma.sum(0)) / n * math.sqrt(K) - 1.0


class PIController:
    """ControlVAE-style PI controller for the modularity weight.

    Tracks a decreasing target for the mean normalized assignment entropy: if assignments are
    softer than the target, the modularity weight goes up (modularity favours crisp partitions).
    """

    def __init__(self, base=1.0, kp=2.0, ki=0.05, lo=0.0, hi=10.0):
        self.base, self.kp, self.ki, self.lo, self.hi = base, kp, ki, lo, hi
        self.integral = 0.0

    def step(self, err):
        raw = self.base + self.kp * err + self.ki * (self.integral + err)
        if self.lo < raw < self.hi:  # anti-windup: integrate only when not saturated
            self.integral += err
        return float(min(max(raw, self.lo), self.hi))
