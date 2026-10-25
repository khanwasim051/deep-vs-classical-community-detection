"""Dataset loaders for the 10 real-world + LFR benchmark suite.

Every loader returns a dict:
    name, n, edge_index (2, 2M) int64 undirected (both directions, no self loops, no duplicates),
    x (n, d) float32 or None (None = non-attributed), y (n,) int64 labels 0..K-1
"""
import gzip
import os
import urllib.request
import warnings

import networkx as nx
import numpy as np

MIRROR = "https://raw.githubusercontent.com/vlivashkin/community-graphs/master/gml_connected_subgraphs/{}.gml"
UK_URL = "https://raw.githubusercontent.com/igraph/igraphdata/master/data/UKfaculty.rda"
SNAP_EMAIL = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
SNAP_EMAIL_LAB = "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"

NON_ATTRIBUTED = ["email-eu-core", "football", "dolphins", "ukfaculty", "kki"]
ATTRIBUTED = ["amazon-computers", "amazon-photo", "citeseer", "cora", "pubmed"]
# added for external validity (reviewer concern 5): more attributed graphs, including low-homophily ones
ATTRIBUTED_EXTRA = ["wiki", "blogcatalog", "chameleon", "squirrel", "actor", "coauthor-cs"]

# LFR settings. N, average degree, max degree, tau1, tau2 and mu follow the
# (N, d_avg, d_max, tau1, tau2, mu) tuples of the benchmark suite. Community-size
# bounds are not specified there, so they are chosen to give roughly the same
# number of communities; the actual count is printed when a graph is generated.
LFR = {
    "lfr1": dict(n=2048, average_degree=30, max_degree=200, tau1=3.0, tau2=1.5, mu=0.4, min_community=40, max_community=200),
    "lfr2": dict(n=5000, average_degree=20, max_degree=50, tau1=3.0, tau2=2.0, mu=0.5, min_community=25, max_community=60),
    "lfr3": dict(n=5000, average_degree=20, max_degree=50, tau1=3.0, tau2=2.0, mu=0.2, min_community=25, max_community=60),
    "lfr4": dict(n=2048, average_degree=30, max_degree=200, tau1=3.0, tau2=1.5, mu=0.2, min_community=40, max_community=200),
    "lfr5": dict(n=20000, average_degree=10, max_degree=50, tau1=2.5, tau2=1.5, mu=0.3, min_community=20, max_community=100),
    # harder settings (the mu=0.2 graphs saturate at 1.0 for most methods)
    "lfr6": dict(n=5000, average_degree=20, max_degree=50, tau1=3.0, tau2=2.0, mu=0.6, min_community=25, max_community=60),
    "lfr7": dict(n=5000, average_degree=20, max_degree=50, tau1=3.0, tau2=2.0, mu=0.7, min_community=25, max_community=60),
}

ALL = NON_ATTRIBUTED + ATTRIBUTED + ["lfr1", "lfr2", "lfr3", "lfr4", "lfr5"]

# mixing-parameter sweep: LFR graphs with the LFR-2 settings (n=5000, avg degree 20, max degree 50,
# tau1=3, tau2=2, community sizes 25-60), mu = 0.1 ... 0.8, three independent graphs per mu.
SWEEP_MUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SWEEP_GRAPHS = [1, 2, 3]
SWEEP = [f"mu{m:.1f}g{s}" for m in SWEEP_MUS for s in SWEEP_GRAPHS]
for _m in SWEEP_MUS:
    for _s in SWEEP_GRAPHS:
        LFR[f"mu{_m:.1f}g{_s}"] = dict(n=5000, average_degree=20, max_degree=50, tau1=3.0, tau2=2.0, mu=_m,
                                       min_community=25, max_community=60, _seed=1000 * _s + int(round(_m * 10)))


# structural heterogeneity: LFR graphs at mu = 0.4 with different degree (tau1) and community-size (tau2) exponents,
# three independent graphs per setting (n = 5000, average degree 20, max degree 100, community sizes 20-200)
HET = []
for _t1 in (2.0, 2.5, 3.0):
    for _t2 in (1.1, 2.0):
        for _s in (1, 2, 3):
            _nm = f"het_t{_t1:g}_c{_t2:g}_g{_s}"
            HET.append(_nm)
            LFR[_nm] = dict(n=5000, average_degree=20, max_degree=100, tau1=_t1, tau2=_t2, mu=0.4,
                            min_community=20, max_community=200, _seed=5000 + 100 * _s + int(_t1 * 10) + int(_t2 * 10))
# more independent realizations of the mu sweep (graphs 4-6 per mu)
SWEEP_MORE = [f"mu{m:.1f}g{s}" for m in SWEEP_MUS for s in (4, 5, 6)]
for _m in SWEEP_MUS:
    for _s in (4, 5, 6):
        LFR[f"mu{_m:.1f}g{_s}"] = dict(n=5000, average_degree=20, max_degree=50, tau1=3.0, tau2=2.0, mu=_m,
                                       min_community=25, max_community=60, _seed=1000 * _s + int(round(_m * 10)))


def expand_names(arg):
    """'all', 'sweep', or a comma-separated list (which may itself contain 'sweep')."""
    if arg == "all":
        return list(ALL)
    out = []
    for s in arg.split(","):
        s = s.strip()
        out += {"all": list(ALL), "sweep": SWEEP, "sweep_more": SWEEP_MORE, "het": HET, "attr_extra": ATTRIBUTED_EXTRA}.get(s, [s])
    return out


def _download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)
    return path


def _finalize(name, n, edges, y, x=None):
    """edges: (M,2) array, possibly directed/duplicated/self-loops."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    ei = np.concatenate([edges.T, edges[:, ::-1].T], axis=1)
    y = np.unique(np.asarray(y), return_inverse=True)[1].astype(np.int64)
    if x is not None:
        x = np.asarray(x, dtype=np.float32)
    return dict(name=name, n=int(n), edge_index=ei, x=x, y=y)


def _from_nx(name, G, label_attr):
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    y = [G.nodes[v][label_attr] for v in nodes]
    return _finalize(name, len(nodes), edges, y)


def _gml_mirror(name, fname, root):
    path = _download(MIRROR.format(fname), os.path.join(root, "raw", fname + ".gml"))
    G = nx.read_gml(path, label="id")
    return _from_nx(name, G, "gt")


def load_email(root):
    try:
        e = _download(SNAP_EMAIL, os.path.join(root, "raw", "email-Eu-core.txt.gz"))
        lab = _download(SNAP_EMAIL_LAB, os.path.join(root, "raw", "email-Eu-core-department-labels.txt.gz"))
        with gzip.open(e, "rt") as f:
            edges = np.loadtxt(f, dtype=np.int64)
        with gzip.open(lab, "rt") as f:
            L = np.loadtxt(f, dtype=np.int64)
        n = int(max(L[:, 0].max(), edges.max()) + 1)
        y = np.zeros(n, dtype=np.int64)
        y[L[:, 0]] = L[:, 1]
        return _finalize("email-eu-core", n, edges, y)
    except Exception as ex:  # SNAP unreachable -> largest connected component mirror (986 nodes)
        warnings.warn(f"SNAP download failed ({ex}); using the largest-connected-component mirror (986 nodes).")
        return _gml_mirror("email-eu-core", "eu-core", root)


def load_ukfaculty(root):
    import rdata

    path = _download(UK_URL, os.path.join(root, "raw", "UKfaculty.rda"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.conversion.convert(rdata.parser.parse_file(path))["UKfaculty"]
    n = int(obj[0][0])
    src, dst = obj[2].astype(np.int64), obj[3].astype(np.int64)  # igraph stores 0-based ids
    group = np.asarray(obj[8][2]["Group"]).astype(np.int64)
    keep = np.where(group != 4)[0]  # group 4 has only 2 members -> 79 nodes, 3 groups
    remap = -np.ones(n, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    m = (remap[src] >= 0) & (remap[dst] >= 0)
    edges = np.stack([remap[src[m]], remap[dst[m]]], 1)
    return _finalize("ukfaculty", len(keep), edges, group[keep])


def load_kki(root):
    """KKI (TUDataset) = 83 brain graphs; the 'communities' are the 83 graphs (disjoint union)."""
    from torch_geometric.datasets import TUDataset

    ds = TUDataset(os.path.join(root, "tu"), "KKI")
    edges, ys, off = [], [], 0
    for gi, g in enumerate(ds):
        edges.append(g.edge_index.numpy().T + off)
        ys.append(np.full(g.num_nodes, gi))
        off += g.num_nodes
    return _finalize("kki", off, np.concatenate(edges), np.concatenate(ys))


def load_pyg(name, root):
    from torch_geometric.datasets import Amazon, Planetoid

    if name.startswith("amazon"):
        sub = "computers" if name.endswith("computers") else "photo"
        fname = f"amazon_electronics_{sub}.npz"
        raw = os.path.join(root, "amazon", sub.capitalize(), "raw", fname)
        try:  # pre-fetch from raw.githubusercontent (github.com redirects are blocked on some networks)
            _download("https://raw.githubusercontent.com/shchur/gnn-benchmark/master/data/npz/" + fname, raw)
        except Exception:
            pass  # PyG will try its own URL
        ds = Amazon(os.path.join(root, "amazon"), sub.capitalize())
    else:
        ds = Planetoid(os.path.join(root, "planetoid"), {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name])
    d = ds[0]
    return _finalize(name, d.num_nodes, d.edge_index.numpy().T, d.y.numpy(), d.x.numpy())


def load_extra(name, root):
    """Additional attributed graphs from PyTorch Geometric (homophilous and heterophilous)."""
    import torch_geometric.datasets as D
    if name in ("wiki", "blogcatalog"):
        ds = D.AttributedGraphDataset(os.path.join(root, "attributed"), {"wiki": "Wiki", "blogcatalog": "BlogCatalog"}[name])
    elif name in ("chameleon", "squirrel"):
        ds = D.WikipediaNetwork(os.path.join(root, "wikipedia"), name, geom_gcn_preprocess=True)
    elif name == "actor":
        ds = D.Actor(os.path.join(root, "actor"))
    elif name == "coauthor-cs":
        fname = "ms_academic_cs.npz"
        raw = os.path.join(root, "coauthor", "CS", "raw", fname)
        try:
            _download("https://raw.githubusercontent.com/shchur/gnn-benchmark/master/data/npz/" + fname, raw)
        except Exception:
            pass
        ds = D.Coauthor(os.path.join(root, "coauthor"), "CS")
    else:
        raise ValueError(name)
    d = ds[0]
    x = d.x.to_dense().numpy() if d.x.is_sparse else d.x.numpy()
    return _finalize(name, d.num_nodes, d.edge_index.numpy().T, d.y.numpy(), x)


def homophily(g):
    """Edge homophily: fraction of edges whose endpoints share a label."""
    r, c = g["edge_index"]
    return float((g["y"][r] == g["y"][c]).mean())


def load_lfr(name, root, seed=42):
    path = os.path.join(root, "lfr", f"{name}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return _finalize(name, int(z["n"]), z["edges"], z["y"])
    cfg = dict(LFR[name])
    n = cfg.pop("n")
    seed = cfg.pop("_seed", seed)
    last = None
    for s in range(seed, seed + 20):
        try:
            print(f"  generating {name} (seed {s}) ... this can take a few minutes for large N")
            G = nx.LFR_benchmark_graph(n, seed=s, max_iters=5000, **cfg)
            break
        except Exception as ex:
            last = ex
    else:
        raise RuntimeError(f"LFR generation failed for {name}: {last}")
    comm = {}
    for v in G.nodes():
        comm.setdefault(frozenset(G.nodes[v]["community"]), len(comm))
    y = np.array([comm[frozenset(G.nodes[v]["community"])] for v in range(n)])
    edges = np.array(list(G.edges()), dtype=np.int64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, n=n, edges=edges, y=y)
    print(f"  {name}: {n} nodes, {len(edges)} edges, {len(comm)} communities")
    return _finalize(name, n, edges, y)


def load_dataset(name, root="data"):
    name = name.lower()
    if name == "email-eu-core":
        return load_email(root)
    if name == "football":
        return _gml_mirror("football", "football", root)
    if name == "dolphins":
        return _gml_mirror("dolphins", "dolphins", root)
    if name == "ukfaculty":
        return load_ukfaculty(root)
    if name == "kki":
        return load_kki(root)
    if name in ATTRIBUTED:
        return load_pyg(name, root)
    if name in ATTRIBUTED_EXTRA:
        return load_extra(name, root)
    if name in LFR:
        return load_lfr(name, root)
    raise ValueError(f"unknown dataset {name}; choose from {ALL + ['lfr6', 'lfr7']}")


def describe(g):
    K = len(np.unique(g["y"]))
    feat = "none (structural)" if g["x"] is None else f"{g['x'].shape[1]}-d attributes"
    return f"{g['name']}: n={g['n']}, undirected edges={g['edge_index'].shape[1] // 2}, K={K}, features={feat}, homophily={homophily(g):.2f}"
