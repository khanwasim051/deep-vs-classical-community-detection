"""Run PyAGC's own benchmark scripts (unchanged) on our datasets, one seed at a time, under our protocol.

PyAGC (Cloudy1225/PyAGC) ships reference implementations and tuned configs for Cora and Photo.
This wrapper only (1) feeds our graph and input features to their script instead of their loader,
(2) captures the predicted labels, and (3) scores them with the same metrics.py as every other method.

    python pyagc_run.py --methods GAE,DMoN --dataset cora --runs 5
    python pyagc_run.py --dataset all --runs 5            # all methods, all 17 datasets

Config used per dataset (PyAGC only tunes Cora and Photo):
    amazon-photo, amazon-computers -> Photo config;   every other dataset -> Cora config.
Results go to results_pyagc/<dataset>.csv with method names like "DMoN[PyAGC]".
"""
import argparse
import json
import csv
import importlib.util
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import expand_names, load_dataset  # noqa: E402
from features import spectral_diffusion_features  # noqa: E402
from metrics import evaluate, repair_and_done  # noqa: E402

BENCH = os.path.join(HERE, "pyagc_benchmark")
METHODS = ["GAE", "DGI", "DAEGC", "DMoN", "DinkNet", "SSGC", "NS4GC", "CCASSG", "MS2CAG", "S3GC", "MAGI"]
PHOTO_LIKE = {"amazon-photo", "amazon-computers"}


def our_inputs(name, root, features="auto", K_hat=None):
    g = load_dataset(name, root)
    n, ei, K = g["n"], g["edge_index"], len(np.unique(g["y"]))
    if K_hat:  # K-unknown setting: nothing below may depend on the true K
        K = int(K_hat)
    if g["x"] is None or features == "structural":  # same structural features as every other method
        x = spectral_diffusion_features(ei, n, dim=int(min(max(64, K + 16), 512)))
    elif features == "l1":  # PyAGC's own preprocessing (torch_geometric NormalizeFeatures: rows sum to one)
        x = g["x"] / np.maximum(g["x"].sum(axis=1, keepdims=True), 1e-12)
    else:
        x = g["x"] / (np.linalg.norm(g["x"], axis=1, keepdims=True) + 1e-12)
    y = torch.as_tensor(g["y"], dtype=torch.float32)
    if K_hat:  # the reference scripts read the number of clusters as max(label) + 1, so pass K_hat that way;
        y = y.clamp(max=K - 1)  # our own metrics always use the true labels
        y[0] = K - 1
    return g, torch.tensor(x, dtype=torch.float32), torch.as_tensor(ei, dtype=torch.long), y


def run_one(method, name, seed, root, device, captured, config="auto", features="auto", K_hat=None):
    g, x, ei, y = our_inputs(name, root, features, K_hat)
    cfg = config if config != "auto" else ("Photo" if name in PHOTO_LIKE else "Cora")
    mdir = os.path.join(BENCH, method)
    spec = importlib.util.spec_from_file_location(f"pyagc_bench_{method}", os.path.join(mdir, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def fake_get_dataset(*a, **k):
        return x.clone(), ei.clone(), y.clone()

    orig_label_metrics = mod.label_metrics

    def capture_label_metrics(y_true, pred, *a, **k):
        captured.append(pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred))
        return orig_label_metrics(y_true, pred, *a, **k)

    mod.get_dataset = fake_get_dataset
    if hasattr(mod, "get_training_config"):  # all our graphs fit in memory: run every configuration full-batch
        orig_gtc = mod.get_training_config

        def full_batch_config(*a, **k):
            conf = dict(orig_gtc(*a, **k))
            conf["mini_batch"] = False
            return conf
        mod.get_training_config = full_batch_config
    mod.label_metrics = capture_label_metrics
    os.makedirs(os.path.join(HERE, "tmp_pyagc"), exist_ok=True)  # keep temporary files off a full C: drive
    tmp = tempfile.mkdtemp(prefix="run_", dir=os.path.join(HERE, "tmp_pyagc"))
    src = open(os.path.join(mdir, "main.py"), encoding="utf-8").read()
    argv = ["main.py", "--dataset", cfg, "--seed", str(seed), "--runs", "1", "--root", tmp,
            "--log_dir", os.path.join(tmp, "logs"), "--device", device]
    for opt in ["ckpt_dir", "cache_dir"]:  # only pass options this script actually has
        if f"'--{opt}'" in src:
            argv += [f"--{opt}", os.path.join(tmp, opt)]
    old_argv, old_cwd = sys.argv, os.getcwd()
    try:
        sys.argv = argv
        os.chdir(mdir)  # their script reads train.conf.yaml from its own folder
        mod.main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--root", default=os.path.join(HERE, "data"))
    p.add_argument("--out", default=os.path.join(HERE, "results_pyagc"))
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--config", default="auto", choices=["auto", "Cora", "Photo", "Physics", "HM"],
                   help="sensitivity check: force one of PyAGC's published configurations for every dataset")
    p.add_argument("--features", default="auto", choices=["auto", "structural", "l1"],
                   help="structural = structural features even on attributed graphs (topology-only control)")
    p.add_argument("--k_hat", default="", help="K-unknown setting: json file {dataset: K_hat}")
    a = p.parse_args()
    khat = json.load(open(a.k_hat)) if a.k_hat else {}
    os.makedirs(a.out, exist_ok=True)
    root = os.path.abspath(a.root)
    for name in expand_names(a.dataset):
        path = os.path.join(os.path.abspath(a.out), f"{name}.csv")
        for method in a.methods.split(","):
            label = f"{method}[PyAGC]" if a.config == "auto" else f"{method}[PyAGC-{a.config}]"
            if a.features == "structural":
                label = label.replace("]", "-struct]")
            if a.features == "l1":
                label = label.replace("]", "-l1]")
            if a.k_hat:
                label += "@Khat"
            done = repair_and_done(path, label)
            for seed in range(a.runs):
                if seed in done:
                    print(f"  {name} {label} seed {seed}: already saved, skipped")
                    continue
                captured = []
                if a.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                try:
                    g = run_one(method, name, seed, root, a.device, captured, a.config, a.features, khat.get(name))
                except torch.cuda.OutOfMemoryError:
                    print(f"  {name} {label}: out of GPU memory, skipped")
                    torch.cuda.empty_cache()
                    break
                except ImportError as ex:  # e.g. S3GC needs pyg-lib, which may not exist for your PyTorch on Windows
                    print(f"  {label}: skipped ({ex})")
                    break
                except (Exception, SystemExit) as ex:  # never let one method stop the whole benchmark
                    print(f"  {name} {label} seed {seed}: FAILED ({type(ex).__name__}: {str(ex)[:200]}) - moving on")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                if not captured:
                    print(f"  {name} {label} seed {seed}: no prediction captured, skipped")
                    continue
                pred = captured[-1]
                res = evaluate(g["y"], pred, g["edge_index"])
                mem = round(torch.cuda.max_memory_allocated() / 2 ** 20) if a.device.startswith("cuda") else 0
                new = not os.path.exists(path)
                with open(path, "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["dataset", "method", "seed", "ACC", "NMI", "ARI", "F1", "Q", "K_true", "K_used",
                                    "K_pred", "time_s", "peak_mem_mb"])
                    K = len(np.unique(g["y"]))
                    w.writerow([name, label, seed] + [round(res[k], 4) for k in ["ACC", "NMI", "ARI", "F1", "Q"]]
                               + [K, int(khat.get(name, K)), res["K_pred"], round(time.time() - t0, 1), mem])
                print(f"  {name} {label} seed {seed}: ACC {res['ACC']:.4f} NMI {res['NMI']:.4f} "
                      f"ARI {res['ARI']:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
