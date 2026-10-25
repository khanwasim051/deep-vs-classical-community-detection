"""Run PyAGC's reference scripts exactly as published (their own data loader, feature normalization and seeding:
one pre-training with --seed, then --runs fine-tuning runs) and record their own summary line.

    python pyagc_native.py --methods DAEGC,DinkNet --datasets Cora,Photo --seeds 0,1,2,3,4 > native.txt

Each call of the reference script reports mean ± s.d. over its --runs runs; we repeat it for several --seed values.
"""
import argparse, csv, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--methods", default="MS2CAG,SSGC,DGI,CCASSG,NS4GC,DinkNet,GAE,DAEGC,DMoN")
p.add_argument("--datasets", default="Cora,Photo")
p.add_argument("--seeds", default="0")
p.add_argument("--runs", type=int, default=5)
p.add_argument("--device", default="cuda:0")
p.add_argument("--out", default="results_native.csv")
a = p.parse_args()
root = os.path.join(HERE, "data_pyagc")
pat = re.compile(r"NMI=([\d.]+)\S*?±([\d.]+), ARI=([\d.]+)\S*?±([\d.]+), ACC=([\d.]+)\S*?±([\d.]+)")
run_pat = re.compile(r"Run \d+: NMI=([\d.]+), ARI=([\d.]+), ACC=([\d.]+)")  # scripts without a summary line (DMoN)
new = not os.path.exists(a.out)
with open(a.out, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["dataset", "method", "seed", "runs", "ACC", "ACC_sd", "NMI", "NMI_sd", "ARI", "ARI_sd", "time_s"])
    for ds in a.datasets.split(","):
        for m in a.methods.split(","):
            for s in a.seeds.split(","):
                t0 = time.time()
                tmp = os.path.join(HERE, "tmp_pyagc", f"native_{m}_{ds}_{s}")
                cmd = [sys.executable, "main.py", "--dataset", ds, "--seed", s, "--runs", str(a.runs), "--root", root,
                       "--log_dir", os.path.join(tmp, "logs"), "--device", a.device]
                src = open(os.path.join(HERE, "pyagc_benchmark", m, "main.py"), encoding="utf-8").read()
                for opt in ["ckpt_dir", "cache_dir"]:
                    if f"'--{opt}'" in src:
                        cmd += [f"--{opt}", os.path.join(tmp, opt)]
                env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
                r = subprocess.run(cmd, cwd=os.path.join(HERE, "pyagc_benchmark", m), capture_output=True, text=True, env=env,
                                   encoding="utf-8", errors="replace")
                hits = pat.findall(r.stdout + r.stderr)
                if not hits:
                    per = [[float(x) / 100 for x in h] for h in run_pat.findall(r.stdout + r.stderr)]
                    if per:
                        import numpy as np
                        a_ = np.array(per)  # columns NMI, ARI, ACC
                        hits = [[f"{v * 100:.4f}" for v in (a_[:, 0].mean(), a_[:, 0].std(), a_[:, 1].mean(), a_[:, 1].std(), a_[:, 2].mean(), a_[:, 2].std())]]
                if not hits:
                    print(f"{ds} {m} seed {s}: no result; last lines:", (r.stdout + r.stderr).strip().splitlines()[-3:], flush=True)
                    continue
                nmi, nmis, ari, aris, acc, accs = [float(x) / 100 for x in hits[-1]]
                w.writerow([ds, m, s, a.runs, acc, accs, nmi, nmis, ari, aris, round(time.time() - t0, 1)]); f.flush()
                print(f"{ds} {m} seed {s}: ACC {acc:.4f}±{accs:.4f} NMI {nmi:.4f} ARI {ari:.4f} ({time.time() - t0:.0f}s)", flush=True)
