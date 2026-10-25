# Reproducing "Do deep graph clustering methods beat classical community detection?"

This repository contains everything needed to reproduce the paper: the code of every method and baseline, the
unchanged PyAGC reference scripts, the generated LFR graphs, every per-run result, and the analysis that turns
the results into the paper's tables and figures.

## 1. Environment

Tested on Windows with an NVIDIA RTX 3050 Laptop GPU (6 GB) and 16 GB of RAM; see `env_report.txt` for the
exact CPU, driver and library versions.

```
conda env create -f environment.yml        # or: conda create -n mixflow python=3.10; pip install -r requirements_frozen.txt
conda activate mixflow
```

## 2. Data

Public graphs are downloaded by `data.py` on first use (SNAP, the Network Repository mirrors, TU datasets, PyTorch
Geometric). The 72 LFR graphs used in the paper are included in `data/lfr/*.npz`; `data.py` loads them instead of
regenerating them, because the NetworkX generator's output depends on its version. `python graph_stats.py --dataset all`
prints the statistics of Table 3.

## 3. Runs

Every experiment is one batch file; each runs every job in a fresh process, writes one CSV row per run, and can be
stopped and restarted (finished runs are skipped). The 60-minute time limit per job is set in `run_each.py`.

| Batch file | Experiments | Paper |
|---|---|---|
| `run_all.bat`, `run_pyagc.bat` | MixFlow-CD, classical methods and the nine reference methods on the original graphs, 5 seeds | Sections 5.1–5.5 |
| `run_step8.bat`, `run_step9.bat` | LFR-6/7, Infomap, first 24 sweep graphs (reference methods) | Sections 5, 6.1 |
| `run_sweep.bat`, `run_step11.bat`, `run_step13.bat` | first 24 sweep graphs (classical, Photo configuration, controls) | Section 6.1 |
| `run_step10.bat`, `run_step14.bat` | the four candidate configurations of every tunable method | Section 4.4, Tables 17–19 |
| `run_step12.bat` | known-K, matched-input, structural-input and unknown-K controls | Section 5.6, Tables 10–12 |
| `run_step15.bat` | six further attributed graphs; classical methods on the 42 new LFR graphs | Sections 5, 6.1, 6.2 |
| `run_step16.bat` | deep methods on the 66 synthetic graphs | Sections 6.1, 6.2 |
| `run_step17.bat` | environment report; reproduction check of the reference code | Section 4.8, Table 4 |

Amazon-Computers and the 20,000-node LFR-5 graph appear in the early runs but were excluded from the paper (Section 4.1).

`python export_all.py` collects every result folder into `all_runs.csv`, which is included.

## 4. Analysis

```
python analysis/run_all.py
```

recomputes every table and figure from `all_runs.csv` (see `analysis/README.md` for the mapping).

## 5. Licence

Our code is released under the MIT licence (`LICENSE`). The scripts in `pyagc_benchmark/` are the unchanged
reference implementations of PyAGC (https://github.com/Cloudy1225/PyAGC), MIT licence, Copyright (c) 2026 Yunhui Liu.
