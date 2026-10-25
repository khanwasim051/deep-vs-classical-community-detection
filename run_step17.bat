REM Step 17: reproducibility (reviewer concern 6). About 1-2 hours.
REM Run from the (mixflow) prompt in D:\mixflow_cd, with internet access (PyAGC's own loader downloads Cora and Photo).

REM 1. Exact environment for the paper and the repository
python env_report.py > env_report.txt
call conda env export --no-builds > environment.yml
pip freeze > requirements_frozen.txt

REM 2. Reproduction check of the reference implementations on Cora and Amazon-Photo
REM    (a) our input (rows scaled to unit length) and (b) PyAGC's own input (rows sum to one): 10 seeds for DAEGC and DinkNet
python run_each.py pyagc_run.py --dataset cora,amazon-photo --runs 10 --split-methods --methods DAEGC,DinkNet --out results_repro
python run_each.py pyagc_run.py --dataset cora,amazon-photo --runs 10 --split-methods --methods DAEGC,DinkNet --features l1 --out results_repro
REM    (c) the other seven methods with PyAGC's own input, 5 seeds
python run_each.py pyagc_run.py --dataset cora,amazon-photo --runs 5 --split-methods --methods MS2CAG,SSGC,DGI,CCASSG,NS4GC,GAE,DMoN --features l1 --out results_repro
REM    (d) the reference scripts exactly as published (their own loader, normalization and seeding), seeds 0-2
python pyagc_native.py --methods MS2CAG,SSGC,DGI,CCASSG,NS4GC,DinkNet,GAE,DAEGC,DMoN --datasets Cora,Photo --seeds 0,1,2 > native.txt

REM 3. One file with every run
python export_all.py

REM 4. Package everything for the public repository (code, LFR graphs, per-run results, analysis, environment)
python make_release.py
echo Done. Send: env_report.txt, environment.yml, requirements_frozen.txt, all_runs.csv, results_native.csv, native.txt
pause
