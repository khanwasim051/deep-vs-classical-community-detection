REM Step 8: PyAGC reference methods on the mu sweep (1 seed per graph) and on LFR-6/7 (5 seeds)
REM Run from the (mixflow) prompt in D:\mixflow_cd. Safe to stop and restart: finished runs are skipped.
python run_each.py pyagc_run.py --dataset sweep --runs 1 --split-methods --methods SSGC,MS2CAG,NS4GC,CCASSG,GAE,DAEGC --out results_sweep_pyagc
python run_each.py pyagc_run.py --dataset lfr6,lfr7 --runs 5 --split-methods
python sweep_plot.py --results results_sweep,results_sweep_pyagc --set reference > sweep_ref.txt
python check_coverage.py --results results_sweep_pyagc --runs 1 > coverage_sweep.txt
python check_coverage.py --results results_pyagc > coverage.txt
type coverage_sweep.txt
pause
