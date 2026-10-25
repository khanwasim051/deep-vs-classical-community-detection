REM Step 10: sensitivity check. Rerun the PyAGC methods with the OTHER published configuration (Photo instead of Cora)
REM on 8 graphs, 5 seeds each. Results go to results_sens. Safe to rerun: finished runs are skipped.
python run_each.py pyagc_run.py --dataset cora,citeseer,email-eu-core,football,kki,lfr1,lfr2,lfr6 --runs 5 --split-methods --methods CCASSG,DGI,NS4GC,MS2CAG,SSGC,DinkNet,GAE,DAEGC,DMoN --config Photo --out results_sens
python check_coverage.py --results results_sens > coverage_sens.txt
python stats.py --results results_sens > stats_sens.txt
type coverage_sens.txt
pause
