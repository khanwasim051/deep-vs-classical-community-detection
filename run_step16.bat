REM Step 16: deep methods on the synthetic graphs (reviewer concerns 4 and 5). Run AFTER Step 15 has finished.
REM Each method uses ONE configuration, chosen without labels: the candidate with the highest mean modularity
REM on the six benchmark LFR graphs in Step 14. One seed per graph, as in the original sweep.
REM Run from the (mixflow) prompt in D:\mixflow_cd. Long run (about 6-8 hours): start it in the evening. Safe to stop and rerun.

REM 0. Statistics of the 15 original graphs (homophily, components) for the datasets table
python graph_stats.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4 > graph_stats_base.txt

REM 1. Original 24-graph sweep: the selected configurations that are not run yet (incl. DGI, which was missing)
python run_each.py pyagc_run.py --dataset sweep --runs 1 --split-methods --methods MS2CAG,CCASSG --config Photo --out results_syn
python run_each.py pyagc_run.py --dataset sweep --runs 1 --split-methods --methods SSGC,NS4GC --config HM --out results_syn
python run_each.py pyagc_run.py --dataset sweep --runs 1 --split-methods --methods DGI --out results_syn
python run_each.py baselines.py --dataset sweep --methods RegSpectral_t0.25 --runs 1 --resume --out results_syn
python run_each.py train.py --dataset sweep --runs 1 --quiet --resume --tag cfgC --lam_q 2.0 --out results_syn

REM 2. 24 extra sweep graphs and 18 heterogeneous LFR graphs: every deep method with its selected configuration
python run_each.py pyagc_run.py --dataset sweep_more,het --runs 1 --split-methods --methods MS2CAG,CCASSG,DinkNet,GAE,DAEGC,DMoN --config Photo --out results_syn
python run_each.py pyagc_run.py --dataset sweep_more,het --runs 1 --split-methods --methods SSGC,NS4GC --config HM --out results_syn
python run_each.py pyagc_run.py --dataset sweep_more,het --runs 1 --split-methods --methods DGI --out results_syn
python run_each.py train.py --dataset sweep_more,het --runs 1 --quiet --resume --tag cfgC --lam_q 2.0 --out results_syn

REM 3. One file with every run
python export_all.py
python check_coverage.py --results results_syn > coverage_syn.txt
echo Done. Send: all_runs.csv, coverage_syn.txt, graph_stats_base.txt (and run_logs\pyagc_run\FAILED.txt and TIMEOUT.txt if they exist)
pause
