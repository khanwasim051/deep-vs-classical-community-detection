REM Step 12: input-controlled comparisons and the K-unknown setting (reviewer concern 2).
REM Run from the (mixflow) prompt in D:\mixflow_cd. Safe to stop and rerun: finished runs are skipped.

REM 1. Label-free estimate of K for every graph (eigengap), saved to khat.json
python estimate_k.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --out khat.json > khat.txt

REM 2. Known K, extra baselines: Leiden tuned to K communities, k-means with matched feature dimension, joint spectral+attribute baseline
python run_each.py baselines.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --methods LeidenK,KMeans-featM,SpecAttr --runs 5 --resume --out results_ctrl

REM 3. Topology-only control on the attributed graphs: six reference methods given structural features instead of attributes
python run_each.py pyagc_run.py --dataset cora,citeseer,pubmed,amazon-photo --runs 5 --split-methods --methods MS2CAG,SSGC,DGI,CCASSG,NS4GC,GAE --features structural --out results_ctrl

REM 4. Unknown K: every method that needs K now gets K_hat instead of the true K
python run_each.py baselines.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --methods RegSpectral,KMeans-feat,SpecAttr --runs 5 --resume --k_hat khat.json --out results_khat
python run_each.py train.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --runs 5 --quiet --resume --tag khat --k_hat khat.json --out results_khat
python run_each.py pyagc_run.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --runs 5 --split-methods --methods MS2CAG,SSGC,DGI,CCASSG,NS4GC,DinkNet,GAE,DAEGC,DMoN --k_hat khat.json --out results_khat

REM 5. Summaries to send back
python check_coverage.py --results results_ctrl > coverage_ctrl.txt
python check_coverage.py --results results_khat > coverage_khat.txt
python stats.py --results results,results_pyagc,results_ctrl > stats_ctrl.txt
python stats.py --results results,results_khat > stats_khat.txt
echo Done. Send: khat.txt, stats_ctrl.txt, stats_khat.txt, coverage_ctrl.txt, coverage_khat.txt
pause
