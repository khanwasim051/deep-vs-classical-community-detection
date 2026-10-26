REM Step 18: second label-free estimator of the number of communities (reviewer issue 4). About 1-2 hours.
REM Run from the (mixflow) prompt in D:\mixflow_cd. Safe to stop and rerun.

REM 1. Bethe Hessian estimate of K for the 15 original graphs, saved to khat_bethe.json
python estimate_k.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --method bethe --out khat_bethe.json > khat_bethe.txt

REM 2. Every K-dependent method with this estimate (same configurations as the eigengap runs)
python run_each.py baselines.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --methods RegSpectral,KMeans-feat,SpecAttr --runs 5 --resume --k_hat khat_bethe.json --khat_label KhatBH --out results_khat_bethe
python run_each.py train.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --runs 5 --quiet --resume --tag khatBH --k_hat khat_bethe.json --out results_khat_bethe
python run_each.py pyagc_run.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --runs 5 --split-methods --methods MS2CAG,SSGC,DGI,CCASSG,NS4GC,DinkNet,GAE,DAEGC,DMoN --k_hat khat_bethe.json --khat_label KhatBH --out results_khat_bethe

REM 3. One file with every run
python export_all.py
python check_coverage.py --results results_khat_bethe > coverage_khat_bethe.txt
echo Done. Send: khat_bethe.txt, khat_bethe.json, coverage_khat_bethe.txt, all_runs.csv (and run_logs\pyagc_run\FAILED.txt, TIMEOUT.txt if they exist)
pause
